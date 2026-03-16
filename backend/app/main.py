"""
Confluence AI Chatbot — Main Application
==========================================
FastAPI backend with these endpoints:

  POST /api/sync          → sync all Confluence content to vector DB
  POST /api/chat          → ask a question, get AI answer
  GET  /api/chat/stream   → streaming version (real-time typing effect)
  GET  /api/stats         → show sync status and stats
  GET  /health            → health check for deployment

How to run:
  uvicorn app.main:app --reload --port 8000
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.ai_provider import get_ai_provider, SYSTEM_PROMPT
from app.services.confluence_connector import ConfluenceConnector
from app.services.vector_store import VectorStoreService

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Shared state ─────────────────────────────────────────────────────────────
vector_store: VectorStoreService = None
ai_provider = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global vector_store, ai_provider

    log.info("[App] Starting Confluence AI Chatbot...")

    # Initialize vector store
    vector_store = VectorStoreService()
    await vector_store.initialize()

    # Initialize AI provider (reads AI_PROVIDER from .env)
    ai_provider = get_ai_provider()

    log.info("[App] Ready!")
    yield
    log.info("[App] Shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Confluence AI Chatbot",
    description="Ask questions about your Confluence documentation",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from React frontend and embeddable widget
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production: restrict to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: int = 5      # number of context chunks to retrieve

class ChatResponse(BaseModel):
    answer:   str
    sources:  list[dict]  # which pages/docs were used
    provider: str         # which AI model answered


class SyncResponse(BaseModel):
    success:      bool
    pages_synced: int
    chunks_stored: int
    message:      str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Used by Docker and Render.com to check if app is running."""
    return {"status": "healthy", "service": "confluence-ai-chatbot"}


@app.post("/api/sync", response_model=SyncResponse)
async def sync_confluence():
    """
    Fetches all content from Confluence and indexes it.
    
    This should be called:
    - Once after initial setup
    - Whenever Confluence content changes
    - Can be automated via webhook or daily cron job
    
    Takes 1-5 minutes depending on space size.
    """
    try:
        log.info("[Sync] Starting Confluence sync...")

        # 1. Fetch all documents from Confluence
        connector = ConfluenceConnector()
        documents = await connector.fetch_all_documents()

        if not documents:
            return SyncResponse(
                success=False,
                pages_synced=0,
                chunks_stored=0,
                message="No documents found in Confluence space"
            )

        # 2. Clear old data and re-index
        await vector_store.clear_collection()
        chunks_stored = await vector_store.index_documents(documents)

        msg = (
            f"Successfully synced {len(documents)} documents "
            f"({chunks_stored} chunks indexed)"
        )
        log.info(f"[Sync] {msg}")

        return SyncResponse(
            success=True,
            pages_synced=len(documents),
            chunks_stored=chunks_stored,
            message=msg
        )

    except Exception as e:
        log.error(f"[Sync] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. 
    
    Flow:
    1. Embed the question
    2. Find relevant chunks in vector DB
    3. Build prompt with context
    4. Send to AI provider
    5. Return answer + sources
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # 1. Find relevant context from vector DB
        relevant_chunks = await vector_store.search(
            request.question, top_k=request.top_k
        )

        if not relevant_chunks:
            return ChatResponse(
                answer=(
                    "I could not find any relevant information in the documentation. "
                    "Please try rephrasing your question or check if the content has been synced."
                ),
                sources=[],
                provider=f"{os.getenv('AI_PROVIDER')} / {os.getenv('AI_MODEL')}"
            )

        # 2. Build context string from top chunks
        context = "\n\n---\n\n".join([
            f"From: {chunk['title']} ({chunk['doc_type'].upper()})\n{chunk['content']}"
            for chunk in relevant_chunks
        ])

        # 3. Get AI answer
        answer = await ai_provider.chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=request.question,
            context=context,
        )

        # 4. Build sources list (deduplicated by URL)
        seen_urls = set()
        sources   = []
        for chunk in relevant_chunks:
            url = chunk["source_url"]
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title":    chunk["title"],
                    "url":      url,
                    "doc_type": chunk["doc_type"],
                    "score":    chunk["score"],
                })

        return ChatResponse(
            answer=answer,
            sources=sources,
            provider=f"{os.getenv('AI_PROVIDER')} / {os.getenv('AI_MODEL')}"
        )

    except Exception as e:
        log.error(f"[Chat] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/stream")
async def chat_stream(question: str, top_k: int = 5):
    """
    Streaming chat endpoint — tokens arrive one by one.
    Creates the "typing" effect in the UI.
    
    Usage: GET /api/chat/stream?question=How+do+I+reset+password
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def generate():
        try:
            relevant_chunks = await vector_store.search(question, top_k=top_k)

            if not relevant_chunks:
                yield "data: I could not find any relevant information.\n\n"
                return

            context = "\n\n---\n\n".join([
                f"From: {chunk['title']}\n{chunk['content']}"
                for chunk in relevant_chunks
            ])

            # Stream tokens
            async for token in ai_provider.stream_chat(
                system_prompt=SYSTEM_PROMPT,
                user_message=question,
                context=context,
            ):
                # Server-Sent Events format
                yield f"data: {token}\n\n"

            # Signal end of stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/stats")
async def get_stats():
    """Returns information about indexed content."""
    try:
        stats = await vector_store.get_stats()
        stats["ai_provider"] = os.getenv("AI_PROVIDER", "unknown")
        stats["ai_model"]    = os.getenv("AI_MODEL", "unknown")
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
