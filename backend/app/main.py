import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from app.services.vector_store import VectorStore
from app.services.confluence_connector import ConfluenceConnector
from app.core.ai_provider import AIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
vector_store = None
confluence_connector = None
ai_provider = None

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    conversation_id: Optional[str] = None

class SyncResponse(BaseModel):
    status: str
    documents_synced: int
    message: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_store, confluence_connector, ai_provider
    logger.info("[App] Starting Confluence AI Chatbot...")
    
    # Initialize services
    vector_store = VectorStore()
    await vector_store.initialize()
    
    confluence_connector = ConfluenceConnector()
    ai_provider = AIProvider()
    
    logger.info("[App] Services initialized successfully")
    yield
    # Shutdown
    logger.info("[App] Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Confluence AI Chatbot",
    description="AI chatbot for Confluence documentation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Confluence AI Chatbot API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/api/chat",
            "/api/sync",
            "/api/stats"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"[API] Received question: {request.question[:50]}...")
        
        if not vector_store or not ai_provider:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Search for relevant documents
        similar_docs = await vector_store.similarity_search(request.question, k=5)
        
        # Prepare context from documents
        context = ""
        sources = []
        for doc in similar_docs:
            context += f"\n{doc['text']}\n"
            sources.append({
                "title": doc['metadata'].get('title', 'Unknown'),
                "url": doc['metadata'].get('url', ''),
                "similarity": doc['similarity']
            })
        
        # Generate answer using AI
        answer = await ai_provider.generate_answer(
            question=request.question,
            context=context
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources[:3],  # Return top 3 sources
            conversation_id=request.conversation_id
        )
        
    except Exception as e:
        logger.error(f"[API] Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync", response_model=SyncResponse)
async def sync_confluence(space_key: Optional[str] = None):
    """Sync content from Confluence"""
    try:
        logger.info("[API] Starting Confluence sync...")
        
        if not confluence_connector or not vector_store:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Fetch pages from Confluence
        pages = await confluence_connector.get_all_pages(space_key)
        
        # Process and store each page
        synced_count = 0
        for page in pages:
            # Get page content
            content = await confluence_connector.get_page_content(page['id'])
            
            # Store in vector database
            await vector_store.add_document(
                text=content,
                metadata={
                    'title': page['title'],
                    'url': page['url'],
                    'space': page.get('space', ''),
                    'type': 'confluence_page'
                }
            )
            synced_count += 1
        
        logger.info(f"[API] Synced {synced_count} documents")
        
        return SyncResponse(
            status="success",
            documents_synced=synced_count,
            message=f"Successfully synced {synced_count} documents from Confluence"
        )
        
    except Exception as e:
        logger.error(f"[API] Error in sync: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get statistics about synced documents"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        doc_count = len(vector_store.documents) if vector_store.documents else 0
        
        return {
            "total_documents": doc_count,
            "services": {
                "vector_store": "initialized",
                "ai_provider": "initialized",
                "confluence_connector": "initialized"
            }
        }
    except Exception as e:
        logger.error(f"[API] Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)