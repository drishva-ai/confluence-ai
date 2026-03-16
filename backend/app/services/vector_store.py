"""
Vector Store Service
=====================
Takes documents from Confluence and:
1. Splits them into chunks (small pieces)
2. Converts each chunk to a vector (numbers)
3. Stores in Qdrant vector database
4. Searches for relevant chunks when user asks a question

Uses sentence-transformers for embeddings — completely FREE, runs locally.
No OpenAI embedding costs ever.
"""

import os
import logging
import hashlib
from typing import List, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "confluence_docs")


@dataclass
class DocumentChunk:
    """A small piece of a document ready for vector storage."""
    chunk_id:    str
    doc_id:      str
    title:       str
    content:     str
    source_url:  str
    doc_type:    str
    chunk_index: int


class VectorStoreService:
    """
    Manages the vector database — store and search documents.
    
    Usage:
        store = VectorStoreService()
        await store.initialize()
        await store.index_documents(documents)
        results = await store.search("how to reset password", top_k=5)
    """

    def __init__(self):
        self._embedding_model = None
        self._qdrant_client   = None

    async def initialize(self):
        """Load embedding model and connect to Qdrant."""
        log.info(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer
        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        log.info(f"[VectorStore] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        # Create collection if it doesn't exist
        existing = [c.name for c in self._qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            # Get embedding dimension from model
            sample_embedding = self._embedding_model.encode(["test"])
            dimension = len(sample_embedding[0])

            self._qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            log.info(f"[VectorStore] Created collection: {COLLECTION_NAME} (dim={dimension})")
        else:
            log.info(f"[VectorStore] Using existing collection: {COLLECTION_NAME}")

    # ── Indexing ──────────────────────────────────────────────────────────────

    async def index_documents(self, documents) -> int:
        """
        Takes Confluence documents, chunks them, embeds them, stores in Qdrant.
        Returns total number of chunks stored.
        """
        from qdrant_client.models import PointStruct

        all_chunks = []
        for doc in documents:
            chunks = self._split_into_chunks(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            log.warning("[VectorStore] No chunks to index")
            return 0

        log.info(f"[VectorStore] Embedding {len(all_chunks)} chunks...")

        # Embed all chunks in one batch (fast)
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Store in Qdrant
        points = []
        for chunk, embedding in zip(all_chunks, embeddings):
            points.append(PointStruct(
                id      = abs(hash(chunk.chunk_id)) % (2**63),
                vector  = embedding.tolist(),
                payload = {
                    "chunk_id":   chunk.chunk_id,
                    "doc_id":     chunk.doc_id,
                    "title":      chunk.title,
                    "content":    chunk.content,
                    "source_url": chunk.source_url,
                    "doc_type":   chunk.doc_type,
                    "chunk_index": chunk.chunk_index,
                }
            ))

        # Upload in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )

        log.info(f"[VectorStore] Indexed {len(all_chunks)} chunks successfully")
        return len(all_chunks)

    # ── Searching ─────────────────────────────────────────────────────────────

    async def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Finds the most relevant document chunks for a query.
        
        How it works:
        1. Convert query to a vector
        2. Find top_k most similar vectors in Qdrant
        3. Return the text of those chunks
        
        This is much better than keyword search because it
        understands meaning, not just exact words.
        """
        # Convert query to vector
        query_embedding = self._embedding_model.encode(
            [query], normalize_embeddings=True
        )[0]

        # Search Qdrant
        results = self._qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=0.3,   # minimum similarity score
        )

        # Format results
        chunks = []
        for result in results:
            chunks.append({
                "content":    result.payload["content"],
                "title":      result.payload["title"],
                "source_url": result.payload["source_url"],
                "doc_type":   result.payload["doc_type"],
                "score":      round(result.score, 3),
            })

        return chunks

    async def get_stats(self) -> dict:
        """Returns statistics about the indexed content."""
        info = self._qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "total_chunks": info.points_count,
            "collection":   COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
        }

    async def clear_collection(self):
        """Deletes all indexed content. Used before a full re-sync."""
        self._qdrant_client.delete_collection(COLLECTION_NAME)
        await self.initialize()
        log.info("[VectorStore] Collection cleared and recreated")

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _split_into_chunks(self, doc) -> List[DocumentChunk]:
        """
        Splits a document into overlapping chunks.
        
        Why chunks?
        LLMs have a context limit (e.g. 4000 tokens).
        A Confluence space might have 10MB of text.
        We split into small pieces, find the relevant ones,
        then only pass those to the LLM.
        
        Why overlap?
        So sentences at chunk boundaries are not cut off.
        """
        content = doc.content.strip()
        if not content:
            return []

        chunks  = []
        start   = 0
        index   = 0

        while start < len(content):
            end = start + CHUNK_SIZE

            # Try to break at a sentence boundary
            if end < len(content):
                # Look for period, newline in last 200 chars
                break_point = content.rfind("\n", start, end)
                if break_point == -1:
                    break_point = content.rfind(". ", start, end)
                if break_point > start:
                    end = break_point + 1

            chunk_text = content[start:end].strip()
            if chunk_text:
                # Add title prefix for context
                full_text = f"Source: {doc.title}\n\n{chunk_text}"

                chunk_id = hashlib.md5(
                    f"{doc.id}_{index}".encode()
                ).hexdigest()

                chunks.append(DocumentChunk(
                    chunk_id    = chunk_id,
                    doc_id      = doc.id,
                    title       = doc.title,
                    content     = full_text,
                    source_url  = doc.source_url,
                    doc_type    = doc.doc_type,
                    chunk_index = index,
                ))
                index += 1

            start = end - CHUNK_OVERLAP
            if start >= len(content):
                break

        return chunks
