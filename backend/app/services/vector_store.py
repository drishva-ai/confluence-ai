import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Use smaller embedding model to reduce memory usage
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Changed from full path to smaller cached version

class VectorStore:
    def __init__(self, persist_directory: str = None):
        # Use /tmp for Render compatibility (writable directory)
        if persist_directory is None:
            persist_directory = os.environ.get('DATA_DIR', '/tmp/data/embeddings')
        
        self.persist_directory = persist_directory
        self.persist_file = os.path.join(persist_directory, "documents.pkl")
        self._embedding_model = None
        self.documents = []  # List of dicts with 'text', 'metadata', 'embedding'
        
        # Create persist directory if it doesn't exist
        try:
            os.makedirs(persist_directory, exist_ok=True)
            logger.info(f"[VectorStore] Using persist directory: {persist_directory}")
        except Exception as e:
            logger.error(f"[VectorStore] Failed to create directory: {e}")
            # Fallback to current directory
            self.persist_directory = "./data/embeddings"
            os.makedirs(self.persist_directory, exist_ok=True)
        
    async def initialize(self):
        """Initialize the embedding model and load existing documents"""
        logger.info("[VectorStore] Loading embedding model: all-MiniLM-L6-v2")
        
        try:
            # Get cache directory from environment variable
            cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/tmp/sentence_transformers_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"[VectorStore] Using cache directory: {cache_dir}")
            
            # Load the embedding model with explicit cache directory
            self._embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=cache_dir
            )
            
            logger.info("[VectorStore] Embedding model loaded successfully")
            
            # Load existing documents from file if any
            if os.path.exists(self.persist_file):
                try:
                    with open(self.persist_file, 'rb') as f:
                        self.documents = pickle.load(f)
                    logger.info(f"[VectorStore] Loaded {len(self.documents)} existing documents")
                except Exception as e:
                    logger.error(f"[VectorStore] Error loading existing documents: {e}")
                    self.documents = []
            else:
                logger.info("[VectorStore] No existing documents found, starting fresh")
                
        except Exception as e:
            logger.error(f"[VectorStore] Error initializing: {e}")
            raise
    
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Add a document to the vector store"""
        if not self._embedding_model:
            logger.error("[VectorStore] Embedding model not initialized")
            return False
            
        try:
            # Generate embedding
            embedding = self._embedding_model.encode(text).tolist()
            
            # Store document
            doc = {
                'text': text,
                'metadata': metadata,
                'embedding': embedding
            }
            
            self.documents.append(doc)
            
            # Persist to disk
            await self._persist()
            
            logger.info(f"[VectorStore] Added document: {metadata.get('title', 'Untitled')}")
            return True
            
        except Exception as e:
            logger.error(f"[VectorStore] Error adding document: {e}")
            return False
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self._embedding_model or not self.documents:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self._embedding_model.encode(query)
            
            # Calculate similarities
            similarities = []
            for doc in self.documents:
                doc_embedding = np.array(doc['embedding'])
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            # Return top documents
            results = []
            for idx in top_indices:
                results.append({
                    'text': self.documents[idx]['text'],
                    'metadata': self.documents[idx]['metadata'],
                    'similarity': float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"[VectorStore] Error in similarity search: {e}")
            return []
    
    async def _persist(self):
        """Save documents to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persist_file), exist_ok=True)
            with open(self.persist_file, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.debug(f"[VectorStore] Persisted {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"[VectorStore] Error persisting documents: {e}")
    
    async def clear(self):
        """Clear all documents"""
        self.documents = []
        if os.path.exists(self.persist_file):
            os.remove(self.persist_file)
        logger.info("[VectorStore] Cleared all documents")