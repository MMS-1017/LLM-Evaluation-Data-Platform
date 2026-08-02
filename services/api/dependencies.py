"""
FastAPI dependency injection providers.

Using functools.lru_cache ensures each object is a singleton within
the process lifetime, avoiding redundant model loading on every request.
"""
from __future__ import annotations

from functools import lru_cache

from src.config.settings import get_settings


@lru_cache(maxsize=1)
def get_vector_store():
    """Singleton QdrantVectorStore instance."""
    from services.vectorstore.qdrant_store import QdrantVectorStore
    return QdrantVectorStore(get_settings())


@lru_cache(maxsize=1)
def get_embedder():
    """Singleton EmbeddingService instance (loads model on first call)."""
    from pipelines.embeddings.embedder import EmbeddingService
    return EmbeddingService(get_settings())


@lru_cache(maxsize=1)
def get_retriever():
    """Singleton RAGRetriever instance sharing the store and embedder."""
    from services.retriever.rag_retriever import RAGRetriever
    return RAGRetriever(
        vector_store=get_vector_store(),
        embedder=get_embedder(),
        settings=get_settings(),
    )
