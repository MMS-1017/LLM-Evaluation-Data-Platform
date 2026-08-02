"""
QdrantVectorStore — production-grade typed wrapper around the Qdrant client.

Responsibilities:
  - Collection lifecycle (create if not exists)
  - Batch upsert with stable integer IDs
  - Similarity search with score threshold
  - Health check for API startup
"""
from __future__ import annotations

import uuid
from typing import Any

from src.common.exceptions import VectorStoreError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings

log = get_logger(__name__)


class QdrantVectorStore:
    """
    Typed wrapper around qdrant_client.QdrantClient.

    All operations are idempotent where possible (ensure_collection,
    upsert) to make pipelines safe to re-run.

    Example:
        store = QdrantVectorStore()
        store.ensure_collection(dimension=384)
        store.upsert(ids=["abc"], vectors=[[0.1]*384], payloads=[{"content": "..."}])
        results = store.search(query_vector=[0.1]*384, top_k=5)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    # ── Client management ──────────────────────────────────────────────────────

    def _get_client(self):
        """Lazily initialize and return the Qdrant client."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            ) from exc

        try:
            self._client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                api_key=self.settings.qdrant_api_key or None,
                timeout=30,
            )
            log.info(
                f"Qdrant client connected to "
                f"{self.settings.qdrant_host}:{self.settings.qdrant_port}"
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to connect to Qdrant",
                details={
                    "host": self.settings.qdrant_host,
                    "port": self.settings.qdrant_port,
                    "error": str(exc),
                },
            ) from exc

        return self._client

    # ── Collection lifecycle ───────────────────────────────────────────────────

    def ensure_collection(self, dimension: int) -> None:
        """
        Create the configured collection if it does not already exist.
        Uses cosine distance (suitable for normalized embedding vectors).
        """
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        name = self.settings.qdrant_collection

        existing_names = [c.name for c in client.get_collections().collections]
        if name not in existing_names:
            log.info(f"Creating Qdrant collection '{name}' (dim={dimension})")
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            log.info(f"Collection '{name}' created")
        else:
            log.debug(f"Collection '{name}' already exists — skipping creation")

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        """
        Upsert points into the collection. IDs are converted to stable
        unsigned integers using SHA-256-based modular arithmetic.

        Args:
            ids:      String chunk IDs.
            vectors:  Embedding vectors, one per ID.
            payloads: Metadata dicts stored alongside vectors in Qdrant.
        """
        from qdrant_client.models import PointStruct

        client = self._get_client()
        points = [
            PointStruct(
                id=self._str_to_uint(chunk_id),
                vector=vec,
                payload=payload,
            )
            for chunk_id, vec, payload in zip(ids, vectors, payloads)
        ]

        try:
            client.upsert(collection_name=self.settings.qdrant_collection, points=points)
        except Exception as exc:
            raise VectorStoreError(
                "Upsert failed",
                details={"n_points": len(points), "error": str(exc)},
            ) from exc

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Perform cosine similarity search.

        Returns:
            List of payload dicts, each augmented with a 'score' key,
            ordered by descending similarity.
        """
        client = self._get_client()
        try:
            hits = client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            return [{"score": hit.score, **(hit.payload or {})} for hit in hits]
        except Exception as exc:
            raise VectorStoreError(
                "Search failed",
                details={"top_k": top_k, "error": str(exc)},
            ) from exc

    # ── Observability ──────────────────────────────────────────────────────────

    def get_collection_info(self) -> dict[str, Any]:
        """Return collection statistics for health checks and monitoring."""
        client = self._get_client()
        try:
            info = client.get_collection(self.settings.qdrant_collection)
            return {
                "name": self.settings.qdrant_collection,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception as exc:
            return {
                "name": self.settings.qdrant_collection,
                "error": str(exc),
            }

    def health_check(self) -> bool:
        """Return True if Qdrant is reachable, False otherwise."""
        try:
            self._get_client().get_collections()
            return True
        except Exception:
            return False

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _str_to_uint(chunk_id: str) -> int:
        """Convert a hex or arbitrary string chunk ID to a stable unsigned int."""
        try:
            return int(chunk_id, 16) % (2**63)
        except (ValueError, TypeError):
            # Fall back to UUID v5 integer for non-hex IDs
            return uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)).int % (2**63)
