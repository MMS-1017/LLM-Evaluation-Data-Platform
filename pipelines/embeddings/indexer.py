"""
VectorIndexer — reads Gold Parquet, embeds chunks, and upserts into Qdrant.

Processes the Gold layer in configurable batches to avoid OOM on large datasets.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.exceptions import EmbeddingError, VectorStoreError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings

log = get_logger(__name__)


class VectorIndexer:
    """
    Reads Gold-layer Parquet, generates embeddings, and upserts them into Qdrant.

    Example:
        indexer = VectorIndexer()
        n = indexer.run()
        print(f"Indexed {n} chunks")
    """

    def __init__(
        self,
        embedder=None,
        vector_store=None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        # Lazy imports to avoid circular dependency at module level
        if embedder is None:
            from pipelines.embeddings.embedder import EmbeddingService
            embedder = EmbeddingService(self.settings)
        if vector_store is None:
            from services.vectorstore.qdrant_store import QdrantVectorStore
            vector_store = QdrantVectorStore(self.settings)

        self.embedder = embedder
        self.vector_store = vector_store

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, gold_dir: Path | None = None) -> int:
        """
        Embed and index all chunks in the Gold Parquet layer.

        Returns:
            Total number of chunks indexed.

        Raises:
            EmbeddingError: If Parquet cannot be read or embeddings fail.
            VectorStoreError: If Qdrant upsert fails.
        """
        gold_path = gold_dir or self.settings.resolve_path(self.settings.data_gold_dir)
        log.info(f"Reading gold data from: {gold_path}")

        try:
            df = pd.read_parquet(str(gold_path))
        except FileNotFoundError as exc:
            raise EmbeddingError(
                f"Gold Parquet not found at {gold_path} — run the Spark jobs first"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(f"Failed to read gold data: {exc}") from exc

        if df.empty:
            log.warning("Gold layer is empty — nothing to index")
            return 0

        log.info(f"Gold chunks to embed + index: {len(df)}")

        # Ensure the Qdrant collection exists before upserting
        self.vector_store.ensure_collection(dimension=self.embedder.dimension)

        bs = self.settings.embedding_batch_size
        total = 0

        for i in range(0, len(df), bs):
            batch = df.iloc[i: i + bs]
            texts = batch["content"].tolist()

            log.info(f"Embedding batch {i // bs + 1}/{(len(df) + bs - 1) // bs} ({len(texts)} chunks)")
            embeddings = self.embedder.embed(texts)

            ids = batch["chunk_id"].tolist()
            payloads = [
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "source": row["source"],
                    "content": row["content"],
                    "quality_score": float(row.get("quality_score", 1.0)),
                }
                for _, row in batch.iterrows()
            ]

            self.vector_store.upsert(ids=ids, vectors=embeddings, payloads=payloads)
            total += len(ids)
            log.info(f"Progress: {total}/{len(df)} chunks indexed")

        log.info(f"Indexing complete: {total} chunks in Qdrant collection '{self.settings.qdrant_collection}'")
        return total
