#!/usr/bin/env python3
"""
Entry point for the embedding and indexing pipeline (Gold → Qdrant).

Usage:
    python -m pipelines.embeddings.run_embeddings
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.common.logging import get_logger, setup_logger
from src.config.settings import get_settings

setup_logger(level="INFO", log_file=Path("logs/embeddings.log"))
log = get_logger(__name__)


def run() -> int:
    """Execute embedding + indexing pipeline. Returns 0 on success."""
    settings = get_settings()
    log.info("=" * 60)
    log.info("Starting embedding and indexing pipeline (Gold → Qdrant)")
    log.info(f"  Model     : {settings.embedding_provider}/{settings.embedding_model}")
    log.info(f"  Qdrant    : {settings.qdrant_host}:{settings.qdrant_port}")
    log.info(f"  Collection: {settings.qdrant_collection}")
    log.info("=" * 60)

    try:
        from pipelines.embeddings.indexer import VectorIndexer

        indexer = VectorIndexer(settings=settings)
        count = indexer.run()
        log.info(f"Embedding pipeline complete: {count} chunks indexed")
        return 0
    except Exception as exc:
        log.exception(f"Embedding pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
