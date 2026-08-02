#!/usr/bin/env python3
"""
Entry point for the ingestion pipeline (Raw → Bronze).

Usage:
    python -m pipelines.ingestion.run_ingestion
    # or
    python pipelines/ingestion/run_ingestion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.common.logging import get_logger, setup_logger
from src.config.settings import get_settings

setup_logger(level="INFO", log_file=Path("logs/ingestion.log"))
log = get_logger(__name__)


def run() -> int:
    """Execute ingestion pipeline. Returns 0 on success, 1 on failure."""
    settings = get_settings()
    log.info("=" * 60)
    log.info("Starting ingestion pipeline (Raw → Bronze)")
    log.info(f"  Raw dir   : {settings.resolve_path(settings.data_raw_dir)}")
    log.info(f"  Bronze dir: {settings.resolve_path(settings.data_bronze_dir)}")
    log.info("=" * 60)

    try:
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter

        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)

        documents = loader.load_directory()

        if not documents:
            log.error("No documents found in the raw directory — nothing to ingest")
            return 1

        output_path = writer.write(documents)
        log.info(f"Ingestion complete: {len(documents)} documents → {output_path}")
        return 0

    except Exception as exc:
        log.exception(f"Ingestion pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
