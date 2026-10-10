#!/usr/bin/env python3
"""
Entry point for the ingestion pipeline (Raw → Bronze).

Supports two modes:

* **Full ingest** (default): load every file in ``data/raw/`` and write them
  all to a new Bronze batch file.  Documents that are already recorded in the
  ingestion-state file are skipped automatically (incremental by default).

* **Force re-ingest** (``--force``): reset the state and process every file
  regardless of whether it was ingested before.

Usage::

    python -m pipelines.ingestion.run_ingestion
    python -m pipelines.ingestion.run_ingestion --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.common.logging import get_logger, setup_logger
from src.config.settings import get_settings

setup_logger(level="INFO", log_file=Path("logs/ingestion.log"))
log = get_logger(__name__)


def run(force: bool = False) -> int:
    """
    Execute the ingestion pipeline.

    Args:
        force: If True, re-ingest all files even if previously seen.

    Returns:
        0 on success, 1 on failure.
    """
    settings = get_settings()
    log.info("=" * 60)
    log.info("Starting ingestion pipeline (Raw → Bronze)")
    log.info(f"  Raw dir   : {settings.resolve_path(settings.data_raw_dir)}")
    log.info(f"  Bronze dir: {settings.resolve_path(settings.data_bronze_dir)}")
    log.info(f"  Mode      : {'full re-ingest (--force)' if force else 'incremental'}")
    log.info("=" * 60)

    try:
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter
        from pipelines.ingestion.state_tracker import IngestStateTracker

        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)
        tracker = IngestStateTracker(settings)

        # Load persisted state so we can skip already-ingested documents
        if force:
            tracker.reset()
        else:
            tracker.load()

        all_documents = loader.load_directory()

        if not all_documents:
            log.error("No documents found in the raw directory — nothing to ingest")
            return 1

        # Filter to only new/changed documents
        documents = tracker.filter_new(all_documents)

        if not documents:
            log.info(
                f"All {len(all_documents)} document(s) were already ingested. "
                "Nothing new to write. Use --force to re-ingest everything."
            )
            return 0

        output_path = writer.write(documents)

        # Persist updated state so next run skips these documents
        batch_id = output_path.stem  # e.g. "batch_20240101_120000"
        tracker.mark_ingested(documents, batch_id=batch_id)

        log.info(
            f"Ingestion complete: {len(documents)}/{len(all_documents)} "
            f"new document(s) → {output_path}"
        )
        return 0

    except Exception as exc:
        log.exception(f"Ingestion pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ingestion pipeline (Raw → Bronze)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Reset state and re-ingest all files (ignores incremental tracking).",
    )
    args = parser.parse_args()
    sys.exit(run(force=args.force))
