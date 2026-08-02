#!/usr/bin/env python3
"""
Local pipeline runner — executes all stages sequentially without Airflow.

Useful for development, integration testing, and quick end-to-end runs.

Usage:
    python -m pipelines.orchestration.pipeline_runner
    python -m pipelines.orchestration.pipeline_runner --skip-spark
    python -m pipelines.orchestration.pipeline_runner --only-eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.common.logging import get_logger, setup_logger
from src.config.settings import get_settings

setup_logger(level="INFO", log_file=Path("logs/pipeline_runner.log"))
log = get_logger(__name__)


def run_stage(name: str, fn) -> bool:
    """Run a pipeline stage, return True on success."""
    log.info(f"{'─' * 50}")
    log.info(f"Stage: {name}")
    log.info(f"{'─' * 50}")
    try:
        exit_code = fn()
        if exit_code == 0:
            log.info(f"✓ {name} succeeded")
            return True
        else:
            log.error(f"✗ {name} failed (exit code {exit_code})")
            return False
    except Exception as exc:
        log.exception(f"✗ {name} raised an exception: {exc}")
        return False


def main(args: argparse.Namespace) -> int:
    settings = get_settings()
    log.info("=" * 60)
    log.info("LLM Evaluation Data Platform — Full Pipeline Run")
    log.info("=" * 60)

    failures: list[str] = []

    if not args.skip_ingest:
        from pipelines.ingestion.run_ingestion import run as ingest
        if not run_stage("Ingestion (Raw → Bronze)", ingest):
            failures.append("Ingestion")
            if args.fail_fast:
                return 1

    if not args.skip_spark:
        from pipelines.spark.bronze_to_silver import run as b2s
        if not run_stage("Spark: Bronze → Silver", b2s):
            failures.append("Bronze→Silver")
            if args.fail_fast:
                return 1

        from pipelines.spark.silver_to_gold import run as s2g
        if not run_stage("Spark: Silver → Gold", s2g):
            failures.append("Silver→Gold")
            if args.fail_fast:
                return 1

    if not args.skip_embed:
        from pipelines.embeddings.run_embeddings import run as embed
        if not run_stage("Embedding + Indexing (Gold → Qdrant)", embed):
            failures.append("Embedding")
            if args.fail_fast:
                return 1

    if not args.skip_eval:
        from pipelines.evaluation.run_evaluation import run as evaluate
        if not run_stage("RAG Evaluation (Ragas + MLflow)", evaluate):
            failures.append("Evaluation")

    log.info("=" * 60)
    if failures:
        log.error(f"Pipeline finished with failures: {failures}")
        return 1
    else:
        log.info("All pipeline stages completed successfully")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local pipeline runner")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion stage")
    parser.add_argument("--skip-spark", action="store_true", help="Skip both Spark jobs")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding+indexing")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--fail-fast", action="store_true", default=True,
                        help="Abort on first stage failure (default: True)")
    parser.add_argument("--only-eval", action="store_true",
                        help="Shortcut: run only the evaluation stage")
    args = parser.parse_args()

    if args.only_eval:
        args.skip_ingest = args.skip_spark = args.skip_embed = True

    sys.exit(main(args))
