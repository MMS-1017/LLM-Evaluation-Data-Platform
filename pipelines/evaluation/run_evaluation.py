#!/usr/bin/env python3
"""
Entry point for the RAG evaluation pipeline.

Steps:
  1. Generate synthetic test set from Gold layer
  2. Run Ragas evaluation (or heuristic fallback)
  3. Log results to MLflow

Usage:
    python -m pipelines.evaluation.run_evaluation
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.common.logging import get_logger, setup_logger
from src.config.settings import get_settings

setup_logger(level="INFO", log_file=Path("logs/evaluation.log"))
log = get_logger(__name__)


def run() -> int:
    """Execute evaluation pipeline. Returns 0 on success."""
    settings = get_settings()
    log.info("=" * 60)
    log.info("Starting RAG evaluation pipeline")
    log.info(f"  Experiment: {settings.mlflow_experiment_name}")
    log.info(f"  MLflow    : {settings.mlflow_tracking_uri}")
    log.info(f"  Samples   : {settings.eval_test_set_size}")
    log.info("=" * 60)

    try:
        from pipelines.evaluation.test_set_generator import TestSetGenerator
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator
        from pipelines.evaluation.mlflow_tracker import MLflowTracker

        # Step 1 — Generate test set
        generator = TestSetGenerator(settings)
        samples = generator.generate()

        if not samples:
            log.error("No evaluation samples generated — aborting")
            return 1

        # Step 2 — Evaluate
        evaluator = RagasEvaluator(settings)
        report = evaluator.evaluate(samples)

        # Step 3 — Track in MLflow
        tracker = MLflowTracker(settings)
        try:
            run_id = tracker.log_report(report)
            log.info(f"MLflow run: {run_id}")
        except Exception as exc:
            log.warning(f"MLflow tracking skipped (continuing): {exc}")

        log.info("Evaluation pipeline complete")
        log.info(f"Aggregate metrics: {report.aggregate_metrics}")
        return 0

    except Exception as exc:
        log.exception(f"Evaluation pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(run())
