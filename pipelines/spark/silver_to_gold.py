#!/usr/bin/env python3
"""
PySpark job: Silver → Gold.

Transformations applied:
  1. Quality filter (score >= MIN_QUALITY_SCORE)
  2. Token-count filter (count >= MIN_TOKEN_COUNT)
  3. Deduplication by chunk_id
  4. Select and sort final columns for embedding pipeline

Usage:
    python -m pipelines.spark.silver_to_gold
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.common.exceptions import SparkJobError
from src.common.logging import get_logger, setup_logger
from src.config.settings import Settings, get_settings

setup_logger(level="INFO", log_file=Path("logs/spark_silver_to_gold.log"))
log = get_logger(__name__)

MIN_QUALITY_SCORE = 0.30
MIN_TOKEN_COUNT = 20


def run(spark=None, settings: Settings | None = None) -> int:
    """Execute Silver → Gold job. Returns 0 on success."""
    settings = settings or get_settings()
    from pipelines.spark.session import SparkSessionFactory
    from pyspark.sql import functions as F

    factory = SparkSessionFactory(settings)
    spark_created = spark is None

    try:
        if spark is None:
            spark = factory.create()

        silver_dir = settings.resolve_path(settings.data_silver_dir)
        gold_dir = settings.resolve_path(settings.data_gold_dir)

        log.info("=" * 60)
        log.info("PySpark job: Silver → Gold")
        log.info(f"  Silver: {silver_dir}")
        log.info(f"  Gold  : {gold_dir}")
        log.info(f"  Min quality score : {MIN_QUALITY_SCORE}")
        log.info(f"  Min token count   : {MIN_TOKEN_COUNT}")
        log.info("=" * 60)

        # ── Read ────────────────────────────────────────────────────────────────
        silver_df = spark.read.parquet(str(silver_dir))
        n_silver = silver_df.count()
        log.info(f"Silver records: {n_silver}")

        # ── Quality filter ──────────────────────────────────────────────────────
        quality_filtered = silver_df.filter(
            (F.col("quality_score") >= MIN_QUALITY_SCORE)
            & (F.col("token_count") >= MIN_TOKEN_COUNT)
        )
        n_quality = quality_filtered.count()
        log.info(f"After quality filter: {n_quality} chunks")

        # ── Deduplicate by chunk_id ─────────────────────────────────────────────
        deduped = quality_filtered.dropDuplicates(["chunk_id"])
        n_deduped = deduped.count()
        log.info(f"After chunk deduplication: {n_deduped} chunks")

        # ── Select gold columns ─────────────────────────────────────────────────
        gold_df = (
            deduped
            .select(
                "chunk_id",
                "doc_id",
                "source",
                "content",
                "token_count",
                "quality_score",
                "file_type",
                "metadata",
            )
            .orderBy("doc_id", "chunk_id")
        )

        # ── Write ───────────────────────────────────────────────────────────────
        gold_dir.mkdir(parents=True, exist_ok=True)
        gold_df.write.mode("overwrite").parquet(str(gold_dir))

        n_gold = gold_df.count()
        log.info(f"Silver → Gold complete: {n_gold} chunks written to {gold_dir}")
        return 0

    except Exception as exc:
        log.exception(f"Silver → Gold job failed: {exc}")
        raise SparkJobError(
            "Silver to Gold transformation failed",
            details={"error": str(exc)},
        ) from exc
    finally:
        if spark_created and spark is not None:
            factory.stop(spark)


if __name__ == "__main__":
    sys.exit(run())
