#!/usr/bin/env python3
"""
PySpark job: Bronze → Silver.

Transformations applied:
  1. Deduplicate documents by doc_id (keep latest ingested_at)
  2. Filter documents with content shorter than 100 characters
  3. Clean text (normalize whitespace, remove boilerplate)
  4. Chunk documents using a sliding window
  5. Score chunk quality
  6. Write results to the Silver Parquet layer

Usage:
    python -m pipelines.spark.bronze_to_silver
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from src.common.exceptions import SparkJobError
from src.common.logging import get_logger, setup_logger
from src.config.settings import Settings, get_settings

setup_logger(level="INFO", log_file=Path("logs/spark_bronze_to_silver.log"))
log = get_logger(__name__)

# ── UDF return type ────────────────────────────────────────────────────────────
# Must be importable at module level so Spark can serialize it.
def _make_chunk_udf(chunk_size: int, overlap: int, min_length: int):
    """
    Return a PySpark UDF that chunks a document and returns a list of chunk dicts.
    Inner function is defined here to capture parameters via closure.
    """
    import hashlib

    from pyspark.sql.functions import udf
    from pyspark.sql.types import (
        ArrayType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    RETURN_TYPE = ArrayType(
        StructType([
            StructField("chunk_id", StringType()),
            StructField("content", StringType()),
            StructField("chunk_index", IntegerType()),
            StructField("char_start", IntegerType()),
            StructField("char_end", IntegerType()),
            StructField("token_count", IntegerType()),
            StructField("quality_score", DoubleType()),
        ])
    )

    def _chunk_fn(doc_id: str, content: str) -> list[dict]:
        # Import inside UDF — Spark workers need these independently
        import re
        import unicodedata

        def _normalize(t: str) -> str:
            t = unicodedata.normalize("NFKD", t)
            t = re.sub(r"\s+", " ", t)
            return t.strip()

        def _remove_boilerplate(t: str) -> str:
            t = re.sub(r"^\s*\d+\s*$", "", t, flags=re.MULTILINE)
            t = re.sub(r"^[\-=_*#~]{5,}\s*$", "", t, flags=re.MULTILINE)
            return t.strip()

        def _quality(t: str) -> float:
            if not t or len(t) < 10:
                return 0.0
            alpha = sum(c.isalpha() for c in t) / len(t)
            words = t.lower().split()
            uniq = len(set(words)) / len(words) if words else 0.0
            return round(min(max(alpha * 0.6 + uniq * 0.4, 0.0), 1.0), 4)

        def _token_count(t: str) -> int:
            return max(1, int(len(t.split()) * 1.3))

        if not content or not doc_id:
            return []

        cleaned = _remove_boilerplate(_normalize(content))
        step = max(1, chunk_size - overlap)
        chunks = []
        start = 0

        while start < len(cleaned):
            end = min(start + chunk_size, len(cleaned))
            chunk_text = cleaned[start:end].strip()
            if len(chunk_text) >= min_length:
                idx = len(chunks)
                cid = hashlib.sha256(f"{doc_id}:{idx}".encode()).hexdigest()[:16]
                chunks.append({
                    "chunk_id": cid,
                    "content": chunk_text,
                    "chunk_index": idx,
                    "char_start": start,
                    "char_end": end,
                    "token_count": _token_count(chunk_text),
                    "quality_score": float(_quality(chunk_text)),
                })
            start += step

        return chunks

    return udf(_chunk_fn, RETURN_TYPE)


def run(spark=None, settings: Settings | None = None) -> int:
    """Execute Bronze → Silver job. Returns 0 on success."""
    settings = settings or get_settings()
    from pipelines.spark.session import SparkSessionFactory
    from pyspark.sql import functions as F

    factory = SparkSessionFactory(settings)
    spark_created = spark is None

    try:
        if spark is None:
            spark = factory.create()

        bronze_dir = settings.resolve_path(settings.data_bronze_dir)
        silver_dir = settings.resolve_path(settings.data_silver_dir)

        log.info("=" * 60)
        log.info("PySpark job: Bronze → Silver")
        log.info(f"  Bronze: {bronze_dir}")
        log.info(f"  Silver: {silver_dir}")
        log.info("=" * 60)

        # ── Read ────────────────────────────────────────────────────────────────
        bronze_df = spark.read.parquet(str(bronze_dir))
        n_bronze = bronze_df.count()
        log.info(f"Bronze records: {n_bronze}")

        # ── Deduplicate ─────────────────────────────────────────────────────────
        # Keep the record with the latest ingested_at per doc_id
        from pyspark.sql.window import Window
        w = Window.partitionBy("doc_id").orderBy(F.col("ingested_at").desc())
        deduped = (
            bronze_df
            .withColumn("_rank", F.row_number().over(w))
            .filter(F.col("_rank") == 1)
            .drop("_rank")
        )
        n_deduped = deduped.count()
        log.info(f"After deduplication: {n_deduped} documents")

        # ── Filter ──────────────────────────────────────────────────────────────
        filtered = deduped.filter(F.length(F.col("content")) > 100)
        n_filtered = filtered.count()
        log.info(f"After length filter: {n_filtered} documents")

        # ── Chunk ───────────────────────────────────────────────────────────────
        chunk_udf = _make_chunk_udf(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            min_length=settings.min_chunk_length,
        )

        chunked = filtered.withColumn(
            "chunks",
            chunk_udf(F.col("doc_id"), F.col("content")),
        )

        # ── Explode ─────────────────────────────────────────────────────────────
        silver_df = (
            chunked
            .select("doc_id", "source", "file_type", "metadata", F.explode("chunks").alias("chunk"))
            .select(
                F.col("chunk.chunk_id").alias("chunk_id"),
                F.col("doc_id"),
                F.col("source"),
                F.col("chunk.content").alias("content"),
                F.col("chunk.chunk_index").alias("chunk_index"),
                F.col("chunk.char_start").alias("char_start"),
                F.col("chunk.char_end").alias("char_end"),
                F.col("chunk.token_count").alias("token_count"),
                F.col("chunk.quality_score").alias("quality_score"),
                F.lit(datetime.utcnow().isoformat()).alias("created_at"),
                F.col("file_type"),
                F.col("metadata"),
            )
        )

        # ── Write ───────────────────────────────────────────────────────────────
        silver_dir.mkdir(parents=True, exist_ok=True)
        silver_df.write.mode("overwrite").parquet(str(silver_dir))

        n_silver = silver_df.count()
        log.info(f"Bronze → Silver complete: {n_silver} chunks written to {silver_dir}")
        return 0

    except Exception as exc:
        log.exception(f"Bronze → Silver job failed: {exc}")
        raise SparkJobError(
            "Bronze to Silver transformation failed",
            details={"error": str(exc)},
        ) from exc
    finally:
        if spark_created and spark is not None:
            factory.stop(spark)


if __name__ == "__main__":
    sys.exit(run())
