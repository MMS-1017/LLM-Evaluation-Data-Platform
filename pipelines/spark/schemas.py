"""
PySpark StructType schemas for each data layer.

Centralizing schemas here avoids drift between the reader and writer
sides of every Spark job.
"""
from __future__ import annotations

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ── Bronze layer ──────────────────────────────────────────────────────────────
# Raw documents as loaded from disk — no transformations applied.
BRONZE_SCHEMA = StructType([
    StructField("doc_id", StringType(), nullable=False),
    StructField("source", StringType(), nullable=False),
    StructField("content", StringType(), nullable=False),
    StructField("file_type", StringType(), nullable=True),
    StructField("byte_size", LongType(), nullable=True),
    StructField("ingested_at", StringType(), nullable=True),
    StructField("metadata", StringType(), nullable=True),
])

# ── Silver layer ──────────────────────────────────────────────────────────────
# Deduplicated, cleaned, and chunked documents.
SILVER_SCHEMA = StructType([
    StructField("chunk_id", StringType(), nullable=False),
    StructField("doc_id", StringType(), nullable=False),
    StructField("source", StringType(), nullable=False),
    StructField("content", StringType(), nullable=False),
    StructField("chunk_index", IntegerType(), nullable=False),
    StructField("char_start", IntegerType(), nullable=False),
    StructField("char_end", IntegerType(), nullable=False),
    StructField("token_count", IntegerType(), nullable=False),
    StructField("quality_score", DoubleType(), nullable=False),
    StructField("created_at", StringType(), nullable=True),
    StructField("file_type", StringType(), nullable=True),
    StructField("metadata", StringType(), nullable=True),
])

# ── Gold layer ────────────────────────────────────────────────────────────────
# Quality-filtered chunks ready for embedding and indexing.
GOLD_SCHEMA = StructType([
    StructField("chunk_id", StringType(), nullable=False),
    StructField("doc_id", StringType(), nullable=False),
    StructField("source", StringType(), nullable=False),
    StructField("content", StringType(), nullable=False),
    StructField("token_count", IntegerType(), nullable=False),
    StructField("quality_score", DoubleType(), nullable=False),
    StructField("file_type", StringType(), nullable=True),
    StructField("metadata", StringType(), nullable=True),
])
