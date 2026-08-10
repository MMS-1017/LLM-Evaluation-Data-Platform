"""
Shared pytest fixtures and configuration for the LLM Evaluation Data Platform test suite.

Fixtures:
    settings        — A Settings instance loaded from test defaults (no .env override).
    sample_texts    — A small list of representative chunk texts for embedding tests.
    tmp_data_dir    — A temporary directory mimicking the data/ layout.
    raw_document    — A pre-built RawDocument for ingestion tests.
    processed_chunk — A pre-built ProcessedChunk for evaluation tests.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.models.document import ProcessedChunk, RawDocument


# ── Settings ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def settings() -> Settings:
    """
    A Settings instance using safe test defaults.
    Avoids loading any external .env file that might not exist in CI.
    """
    return Settings(
        embedding_provider="huggingface",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        llm_provider="huggingface",
        llm_model="google/flan-t5-base",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_collection="test_collection",
        mlflow_tracking_uri="http://localhost:5000",
        chunk_size=256,
        chunk_overlap=32,
        min_chunk_length=20,
        eval_test_set_size=5,
    )


# ── Temporary data directories ─────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """
    Create a temporary data directory tree that mirrors the project layout:
        tmp/data/raw/
        tmp/data/bronze/
        tmp/data/silver/
        tmp/data/gold/
    Returns the tmp root path.
    """
    for layer in ("raw", "bronze", "silver", "gold"):
        (tmp_path / "data" / layer).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_raw_dir(tmp_data_dir: Path) -> Path:
    """
    Populate the raw directory with two small .txt files for loader tests.
    Returns the path to data/raw/.
    """
    raw_dir = tmp_data_dir / "data" / "raw"
    (raw_dir / "doc_a.txt").write_text(
        "Transformers are neural network architectures that use self-attention mechanisms. "
        "They were introduced in the paper 'Attention Is All You Need' in 2017.",
        encoding="utf-8",
    )
    (raw_dir / "doc_b.txt").write_text(
        "Vector databases store high-dimensional embeddings and support fast approximate "
        "nearest-neighbour search. Popular options include Qdrant, Pinecone, and Weaviate.",
        encoding="utf-8",
    )
    return raw_dir


# ── Domain model fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def raw_document() -> RawDocument:
    """A minimal, valid RawDocument for unit tests."""
    return RawDocument(
        doc_id="abc123def456abc1",
        source="data/raw/test.txt",
        content=(
            "Large language models are trained on vast amounts of text data. "
            "They learn to predict the next token in a sequence, which enables "
            "them to generate coherent and contextually relevant text."
        ),
        file_type="txt",
        byte_size=200,
    )


@pytest.fixture
def processed_chunk(raw_document: RawDocument) -> ProcessedChunk:
    """A minimal, valid ProcessedChunk derived from raw_document."""
    return ProcessedChunk(
        chunk_id=ProcessedChunk.build_id(raw_document.doc_id, 0),
        doc_id=raw_document.doc_id,
        source=raw_document.source,
        content=raw_document.content[:128].strip(),
        chunk_index=0,
        char_start=0,
        char_end=128,
        token_count=25,
        quality_score=0.75,
    )


# ── Representative text samples ────────────────────────────────────────────────

@pytest.fixture
def sample_texts() -> list[str]:
    """Small corpus of diverse, realistic chunk texts."""
    return [
        "Transformer architectures rely on self-attention to model relationships between tokens.",
        "RAG (Retrieval-Augmented Generation) combines vector search with language model generation.",
        "Qdrant is an open-source vector database optimised for cosine similarity search.",
        "The Medallion architecture organises data into Bronze, Silver, and Gold layers.",
        "MLflow tracks experiments, parameters, metrics, and model artifacts for reproducibility.",
    ]
