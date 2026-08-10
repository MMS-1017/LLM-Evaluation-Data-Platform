"""
Unit tests for src/models/document.py.

Validates RawDocument and ProcessedChunk construction,
field validation, and class-method helpers.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from src.models.document import EmbeddedChunk, ProcessedChunk, RawDocument

pytestmark = pytest.mark.unit


# ── RawDocument ────────────────────────────────────────────────────────────────

class TestRawDocument:
    def test_construction(self, raw_document: RawDocument):
        assert raw_document.doc_id == "abc123def456abc1"
        assert raw_document.file_type == "txt"
        assert raw_document.byte_size == 200
        assert isinstance(raw_document.ingested_at, datetime)

    def test_from_file(self, tmp_path: Path):
        txt = tmp_path / "test.txt"
        txt.write_text("Hello, world! This is test content.", encoding="utf-8")
        doc = RawDocument.from_file(txt)

        expected_hash = hashlib.sha256("Hello, world! This is test content.".encode()).hexdigest()
        assert doc.doc_id == expected_hash
        assert doc.source == str(txt)
        assert doc.file_type == "txt"
        assert doc.content == "Hello, world! This is test content."
        assert doc.byte_size > 0

    def test_from_file_no_extension(self, tmp_path: Path):
        f = tmp_path / "noext"
        f.write_text("content", encoding="utf-8")
        doc = RawDocument.from_file(f)
        assert doc.file_type == "txt"  # Default fallback

    def test_doc_id_is_content_addressable(self):
        """Two documents with identical content should produce the same doc_id."""
        content = "Identical content for both documents."
        doc1 = RawDocument(doc_id=hashlib.sha256(content.encode()).hexdigest(),
                           source="a.txt", content=content, file_type="txt", byte_size=100)
        doc2 = RawDocument(doc_id=hashlib.sha256(content.encode()).hexdigest(),
                           source="b.txt", content=content, file_type="txt", byte_size=100)
        assert doc1.doc_id == doc2.doc_id

    def test_metadata_defaults_to_empty_dict(self):
        doc = RawDocument(doc_id="x" * 64, source="f.txt",
                          content="hi", file_type="txt", byte_size=2)
        assert doc.metadata == {}

    def test_byte_size_cannot_be_negative(self):
        with pytest.raises(Exception):
            RawDocument(doc_id="x" * 64, source="f.txt",
                        content="hi", file_type="txt", byte_size=-1)


# ── ProcessedChunk ─────────────────────────────────────────────────────────────

class TestProcessedChunk:
    def test_construction(self, processed_chunk: ProcessedChunk):
        assert processed_chunk.chunk_index == 0
        assert processed_chunk.char_start == 0
        assert processed_chunk.char_end == 128
        assert 0.0 <= processed_chunk.quality_score <= 1.0

    def test_build_id_is_deterministic(self):
        id1 = ProcessedChunk.build_id("doc_abc", 0)
        id2 = ProcessedChunk.build_id("doc_abc", 0)
        assert id1 == id2

    def test_build_id_length(self):
        cid = ProcessedChunk.build_id("some_doc_id", 3)
        assert len(cid) == 16

    def test_build_id_differs_for_different_index(self):
        id0 = ProcessedChunk.build_id("doc_abc", 0)
        id1 = ProcessedChunk.build_id("doc_abc", 1)
        assert id0 != id1

    def test_build_id_differs_for_different_doc(self):
        id_a = ProcessedChunk.build_id("doc_a", 0)
        id_b = ProcessedChunk.build_id("doc_b", 0)
        assert id_a != id_b

    def test_quality_score_must_be_in_range(self):
        with pytest.raises(Exception):
            ProcessedChunk(
                chunk_id="abc",
                doc_id="xyz",
                source="f.txt",
                content="text",
                chunk_index=0,
                char_start=0,
                char_end=4,
                token_count=1,
                quality_score=1.5,  # > 1.0
            )

    def test_quality_score_lower_bound(self):
        with pytest.raises(Exception):
            ProcessedChunk(
                chunk_id="abc",
                doc_id="xyz",
                source="f.txt",
                content="text",
                chunk_index=0,
                char_start=0,
                char_end=4,
                token_count=1,
                quality_score=-0.1,  # < 0.0
            )

    def test_chunk_index_cannot_be_negative(self):
        with pytest.raises(Exception):
            ProcessedChunk(
                chunk_id="abc",
                doc_id="xyz",
                source="f.txt",
                content="text",
                chunk_index=-1,
                char_start=0,
                char_end=4,
                token_count=1,
            )

    def test_default_quality_score_is_one(self):
        chunk = ProcessedChunk(
            chunk_id="abc",
            doc_id="xyz",
            source="f.txt",
            content="text content here",
            chunk_index=0,
            char_start=0,
            char_end=17,
            token_count=3,
        )
        assert chunk.quality_score == 1.0


# ── EmbeddedChunk ──────────────────────────────────────────────────────────────

class TestEmbeddedChunk:
    def test_construction(self):
        embedding = [0.1] * 384
        chunk = EmbeddedChunk(
            chunk_id="abc123",
            doc_id="doc456",
            source="data/raw/test.txt",
            content="Some embedded text content.",
            embedding=embedding,
            embedding_model="all-MiniLM-L6-v2",
        )
        assert chunk.embedding_model == "all-MiniLM-L6-v2"
        assert len(chunk.embedding) == 384
        assert isinstance(chunk.indexed_at, datetime)

    def test_metadata_defaults_to_empty(self):
        chunk = EmbeddedChunk(
            chunk_id="x",
            doc_id="y",
            source="f.txt",
            content="content",
            embedding=[0.0],
            embedding_model="test-model",
        )
        assert chunk.metadata == {}
