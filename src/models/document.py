"""
Pydantic domain models for raw documents, chunks, and embedded chunks.

Covers the first three data layers:
  - RawDocument  → Bronze layer
  - ProcessedChunk → Silver layer
  - EmbeddedChunk  → Gold layer (pre-Qdrant)
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class RawDocument(BaseModel):
    """A raw document as loaded from disk or a remote source (Bronze layer)."""

    doc_id: str = Field(description="SHA-256 of the content (auto-generated if omitted)")
    source: str = Field(description="Original file path or URL")
    content: str = Field(description="Raw text content")
    file_type: str = Field(description="File extension without dot, e.g. 'txt', 'pdf'")
    byte_size: int = Field(description="Size of raw content in bytes", ge=0)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "RawDocument":
        """Convenience constructor from a filesystem path."""
        content = path.read_text(encoding="utf-8", errors="replace")
        return cls(
            doc_id=hashlib.sha256(content.encode()).hexdigest(),
            source=str(path),
            content=content,
            file_type=path.suffix.lstrip(".").lower() or "txt",
            byte_size=path.stat().st_size,
            metadata={"filename": path.name},
        )


class ProcessedChunk(BaseModel):
    """
    A cleaned, chunked segment of a document (Silver layer).
    Each chunk maps back to its parent document via doc_id.
    """

    chunk_id: str = Field(description="Unique identifier for this chunk")
    doc_id: str = Field(description="Parent document ID")
    source: str = Field(description="Original document source path/URL")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Zero-based position within the document", ge=0)
    char_start: int = Field(description="Character offset start in the original text", ge=0)
    char_end: int = Field(description="Character offset end in the original text", ge=0)
    token_count: int = Field(description="Approximate token count", ge=0)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build_id(cls, doc_id: str, chunk_index: int) -> str:
        """Generate a deterministic 16-char chunk ID."""
        return hashlib.sha256(f"{doc_id}:{chunk_index}".encode()).hexdigest()[:16]


class EmbeddedChunk(BaseModel):
    """
    A chunk with its embedding vector ready for Qdrant indexing (Gold layer).
    """

    chunk_id: str
    doc_id: str
    source: str
    content: str
    embedding: list[float] = Field(description="Dense embedding vector")
    embedding_model: str = Field(description="Name of the model that produced this embedding")
    indexed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
