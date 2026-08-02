"""
Pydantic request/response schemas for the FastAPI service.
All schemas are strict Pydantic v2 models with field-level validation.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The natural language question to answer",
        examples=["What is a transformer model?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve from Qdrant",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = Field(description="Source document paths/URLs for retrieved chunks")
    contexts: list[str] = Field(description="Retrieved chunk texts used for generation")
    top_k: int


class IngestRequest(BaseModel):
    raw_dir: str | None = Field(
        default=None,
        description="Optional filesystem path to override the default raw data directory",
    )


class IngestResponse(BaseModel):
    status: str = Field(description="'success' | 'warning' | 'error'")
    documents_ingested: int
    message: str


class HealthResponse(BaseModel):
    status: str = Field(description="'healthy' | 'degraded'")
    qdrant: dict[str, Any] = Field(description="Qdrant collection stats")
    version: str = "0.1.0"
