"""
Typed exception hierarchy for the LLM Evaluation Data Platform.

All exceptions carry a human-readable message and an optional details dict
for structured logging and debugging.
"""
from __future__ import annotations


class LLMEvalPlatformError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class IngestionError(LLMEvalPlatformError):
    """Raised when document loading or bronze-layer writes fail."""


class SparkJobError(LLMEvalPlatformError):
    """Raised when a PySpark transformation job fails."""


class EmbeddingError(LLMEvalPlatformError):
    """Raised when embedding generation fails."""


class VectorStoreError(LLMEvalPlatformError):
    """Raised when Qdrant operations fail (connection, upsert, search)."""


class EvaluationError(LLMEvalPlatformError):
    """Raised when RAG evaluation or test-set generation fails."""


class ConfigurationError(LLMEvalPlatformError):
    """Raised for invalid or missing configuration."""


class RetrieverError(LLMEvalPlatformError):
    """Raised when the RAG retriever fails to retrieve or generate."""
