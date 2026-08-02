"""
EmbeddingService — generates dense text embeddings.

Supports two providers:
  - huggingface: sentence-transformers (local, no API key required)
  - openai: OpenAI Embeddings API

The service uses lazy model loading and batched encoding for efficiency.
"""
from __future__ import annotations

from typing import Iterator

from src.common.exceptions import EmbeddingError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings

log = get_logger(__name__)


class EmbeddingService:
    """
    Generates text embeddings via HuggingFace sentence-transformers or OpenAI.

    Example:
        svc = EmbeddingService()
        vectors = svc.embed(["Hello world", "AI is great"])
        # vectors: list of 384-dim float lists
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._openai_model_name: str | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts. Returns one embedding vector per input text.

        Args:
            texts: List of strings to embed. Empty inputs yield empty list.

        Returns:
            List of float lists, shape (len(texts), embedding_dim).

        Raises:
            EmbeddingError: If the model fails or is not installed.
        """
        if not texts:
            return []

        self._ensure_loaded()

        try:
            if self.settings.embedding_provider == "huggingface":
                return self._embed_hf(texts)
            else:
                return self._embed_openai(texts)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                "Embedding generation failed",
                details={"n_texts": len(texts), "error": str(exc)},
            ) from exc

    def embed_batched(self, texts: list[str]) -> Iterator[list[list[float]]]:
        """
        Yield batches of embedding vectors, consuming *texts* in chunks of
        embedding_batch_size. Use this for large-scale indexing to control
        memory usage.
        """
        bs = self.settings.embedding_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            log.debug(f"Embedding batch {i // bs + 1}/{(len(texts) + bs - 1) // bs} ({len(batch)} texts)")
            yield self.embed(batch)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return self.settings.embedding_dimension

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        provider = self.settings.embedding_provider
        model_name = self.settings.embedding_model
        log.info(f"Loading embedding model [{provider}] {model_name}")

        if provider == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

        elif provider == "openai":
            try:
                from openai import OpenAI
                self._model = OpenAI(api_key=self.settings.openai_api_key)
                self._openai_model_name = model_name
            except ImportError as exc:
                raise EmbeddingError("openai not installed. Run: pip install openai") from exc
        else:
            raise EmbeddingError(f"Unknown embedding provider: {provider!r}")

        log.info("Embedding model loaded")

    def _embed_hf(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        response = self._model.embeddings.create(
            input=texts,
            model=self._openai_model_name,
        )
        return [item.embedding for item in response.data]
