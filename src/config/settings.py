"""
Typed application settings loaded from environment variables and .env files.

Usage:
    from src.config.settings import get_settings
    s = get_settings()
    print(s.qdrant_host)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project paths ──────────────────────────────────────────────────────────
    project_root: Path = Field(default_factory=lambda: Path(__file__).parents[3])
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_bronze_dir: Path = Field(default=Path("data/bronze"))
    data_silver_dir: Path = Field(default=Path("data/silver"))
    data_gold_dir: Path = Field(default=Path("data/gold"))

    # ── Spark ──────────────────────────────────────────────────────────────────
    spark_master: str = Field(default="local[*]")
    spark_app_name: str = Field(default="LLMEvalPlatform")
    spark_executor_memory: str = Field(default="2g")
    spark_driver_memory: str = Field(default="2g")

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["huggingface", "openai"] = Field(default="huggingface")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_batch_size: int = Field(default=64)
    embedding_dimension: int = Field(default=384)

    # ── LLM (generation) ───────────────────────────────────────────────────────
    llm_provider: Literal["huggingface", "openai"] = Field(default="huggingface")
    llm_model: str = Field(default="google/flan-t5-base")
    openai_api_key: str = Field(default="")

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="llm_eval_docs")
    qdrant_api_key: str = Field(default="")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="llm_eval_platform")

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    min_chunk_length: int = Field(default=50)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    eval_test_set_size: int = Field(default=20)
    eval_metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    )

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    api_top_k: int = Field(default=5)

    def resolve_path(self, p: Path) -> Path:
        """Resolve a relative path against the project root."""
        if p.is_absolute():
            return p
        return self.project_root / p


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
