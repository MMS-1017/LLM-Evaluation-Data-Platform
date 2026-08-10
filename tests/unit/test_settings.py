"""
Unit tests for src/config/settings.py.

Validates that Settings loads defaults correctly, resolve_path works,
and that required fields enforce their types.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import Settings, get_settings

pytestmark = pytest.mark.unit


class TestSettingsDefaults:
    """Tests that Settings falls back to documented defaults."""

    def test_embedding_provider_default(self):
        s = Settings()
        assert s.embedding_provider == "huggingface"

    def test_embedding_model_default(self):
        s = Settings()
        assert s.embedding_model == "all-MiniLM-L6-v2"

    def test_embedding_dimension_default(self):
        s = Settings()
        assert s.embedding_dimension == 384

    def test_llm_provider_default(self):
        s = Settings()
        assert s.llm_provider == "huggingface"

    def test_llm_model_default(self):
        s = Settings()
        assert s.llm_model == "google/flan-t5-base"

    def test_qdrant_host_default(self):
        s = Settings()
        assert s.qdrant_host == "localhost"

    def test_qdrant_port_default(self):
        s = Settings()
        assert s.qdrant_port == 6333

    def test_chunk_size_default(self):
        s = Settings()
        assert s.chunk_size == 512

    def test_chunk_overlap_default(self):
        s = Settings()
        assert s.chunk_overlap == 64

    def test_api_top_k_default(self):
        s = Settings()
        assert s.api_top_k == 5

    def test_eval_metrics_default(self):
        s = Settings()
        assert "faithfulness" in s.eval_metrics
        assert "answer_relevancy" in s.eval_metrics


class TestSettingsOverride:
    """Tests that Settings can be constructed with explicit values."""

    def test_override_embedding_provider(self):
        s = Settings(embedding_provider="openai")
        assert s.embedding_provider == "openai"

    def test_override_chunk_size(self):
        s = Settings(chunk_size=256)
        assert s.chunk_size == 256

    def test_override_qdrant_collection(self):
        s = Settings(qdrant_collection="my_collection")
        assert s.qdrant_collection == "my_collection"

    def test_invalid_embedding_provider_raises(self):
        with pytest.raises(Exception):
            Settings(embedding_provider="invalid_provider")

    def test_invalid_llm_provider_raises(self):
        with pytest.raises(Exception):
            Settings(llm_provider="invalid_llm")


class TestResolvePath:
    """Tests for Settings.resolve_path()."""

    def test_relative_path_resolved_against_project_root(self):
        s = Settings()
        relative = Path("data/raw")
        resolved = s.resolve_path(relative)
        assert resolved.is_absolute()
        assert resolved.name == "raw"
        assert "data" in resolved.parts

    def test_absolute_path_returned_unchanged(self):
        s = Settings()
        absolute = Path("C:/some/absolute/path") if __import__("os").name == "nt" else Path("/tmp/test")
        resolved = s.resolve_path(absolute)
        assert resolved == absolute

    def test_data_raw_resolves(self):
        s = Settings()
        raw = s.resolve_path(s.data_raw_dir)
        assert raw.is_absolute()

    def test_data_bronze_resolves(self):
        s = Settings()
        bronze = s.resolve_path(s.data_bronze_dir)
        assert bronze.is_absolute()


class TestGetSettings:
    """Tests for the get_settings() singleton factory."""

    def test_returns_settings_instance(self):
        """get_settings() should always return a fully initialised Settings object."""
        s = get_settings()
        assert isinstance(s, Settings)
        # Spot-check a few fields to confirm it's usable
        assert isinstance(s.qdrant_host, str)
        assert isinstance(s.qdrant_port, int)
        assert isinstance(s.chunk_size, int)

    def test_returns_same_instance_on_repeated_calls(self):
        """Singleton: a second call within the same process returns the cached instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_project_root_is_absolute(self):
        """project_root should always resolve to an absolute path."""
        s = get_settings()
        assert s.project_root.is_absolute()
