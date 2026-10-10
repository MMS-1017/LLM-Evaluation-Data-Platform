"""
Unit tests for the IngestStateTracker (pipelines/ingestion/state_tracker.py).

All tests use tmp_path and synthetic RawDocuments. No external services needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.exceptions import IngestionError
from src.config.settings import Settings
from src.models.document import RawDocument

pytestmark = pytest.mark.unit


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_doc(doc_id: str = "abc123", source: str = "data/raw/test.txt") -> RawDocument:
    return RawDocument(
        doc_id=doc_id,
        source=source,
        content="Sample content for testing the state tracker.",
        file_type="txt",
        byte_size=100,
    )


def _make_tracker(tmp_path: Path):
    """Return a tracker pointing its bronze dir at tmp_path/bronze."""
    from pipelines.ingestion.state_tracker import IngestStateTracker

    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(data_bronze_dir=bronze_dir, project_root=tmp_path)
    tracker = IngestStateTracker(settings)
    tracker._bronze_dir = bronze_dir
    tracker._state_path = bronze_dir / "ingestion_state.json"
    return tracker


# ── Load ───────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_creates_empty_state_when_no_file(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker.load()
        assert tracker.count == 0

    def test_load_reads_existing_state(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        existing = {"doc_abc": {"source": "a.txt", "ingested_at": "2024-01-01T00:00:00", "batch": "batch_1"}}
        tracker._state_path.write_text(json.dumps(existing), encoding="utf-8")

        tracker.load()
        assert tracker.count == 1
        assert tracker.is_ingested("doc_abc")

    def test_load_handles_corrupt_state_file(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state_path.write_text("NOT VALID JSON", encoding="utf-8")
        # Should not raise — falls back to empty state
        tracker.load()
        assert tracker.count == 0


# ── filter_new ─────────────────────────────────────────────────────────────────

class TestFilterNew:
    def test_all_docs_new_when_state_empty(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        docs = [_make_doc("id1"), _make_doc("id2"), _make_doc("id3")]
        result = tracker.filter_new(docs)
        assert len(result) == 3

    def test_already_ingested_docs_filtered_out(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state = {"id1": {}, "id2": {}}
        docs = [_make_doc("id1"), _make_doc("id2"), _make_doc("id3")]
        result = tracker.filter_new(docs)
        assert len(result) == 1
        assert result[0].doc_id == "id3"

    def test_all_docs_already_seen_returns_empty(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state = {"id1": {}, "id2": {}}
        docs = [_make_doc("id1"), _make_doc("id2")]
        result = tracker.filter_new(docs)
        assert result == []

    def test_empty_input_returns_empty(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        result = tracker.filter_new([])
        assert result == []


# ── mark_ingested ──────────────────────────────────────────────────────────────

class TestMarkIngested:
    def test_mark_ingested_adds_to_state(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        doc = _make_doc("new_id")
        tracker.mark_ingested([doc], batch_id="batch_20240101")
        assert tracker.is_ingested("new_id")

    def test_mark_ingested_persists_to_disk(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        doc = _make_doc("persisted_id")
        tracker.mark_ingested([doc], batch_id="batch_test")
        assert tracker._state_path.exists()
        data = json.loads(tracker._state_path.read_text())
        assert "persisted_id" in data

    def test_mark_ingested_stores_source(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        doc = _make_doc("id_x", source="data/raw/special.txt")
        tracker.mark_ingested([doc])
        data = json.loads(tracker._state_path.read_text())
        assert data["id_x"]["source"] == "data/raw/special.txt"

    def test_mark_ingested_multiple_docs(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        docs = [_make_doc(f"id_{i}") for i in range(5)]
        tracker.mark_ingested(docs)
        assert tracker.count == 5

    def test_mark_ingested_twice_does_not_duplicate(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        doc = _make_doc("dup_id")
        tracker.mark_ingested([doc])
        tracker.mark_ingested([doc])
        assert tracker.count == 1


# ── is_ingested / known_doc_ids ────────────────────────────────────────────────

class TestQueryState:
    def test_is_ingested_false_for_unknown(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        assert not tracker.is_ingested("unknown_id")

    def test_is_ingested_true_after_mark(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state["my_id"] = {}
        assert tracker.is_ingested("my_id")

    def test_known_doc_ids_returns_frozenset(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state = {"a": {}, "b": {}, "c": {}}
        ids = tracker.known_doc_ids
        assert isinstance(ids, frozenset)
        assert ids == {"a", "b", "c"}

    def test_count_matches_state_length(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state = {str(i): {} for i in range(7)}
        assert tracker.count == 7


# ── reset ──────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_in_memory_state(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state = {"id1": {}, "id2": {}}
        tracker.reset()
        assert tracker.count == 0

    def test_reset_deletes_state_file(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker._state_path.write_text("{}", encoding="utf-8")
        tracker.reset()
        assert not tracker._state_path.exists()

    def test_reset_when_no_file_does_not_raise(self, tmp_path: Path):
        tracker = _make_tracker(tmp_path)
        tracker.reset()  # Should not raise even when file doesn't exist


# ── Full round-trip ────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_incremental_flow(self, tmp_path: Path):
        """Simulates two consecutive ingestion runs."""
        tracker = _make_tracker(tmp_path)
        tracker.load()  # Empty on first run

        docs_run1 = [_make_doc("id_a"), _make_doc("id_b")]
        new_run1 = tracker.filter_new(docs_run1)
        assert len(new_run1) == 2  # Both new
        tracker.mark_ingested(new_run1, batch_id="batch_1")

        # Second run: same docs + one new
        tracker2 = _make_tracker(tmp_path)
        # Reuse same path
        tracker2._state_path = tracker._state_path
        tracker2._bronze_dir = tracker._bronze_dir
        tracker2.load()

        docs_run2 = [_make_doc("id_a"), _make_doc("id_b"), _make_doc("id_c")]
        new_run2 = tracker2.filter_new(docs_run2)
        assert len(new_run2) == 1  # Only id_c is new
        assert new_run2[0].doc_id == "id_c"
