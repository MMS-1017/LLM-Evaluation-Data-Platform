"""
Integration tests for incremental ingestion (IngestStateTracker).

Tests verify the full Raw → Bronze flow with state tracking, exercising
DocumentLoader + BronzeWriter + IngestStateTracker together using the
filesystem. No Spark or Qdrant required.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_txt(directory: Path, filename: str, content: str) -> Path:
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


def _make_tracker(tmp_path: Path, settings):
    from pipelines.ingestion.state_tracker import IngestStateTracker
    from src.config.settings import Settings

    bronze_dir = tmp_path / "data" / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    custom = Settings(data_bronze_dir=bronze_dir, project_root=tmp_path)
    tracker = IngestStateTracker(custom)
    tracker._bronze_dir = bronze_dir
    tracker._state_path = bronze_dir / "ingestion_state.json"
    return tracker


# ── IngestStateTracker integration ─────────────────────────────────────────────

class TestIncrementalIngestionIntegration:
    """End-to-end incremental ingestion flow using real file I/O."""

    def test_first_run_ingests_all_files(self, tmp_data_dir: Path, settings):
        """On the first run all files must be ingested."""
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter

        raw_dir = tmp_data_dir / "data" / "raw"
        bronze_dir = tmp_data_dir / "data" / "bronze"

        _write_txt(raw_dir, "doc1.txt", "First document about transformers.")
        _write_txt(raw_dir, "doc2.txt", "Second document about vector databases.")

        tracker = _make_tracker(tmp_data_dir, settings)
        tracker.load()

        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)
        writer.bronze_dir = bronze_dir

        all_docs = loader.load_directory(raw_dir)
        new_docs = tracker.filter_new(all_docs)

        assert len(new_docs) == 2
        path = writer.write(new_docs)
        tracker.mark_ingested(new_docs, batch_id=path.stem)

        df = pd.read_parquet(path)
        assert len(df) == 2

    def test_second_run_skips_unchanged_files(self, tmp_data_dir: Path, settings):
        """Unchanged files must be skipped on the second ingestion run."""
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter

        raw_dir = tmp_data_dir / "data" / "raw"
        bronze_dir = tmp_data_dir / "data" / "bronze"

        _write_txt(raw_dir, "doc1.txt", "First document about transformers.")

        # ── Run 1 ──────────────────────────────────────────────────────────────
        tracker1 = _make_tracker(tmp_data_dir, settings)
        tracker1.load()

        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)
        writer.bronze_dir = bronze_dir

        all_docs = loader.load_directory(raw_dir)
        new_docs = tracker1.filter_new(all_docs)
        path1 = writer.write(new_docs)
        tracker1.mark_ingested(new_docs, batch_id=path1.stem)

        # ── Run 2 (same file, should skip) ────────────────────────────────────
        tracker2 = _make_tracker(tmp_data_dir, settings)
        tracker2._state_path = tracker1._state_path   # share state
        tracker2.load()

        all_docs2 = loader.load_directory(raw_dir)
        new_docs2 = tracker2.filter_new(all_docs2)
        assert len(new_docs2) == 0, "Unchanged file should have been skipped"

    def test_second_run_picks_up_new_file(self, tmp_data_dir: Path, settings):
        """A file added between runs must be ingested on the second run."""
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter

        raw_dir = tmp_data_dir / "data" / "raw"
        bronze_dir = tmp_data_dir / "data" / "bronze"

        _write_txt(raw_dir, "doc_old.txt", "Old document already in Bronze.")

        # ── Run 1 ──────────────────────────────────────────────────────────────
        tracker1 = _make_tracker(tmp_data_dir, settings)
        tracker1.load()

        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)
        writer.bronze_dir = bronze_dir

        docs1 = tracker1.filter_new(loader.load_directory(raw_dir))
        path1 = writer.write(docs1)
        tracker1.mark_ingested(docs1, batch_id=path1.stem)

        # ── Add new file ────────────────────────────────────────────────────────
        _write_txt(raw_dir, "doc_new.txt", "New document added after the first run.")

        # ── Run 2 ──────────────────────────────────────────────────────────────
        tracker2 = _make_tracker(tmp_data_dir, settings)
        tracker2._state_path = tracker1._state_path
        tracker2.load()

        all_docs2 = loader.load_directory(raw_dir)
        new_docs2 = tracker2.filter_new(all_docs2)

        assert len(new_docs2) == 1
        assert "doc_new.txt" in new_docs2[0].source

    def test_state_file_created_after_ingestion(self, tmp_data_dir: Path, settings):
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter

        raw_dir = tmp_data_dir / "data" / "raw"
        bronze_dir = tmp_data_dir / "data" / "bronze"
        _write_txt(raw_dir, "doc.txt", "Some content for state tracking test.")

        tracker = _make_tracker(tmp_data_dir, settings)
        tracker.load()

        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)
        writer.bronze_dir = bronze_dir

        docs = tracker.filter_new(loader.load_directory(raw_dir))
        path = writer.write(docs)
        tracker.mark_ingested(docs, batch_id=path.stem)

        assert tracker._state_path.exists(), "State file must be created after ingestion"

    def test_reset_forces_full_re_ingest(self, tmp_data_dir: Path, settings):
        """After reset(), all previously-seen docs should appear as new."""
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter

        raw_dir = tmp_data_dir / "data" / "raw"
        bronze_dir = tmp_data_dir / "data" / "bronze"
        _write_txt(raw_dir, "doc.txt", "Content to test reset behaviour.")

        tracker = _make_tracker(tmp_data_dir, settings)
        tracker.load()

        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)
        writer.bronze_dir = bronze_dir

        docs = tracker.filter_new(loader.load_directory(raw_dir))
        path = writer.write(docs)
        tracker.mark_ingested(docs, batch_id=path.stem)

        assert tracker.count == 1

        # Reset and verify everything looks new again
        tracker.reset()
        assert tracker.count == 0

        new_docs = tracker.filter_new(loader.load_directory(raw_dir))
        assert len(new_docs) == 1


# ── Embedding pipeline (mocked Qdrant) ─────────────────────────────────────────

class TestEmbeddingPipelineIntegration:
    """VectorIndexer end-to-end using a real Gold Parquet but mocked Qdrant."""

    def _write_gold(self, gold_dir: Path, n: int = 5) -> Path:
        import uuid
        import pandas as pd

        gold_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "chunk_id": f"{uuid.uuid4().hex[:8]}_{i:04d}",
                "doc_id": uuid.uuid4().hex[:16],
                "source": f"data/raw/doc_{i}.txt",
                "content": (
                    f"This is an embedding test chunk number {i}. "
                    "Transformer models use multi-head self-attention. "
                    "Vector databases support approximate nearest-neighbour search."
                ),
                "quality_score": 0.75 + i * 0.02,
            }
            for i in range(n)
        ]
        out = gold_dir / "part-00000.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        return gold_dir

    def test_indexer_reads_gold_and_calls_upsert(self, tmp_data_dir: Path, settings):
        from unittest.mock import MagicMock
        from pipelines.embeddings.indexer import VectorIndexer
        from src.config.settings import Settings

        n = 8
        gold_dir = tmp_data_dir / "data" / "gold"
        self._write_gold(gold_dir, n=n)

        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = lambda texts: [[0.0] * 384] * len(texts)
        mock_embedder.dimension = 384

        mock_store = MagicMock()

        settings_override = Settings(embedding_batch_size=4)
        indexer = VectorIndexer(
            embedder=mock_embedder,
            vector_store=mock_store,
            settings=settings_override,
        )
        total = indexer.run(gold_dir=gold_dir)

        assert total == n
        assert mock_store.upsert.call_count == 2  # ceil(8/4) = 2 batches

    def test_indexer_passes_correct_payload_fields(self, tmp_data_dir: Path, settings):
        from unittest.mock import MagicMock, call
        from pipelines.embeddings.indexer import VectorIndexer
        from src.config.settings import Settings

        gold_dir = tmp_data_dir / "data" / "gold"
        self._write_gold(gold_dir, n=2)

        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = lambda texts: [[0.0] * 384] * len(texts)
        mock_embedder.dimension = 384

        payloads_received = []

        def capture_upsert(ids, vectors, payloads):
            payloads_received.extend(payloads)

        mock_store = MagicMock()
        mock_store.upsert.side_effect = capture_upsert

        indexer = VectorIndexer(
            embedder=mock_embedder,
            vector_store=mock_store,
            settings=Settings(embedding_batch_size=10),
        )
        indexer.run(gold_dir=gold_dir)

        assert len(payloads_received) == 2
        for payload in payloads_received:
            assert "chunk_id" in payload
            assert "source" in payload
            assert "content" in payload
            assert "quality_score" in payload
