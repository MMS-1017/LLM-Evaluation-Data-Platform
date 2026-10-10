"""
Integration tests for the evaluation pipeline components.

Tests TestSetGenerator and RagasEvaluator using synthetic Gold-layer
Parquet data written to a temporary directory. No Spark or Qdrant needed.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _write_fake_gold_parquet(gold_dir: Path, n_rows: int = 10) -> Path:
    """Helper: write a minimal Gold-layer Parquet file to gold_dir."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_rows):
        doc_id = uuid.uuid4().hex[:16]
        rows.append({
            "chunk_id": f"{doc_id[:8]}_{i:04d}",
            "doc_id": doc_id,
            "source": f"data/raw/doc_{i}.txt",
            "content": (
                f"This is chunk number {i} about artificial intelligence and machine learning. "
                f"Transformers are neural architectures that use self-attention mechanisms. "
                f"They have revolutionised natural language processing tasks significantly."
            ),
            "token_count": 30 + i,
            "quality_score": 0.75 + (i % 5) * 0.02,
            "file_type": "txt",
            "metadata": "{}",
        })
    df = pd.DataFrame(rows)
    out = gold_dir / "part-00000.parquet"
    df.to_parquet(out, index=False)
    return out


# ── TestSetGenerator ───────────────────────────────────────────────────────────

class TestTestSetGenerator:
    """Tests for synthetic question/answer generation from Gold chunks."""

    def test_generates_requested_number_of_samples(self, tmp_data_dir: Path, settings):
        from pipelines.evaluation.test_set_generator import TestSetGenerator

        gold_dir = tmp_data_dir / "data" / "gold"
        _write_fake_gold_parquet(gold_dir, n_rows=20)

        gen = TestSetGenerator(settings)
        samples = gen.generate(gold_dir=gold_dir, num_samples=5)
        assert len(samples) == 5

    def test_samples_have_required_fields(self, tmp_data_dir: Path, settings):
        from pipelines.evaluation.test_set_generator import TestSetGenerator

        gold_dir = tmp_data_dir / "data" / "gold"
        _write_fake_gold_parquet(gold_dir, n_rows=10)

        samples = TestSetGenerator(settings).generate(gold_dir=gold_dir, num_samples=3)
        for s in samples:
            assert len(s.question) > 0, "Question must not be empty"
            assert len(s.ground_truth) > 0, "Ground truth must not be empty"
            assert len(s.contexts) == 1, "Each sample should have one context chunk"

    def test_generation_is_reproducible(self, tmp_data_dir: Path, settings):
        """Same gold dir + same seed should produce identical questions."""
        from pipelines.evaluation.test_set_generator import TestSetGenerator

        gold_dir = tmp_data_dir / "data" / "gold"
        _write_fake_gold_parquet(gold_dir, n_rows=10)

        gen = TestSetGenerator(settings)
        samples_a = gen.generate(gold_dir=gold_dir, num_samples=5)
        samples_b = gen.generate(gold_dir=gold_dir, num_samples=5)

        questions_a = [s.question for s in samples_a]
        questions_b = [s.question for s in samples_b]
        assert questions_a == questions_b, "Generation must be deterministic (seed=42)"

    def test_caps_samples_at_available_chunks(self, tmp_data_dir: Path, settings):
        """If num_samples > available chunks, output is capped at chunk count."""
        from pipelines.evaluation.test_set_generator import TestSetGenerator

        gold_dir = tmp_data_dir / "data" / "gold"
        _write_fake_gold_parquet(gold_dir, n_rows=3)

        samples = TestSetGenerator(settings).generate(gold_dir=gold_dir, num_samples=100)
        assert len(samples) == 3

    def test_missing_gold_dir_raises(self, tmp_data_dir: Path, settings):
        from pipelines.evaluation.test_set_generator import TestSetGenerator
        from src.common.exceptions import EvaluationError

        missing = tmp_data_dir / "nonexistent"
        with pytest.raises(EvaluationError):
            TestSetGenerator(settings).generate(gold_dir=missing, num_samples=5)


# ── RagasEvaluator (heuristic mode) ───────────────────────────────────────────

class TestRagasEvaluatorHeuristic:
    """
    Tests for the heuristic fallback evaluator.
    Ragas library is NOT imported here — these tests verify the mock path.
    """

    def _make_samples(self, n: int = 5):
        from src.models.evaluation import RAGSample
        return [
            RAGSample(
                question=f"What is concept number {i}?",
                answer=f"Concept {i} relates to artificial intelligence and data engineering.",
                contexts=[
                    f"Artificial intelligence concept {i} is fundamental to modern data engineering. "
                    f"It involves machine learning and statistical methods for extracting insights."
                ],
                ground_truth=f"Concept {i} relates to artificial intelligence.",
            )
            for i in range(n)
        ]

    def test_evaluate_returns_report(self, settings):
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(settings)
        samples = self._make_samples(5)

        # Force heuristic path
        report = evaluator._mock_evaluation(samples)

        assert report.num_samples == 5
        assert len(report.aggregate_metrics) > 0

    def test_aggregate_metrics_all_in_range(self, settings):
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator

        report = RagasEvaluator(settings)._mock_evaluation(self._make_samples(5))
        for name, score in report.aggregate_metrics.items():
            assert 0.0 <= score <= 1.0, f"Metric {name}={score} out of [0, 1]"

    def test_expected_metric_names_present(self, settings):
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator

        report = RagasEvaluator(settings)._mock_evaluation(self._make_samples(3))
        expected = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
        assert expected.issubset(set(report.aggregate_metrics.keys()))

    def test_empty_samples_raises(self, settings):
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator
        from src.common.exceptions import EvaluationError

        with pytest.raises(EvaluationError):
            RagasEvaluator(settings).evaluate([])

    def test_run_id_is_unique_per_report(self, settings):
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(settings)
        r1 = evaluator._mock_evaluation(self._make_samples(3))
        r2 = evaluator._mock_evaluation(self._make_samples(3))
        assert r1.run_id != r2.run_id

    def test_report_metadata_flags_heuristic_mode(self, settings):
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator

        report = RagasEvaluator(settings)._mock_evaluation(self._make_samples(3))
        assert report.metadata.get("mode") == "heuristic_mock"


# ── End-to-end evaluation pipeline (no MLflow) ────────────────────────────────

class TestEvaluationPipelineEndToEnd:
    """Full TestSetGenerator → RagasEvaluator flow without external services."""

    def test_full_evaluation_flow(self, tmp_data_dir: Path, settings):
        from pipelines.evaluation.test_set_generator import TestSetGenerator
        from pipelines.evaluation.ragas_evaluator import RagasEvaluator

        gold_dir = tmp_data_dir / "data" / "gold"
        _write_fake_gold_parquet(gold_dir, n_rows=10)

        samples = TestSetGenerator(settings).generate(gold_dir=gold_dir, num_samples=5)
        report = RagasEvaluator(settings).evaluate(samples)

        assert report.num_samples == 5
        for score in report.aggregate_metrics.values():
            assert 0.0 <= score <= 1.0
