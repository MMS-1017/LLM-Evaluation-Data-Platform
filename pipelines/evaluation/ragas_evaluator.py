"""
RagasEvaluator — evaluates a RAG pipeline using Ragas metrics.

Supports two modes:
  - Full Ragas evaluation (requires Ragas + LLM access)
  - Mock evaluation (fast heuristic scoring for CI/CD pipelines)

The mode is selected automatically based on whether Ragas is importable
and configured (use_llm_metrics in configs/evaluation.yaml).
"""
from __future__ import annotations

import uuid
from typing import Any

from src.common.exceptions import EvaluationError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings
from src.models.evaluation import EvaluationReport, EvaluationResult, MetricScore, RAGSample

log = get_logger(__name__)

_RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


class RagasEvaluator:
    """
    Evaluates RAG quality using Ragas metrics.

    Example:
        evaluator = RagasEvaluator()
        report = evaluator.evaluate(samples)
        print(report.aggregate_metrics)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, samples: list[RAGSample]) -> EvaluationReport:
        """
        Run evaluation on *samples* and return an EvaluationReport.

        Falls back to mock scoring if Ragas is not available.

        Raises:
            EvaluationError: If the sample list is empty.
        """
        if not samples:
            raise EvaluationError("Cannot evaluate an empty sample set")

        log.info(f"Evaluating {len(samples)} RAG samples")

        try:
            import ragas  # noqa: F401
            log.info("Ragas detected — running full metric evaluation")
            return self._run_ragas(samples)
        except ImportError:
            log.warning("Ragas not installed — using heuristic mock evaluation")
            return self._mock_evaluation(samples)
        except Exception as exc:
            log.warning(f"Ragas evaluation failed ({exc}) — falling back to mock")
            return self._mock_evaluation(samples)

    # ── Ragas evaluation ───────────────────────────────────────────────────────

    def _run_ragas(self, samples: list[RAGSample]) -> EvaluationReport:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or s.answer for s in samples],
        })

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        aggregate: dict[str, float] = {
            m: float(df[m].mean()) for m in _RAGAS_METRICS if m in df.columns
        }

        results = [
            EvaluationResult(
                sample=samples[i],
                metrics=[
                    MetricScore(name=m, score=float(df.iloc[i].get(m, 0.0)))
                    for m in aggregate
                ],
            )
            for i in range(len(samples))
        ]

        return EvaluationReport(
            run_id=str(uuid.uuid4()),
            experiment_name=self.settings.mlflow_experiment_name,
            num_samples=len(samples),
            aggregate_metrics=aggregate,
            results=results,
        )

    # ── Mock evaluation ────────────────────────────────────────────────────────

    def _mock_evaluation(self, samples: list[RAGSample]) -> EvaluationReport:
        """
        Compute heuristic scores without an LLM.

        Scores are based on simple overlap between contexts and answers,
        making this suitable for integration tests and CI pipelines.
        """
        import math

        def _overlap_score(text_a: str, text_b: str) -> float:
            words_a = set(text_a.lower().split())
            words_b = set(text_b.lower().split())
            if not words_a or not words_b:
                return 0.0
            intersection = len(words_a & words_b)
            return round(intersection / math.sqrt(len(words_a) * len(words_b)), 4)

        results = []
        metric_totals: dict[str, float] = {m: 0.0 for m in _RAGAS_METRICS}

        for sample in samples:
            ctx_text = " ".join(sample.contexts)
            scores = {
                "faithfulness": _overlap_score(sample.answer, ctx_text),
                "answer_relevancy": _overlap_score(sample.answer, sample.question),
                "context_precision": _overlap_score(ctx_text, sample.question),
                "context_recall": _overlap_score(ctx_text, sample.ground_truth or sample.answer),
            }
            for m, v in scores.items():
                metric_totals[m] += v

            results.append(
                EvaluationResult(
                    sample=sample,
                    metrics=[MetricScore(name=m, score=v) for m, v in scores.items()],
                )
            )

        n = len(samples)
        aggregate = {m: round(v / n, 4) for m, v in metric_totals.items()}

        log.info(f"Mock evaluation complete. Aggregate metrics: {aggregate}")

        return EvaluationReport(
            run_id=str(uuid.uuid4()),
            experiment_name=self.settings.mlflow_experiment_name,
            num_samples=n,
            aggregate_metrics=aggregate,
            results=results,
            metadata={"mode": "heuristic_mock"},
        )
