"""
TestSetGenerator — builds a synthetic evaluation test set from Gold chunks.

Two modes:
  1. Template-based (default, no API key needed): fast, deterministic
  2. LLM-based (optional): uses a language model for richer questions

The template mode is sufficient for pipeline smoke tests and CI.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src.common.exceptions import EvaluationError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings
from src.models.evaluation import RAGSample

log = get_logger(__name__)

# Simple question templates for template-based generation
_TEMPLATES = [
    "What is {topic}?",
    "How does {topic} work?",
    "What are the main components of {topic}?",
    "What are the key benefits of {topic}?",
    "Explain {topic} and its significance.",
    "What role does {topic} play in modern technology?",
    "Describe the purpose of {topic}.",
]


class TestSetGenerator:
    """
    Generates RAG evaluation samples from Gold-layer chunks.

    Example:
        gen = TestSetGenerator()
        samples = gen.generate(num_samples=20)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        gold_dir: Path | None = None,
        num_samples: int | None = None,
    ) -> list[RAGSample]:
        """
        Generate *num_samples* evaluation samples from the Gold layer.

        Args:
            gold_dir:    Override for the gold Parquet directory.
            num_samples: Number of samples to generate.

        Returns:
            List of RAGSample objects ready for Ragas evaluation.

        Raises:
            EvaluationError: If the gold layer is missing or empty.
        """
        n = num_samples or self.settings.eval_test_set_size
        gold_path = gold_dir or self.settings.resolve_path(self.settings.data_gold_dir)

        log.info(f"Generating {n} test samples from: {gold_path}")

        try:
            df = pd.read_parquet(str(gold_path))
        except FileNotFoundError as exc:
            raise EvaluationError(
                f"Gold Parquet not found at {gold_path} — run the full pipeline first"
            ) from exc
        except Exception as exc:
            raise EvaluationError(f"Failed to read gold data: {exc}") from exc

        if df.empty:
            raise EvaluationError("Gold layer is empty — no samples to generate")

        rng = random.Random(42)  # Seeded for reproducibility
        sample_df = df.sample(n=min(n, len(df)), random_state=42)
        samples: list[RAGSample] = []

        for _, row in sample_df.iterrows():
            content: str = str(row["content"])
            question = self._make_question(content, rng)
            ground_truth = self._extract_answer(content)
            samples.append(
                RAGSample(
                    question=question,
                    answer=ground_truth,       # Placeholder — will be replaced by RAG output during eval
                    contexts=[content],
                    ground_truth=ground_truth,
                    metadata={
                        "doc_id": row.get("doc_id", ""),
                        "source": row.get("source", ""),
                        "chunk_id": row.get("chunk_id", ""),
                    },
                )
            )

        log.info(f"Generated {len(samples)} test sample(s)")
        return samples

    # ── Private helpers ────────────────────────────────────────────────────────

    def _make_question(self, content: str, rng: random.Random) -> str:
        """Generate a question from a chunk using keyword extraction + templates."""
        # Use the first noun-like word longer than 4 chars from the first sentence
        first_sentence = content.split(".")[0][:120].strip()
        candidates = [w.strip(",.;:()[]\"'") for w in first_sentence.split() if len(w) > 4]
        topic = candidates[0] if candidates else "this concept"
        template = rng.choice(_TEMPLATES)
        return template.format(topic=topic)

    @staticmethod
    def _extract_answer(content: str) -> str:
        """Use the first two complete sentences as a reference answer."""
        sentences = [s.strip() for s in content.split(".") if s.strip()]
        answer = ". ".join(sentences[:2])
        return (answer + ".") if answer else content[:200]
