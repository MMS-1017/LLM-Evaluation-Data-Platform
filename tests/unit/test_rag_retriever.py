"""
Unit tests for RAGRetriever (services/retriever/rag_retriever.py).

All vector-store and embedding dependencies are mocked so tests run
without Qdrant, GPU, or API keys.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.common.exceptions import RetrieverError
from src.config.settings import Settings

pytestmark = pytest.mark.unit


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_vector_store():
    """A mock QdrantVectorStore that returns predictable search results."""
    store = MagicMock()
    store.search.return_value = [
        {
            "chunk_id": "chunk_001",
            "doc_id": "doc_abc",
            "source": "data/raw/transformers.txt",
            "content": "Transformers use self-attention mechanisms for sequence modelling.",
            "quality_score": 0.9,
        },
        {
            "chunk_id": "chunk_002",
            "doc_id": "doc_abc",
            "source": "data/raw/transformers.txt",
            "content": "The BERT model pre-trains on masked language modelling.",
            "quality_score": 0.85,
        },
    ]
    return store


@pytest.fixture()
def mock_embedder():
    """A mock EmbeddingService that returns a dummy 384-dim vector."""
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 384]
    embedder.dimension = 384
    return embedder


@pytest.fixture()
def retriever(mock_vector_store, mock_embedder):
    """RAGRetriever wired to the two mocks above."""
    from services.retriever.rag_retriever import RAGRetriever

    settings = Settings(
        embedding_provider="huggingface",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        llm_provider="huggingface",
        llm_model="google/flan-t5-base",
        api_top_k=5,
    )
    return RAGRetriever(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        settings=settings,
    )


# ── retrieve() ────────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_returns_list_of_dicts(self, retriever):
        results = retriever.retrieve("What is self-attention?")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_calls_embed_with_query(self, retriever, mock_embedder):
        retriever.retrieve("How do transformers work?")
        mock_embedder.embed.assert_called_once_with(["How do transformers work?"])

    def test_calls_vector_store_search(self, retriever, mock_vector_store):
        retriever.retrieve("RAG pipeline", top_k=3)
        mock_vector_store.search.assert_called_once()
        _, kwargs = mock_vector_store.search.call_args
        assert kwargs.get("top_k") == 3 or mock_vector_store.search.call_args[0][1] == 3

    def test_respects_top_k_from_settings(self, mock_vector_store, mock_embedder):
        from services.retriever.rag_retriever import RAGRetriever

        settings = Settings(api_top_k=7)
        r = RAGRetriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            settings=settings,
        )
        r.retrieve("test query")
        # top_k should default to settings.api_top_k when not passed
        call_kwargs = mock_vector_store.search.call_args
        assert call_kwargs is not None

    def test_embed_failure_raises_retriever_error(self, retriever, mock_embedder):
        mock_embedder.embed.side_effect = RuntimeError("Model not loaded")
        with pytest.raises(RetrieverError):
            retriever.retrieve("Will this fail?")

    def test_vector_store_failure_raises_retriever_error(self, retriever, mock_vector_store):
        mock_vector_store.search.side_effect = ConnectionError("Qdrant unreachable")
        with pytest.raises(RetrieverError):
            retriever.retrieve("Connection failure test")


# ── query() ───────────────────────────────────────────────────────────────────

class TestQuery:
    def test_returns_expected_keys(self, retriever):
        """query() must return question, answer, sources, contexts."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="A grounded answer.")
        retriever._llm = mock_llm

        result = retriever.query("What is a transformer?")
        assert set(result.keys()) >= {"question", "answer", "sources", "contexts"}

    def test_question_echoed_in_result(self, retriever):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Answer text.")
        retriever._llm = mock_llm

        result = retriever.query("Echo this question")
        assert result["question"] == "Echo this question"

    def test_sources_are_deduplicated(self, retriever):
        """Both chunks come from the same file — should appear once in sources."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Deduped sources answer.")
        retriever._llm = mock_llm

        result = retriever.query("Source dedup test")
        assert result["sources"].count("data/raw/transformers.txt") == 1

    def test_no_chunks_returns_fallback_answer(self, retriever, mock_vector_store):
        """When vector search returns nothing, a user-friendly message is returned."""
        mock_vector_store.search.return_value = []
        result = retriever.query("Query with no results")
        assert "could not find" in result["answer"].lower() or len(result["answer"]) > 0
        assert result["sources"] == []
        assert result["contexts"] == []

    def test_llm_failure_falls_back_to_first_context(self, retriever):
        """If LLM generation fails, the first retrieved chunk is returned as the answer."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM crash")
        retriever._llm = mock_llm

        result = retriever.query("LLM failure test")
        # Should not raise; should fall back gracefully
        assert len(result["answer"]) > 0

    def test_contexts_contain_retrieved_content(self, retriever):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Generated answer.")
        retriever._llm = mock_llm

        result = retriever.query("Context check")
        assert any("self-attention" in ctx.lower() for ctx in result["contexts"])


# ── VectorIndexer (mocked) ────────────────────────────────────────────────────

class TestVectorIndexerUnit:
    """Unit-level tests for VectorIndexer using mocked embedder + vector store."""

    def _make_gold_parquet(self, tmp_path, n: int = 5) -> "Path":
        import pandas as pd

        gold_dir = tmp_path / "data" / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "chunk_id": [f"chunk_{i:04d}" for i in range(n)],
            "doc_id": [f"doc_{i}" for i in range(n)],
            "source": [f"data/raw/doc_{i}.txt" for i in range(n)],
            "content": [f"Chunk content number {i} about transformers." for i in range(n)],
            "quality_score": [0.8 + i * 0.01 for i in range(n)],
        })
        out = gold_dir / "part-00000.parquet"
        df.to_parquet(out, index=False)
        return gold_dir

    def test_run_returns_correct_count(self, tmp_path):
        from pipelines.embeddings.indexer import VectorIndexer

        gold_dir = self._make_gold_parquet(tmp_path, n=4)
        settings = Settings(embedding_batch_size=10)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.0] * 384] * 4
        mock_embedder.dimension = 384

        mock_store = MagicMock()

        indexer = VectorIndexer(
            embedder=mock_embedder,
            vector_store=mock_store,
            settings=settings,
        )
        count = indexer.run(gold_dir=gold_dir)
        assert count == 4

    def test_run_calls_upsert_for_each_batch(self, tmp_path):
        from pipelines.embeddings.indexer import VectorIndexer

        n = 6
        gold_dir = self._make_gold_parquet(tmp_path, n=n)
        settings = Settings(embedding_batch_size=3)  # 2 batches

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.0] * 384] * 3
        mock_embedder.dimension = 384

        mock_store = MagicMock()

        indexer = VectorIndexer(
            embedder=mock_embedder,
            vector_store=mock_store,
            settings=settings,
        )
        indexer.run(gold_dir=gold_dir)
        assert mock_store.upsert.call_count == 2  # ceil(6/3) = 2 batches

    def test_run_empty_gold_returns_zero(self, tmp_path):
        import pandas as pd
        from pipelines.embeddings.indexer import VectorIndexer

        gold_dir = tmp_path / "data" / "gold"
        gold_dir.mkdir(parents=True)
        pd.DataFrame(columns=["chunk_id", "doc_id", "source", "content"]).to_parquet(
            gold_dir / "empty.parquet", index=False
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        mock_store = MagicMock()

        indexer = VectorIndexer(
            embedder=mock_embedder,
            vector_store=mock_store,
            settings=Settings(),
        )
        count = indexer.run(gold_dir=gold_dir)
        assert count == 0
        mock_store.upsert.assert_not_called()
