"""
RAGRetriever — retrieval-augmented generation using Qdrant + an LLM.

Retrieval flow:
  1. Embed the user question with EmbeddingService
  2. Search Qdrant for the top-k most similar chunks
  3. Build a context-grounded prompt
  4. Generate an answer with the configured LLM (HuggingFace or OpenAI)

The retriever can also be used in retrieval-only mode (without generation)
for evaluation and testing.
"""
from __future__ import annotations

from typing import Any

from src.common.exceptions import RetrieverError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings

log = get_logger(__name__)


class RAGRetriever:
    """
    RAG pipeline: embedding → vector search → LLM generation.

    Example:
        retriever = RAGRetriever()
        result = retriever.query("What is a transformer model?")
        print(result["answer"])
        print(result["sources"])
    """

    _PROMPT_TEMPLATE = (
        "You are a helpful AI assistant. Answer the question using ONLY the "
        "information provided in the context sections below. "
        "If the context does not contain enough information, say so.\n\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        vector_store=None,
        embedder=None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        if vector_store is None:
            from services.vectorstore.qdrant_store import QdrantVectorStore
            vector_store = QdrantVectorStore(self.settings)
        if embedder is None:
            from pipelines.embeddings.embedder import EmbeddingService
            embedder = EmbeddingService(self.settings)

        self.vector_store = vector_store
        self.embedder = embedder
        self._llm = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for *query*.

        Args:
            query: Natural language question or search string.
            top_k: Number of chunks to return (defaults to settings.api_top_k).

        Returns:
            List of payload dicts ordered by similarity score (descending).

        Raises:
            RetrieverError: On embedding or search failure.
        """
        k = top_k or self.settings.api_top_k
        try:
            embedding = self.embedder.embed([query])[0]
            results = self.vector_store.search(query_vector=embedding, top_k=k)
            log.debug(f"Retrieved {len(results)} chunks for query: '{query[:60]}...'")
            return results
        except Exception as exc:
            raise RetrieverError(
                "Retrieval failed", details={"query": query[:100], "error": str(exc)}
            ) from exc

    def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Full RAG query: retrieve context, then generate an answer.

        Returns:
            Dict with keys: question, answer, sources, contexts.
        """
        log.info(f"RAG query: '{question[:80]}'")
        chunks = self.retrieve(question, top_k)

        if not chunks:
            log.warning("No relevant chunks found — returning empty-context response")
            return {
                "question": question,
                "answer": "I could not find relevant information in the knowledge base.",
                "sources": [],
                "contexts": [],
            }

        contexts = [c.get("content", "") for c in chunks]
        sources = sorted({c.get("source", "unknown") for c in chunks})

        try:
            answer = self._generate(question, contexts)
        except Exception as exc:
            log.warning(f"LLM generation failed ({exc}) — returning best context as answer")
            answer = contexts[0] if contexts else "Unable to generate answer."

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _generate(self, question: str, contexts: list[str]) -> str:
        """Build prompt and call the configured LLM."""
        llm = self._load_llm()
        context_block = "\n\n---\n\n".join(
            f"[Context {i + 1}]:\n{ctx}" for i, ctx in enumerate(contexts)
        )
        prompt = self._PROMPT_TEMPLATE.format(context=context_block, question=question)

        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        else:
            return str(llm(prompt))

    def _load_llm(self):
        """Lazily load the configured LLM."""
        if self._llm is not None:
            return self._llm

        provider = self.settings.llm_provider
        model = self.settings.llm_model
        log.info(f"Loading LLM [{provider}] {model}")

        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=model,
                    api_key=self.settings.openai_api_key,
                    temperature=0.0,
                )
            except ImportError as exc:
                raise RetrieverError(
                    "langchain-openai not installed. Run: pip install langchain-openai"
                ) from exc

        elif provider == "huggingface":
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
                from langchain_community.llms import HuggingFacePipeline

                tokenizer = AutoTokenizer.from_pretrained(model)
                model_obj = AutoModelForSeq2SeqLM.from_pretrained(model)
                pipe = pipeline(
                    "text2text-generation",
                    model=model_obj,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                )
                self._llm = HuggingFacePipeline(pipeline=pipe)
            except ImportError as exc:
                raise RetrieverError(
                    "transformers or langchain-community not installed"
                ) from exc
        else:
            raise RetrieverError(f"Unknown LLM provider: {provider!r}")

        log.info("LLM loaded successfully")
        return self._llm
