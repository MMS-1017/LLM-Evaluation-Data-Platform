"""RAG query endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from services.api.dependencies import get_retriever
from services.api.schemas import QueryRequest, QueryResponse
from src.common.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="RAG query",
    description=(
        "Submit a natural language question. The platform retrieves relevant "
        "document chunks from Qdrant and generates a grounded answer using the "
        "configured LLM."
    ),
)
def query_endpoint(
    request: QueryRequest,
    retriever=Depends(get_retriever),
) -> QueryResponse:
    try:
        result = retriever.query(question=request.question, top_k=request.top_k)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            contexts=result["contexts"],
            top_k=request.top_k,
        )
    except Exception as exc:
        log.exception(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
