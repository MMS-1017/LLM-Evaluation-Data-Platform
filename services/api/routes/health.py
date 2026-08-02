"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from services.api.dependencies import get_vector_store
from services.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Platform health check",
    description="Returns Qdrant connectivity status and collection statistics.",
)
def health_check(store=Depends(get_vector_store)) -> HealthResponse:
    is_healthy = store.health_check()
    qdrant_info = store.get_collection_info() if is_healthy else {"status": "unreachable"}
    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        qdrant=qdrant_info,
    )
