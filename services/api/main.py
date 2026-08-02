"""
FastAPI application factory for the LLM Evaluation Data Platform.

Endpoints:
  GET  /health   — Qdrant connectivity + collection stats
  POST /query    — RAG question answering
  POST /ingest   — Trigger document ingestion to Bronze layer
  GET  /docs     — Swagger UI (auto-generated)
  GET  /redoc    — ReDoc UI (auto-generated)

Usage:
    uvicorn services.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.logging import get_logger, setup_logger
from src.config.settings import get_settings

setup_logger(level="INFO")
log = get_logger(__name__)

_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler: startup warmup and graceful shutdown."""
    settings = get_settings()
    log.info(f"LLM Eval Platform API v{_VERSION} starting")
    log.info(f"  Qdrant : {settings.qdrant_host}:{settings.qdrant_port}")
    log.info(f"  MLflow : {settings.mlflow_tracking_uri}")
    log.info(f"  Embed  : {settings.embedding_provider}/{settings.embedding_model}")
    log.info(f"  LLM    : {settings.llm_provider}/{settings.llm_model}")

    # Warm up the Qdrant connection so the first request isn't slow
    from services.api.dependencies import get_vector_store
    try:
        store = get_vector_store()
        if store.health_check():
            log.info("Qdrant connection: healthy")
        else:
            log.warning("Qdrant unreachable — RAG queries will fail until Qdrant is available")
    except Exception as exc:
        log.warning(f"Startup Qdrant check failed: {exc}")

    yield  # Application runs here

    log.info("LLM Eval Platform API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from services.api.routes import health, query, ingest

    app = FastAPI(
        title="LLM Evaluation Data Platform",
        description=(
            "Production AI Data Engineering platform: ingest documents via a "
            "Medallion Architecture (Bronze → Silver → Gold), embed with "
            "HuggingFace sentence-transformers, index in Qdrant, serve RAG "
            "queries, and evaluate quality with Ragas + MLflow tracking."
        ),
        version=_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "health", "description": "System health checks"},
            {"name": "query", "description": "RAG question answering"},
            {"name": "ingest", "description": "Document ingestion pipeline"},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(ingest.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "services.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False,
        log_level="info",
    )
