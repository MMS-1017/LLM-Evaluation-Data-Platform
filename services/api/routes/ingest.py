"""Ingest trigger endpoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from services.api.schemas import IngestRequest, IngestResponse
from src.common.logging import get_logger

log = get_logger(__name__)
router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger document ingestion",
    description=(
        "Load documents from the configured raw directory and write them to "
        "the Bronze layer. Optionally override the raw directory path."
    ),
)
def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    try:
        from pipelines.ingestion.document_loader import DocumentLoader
        from pipelines.ingestion.bronze_writer import BronzeWriter
        from src.config.settings import get_settings

        settings = get_settings()
        loader = DocumentLoader(settings)
        writer = BronzeWriter(settings)

        raw_dir = Path(request.raw_dir) if request.raw_dir else None
        documents = loader.load_directory(raw_dir)

        if not documents:
            return IngestResponse(
                status="warning",
                documents_ingested=0,
                message="No documents found in the raw directory",
            )

        writer.write(documents)
        return IngestResponse(
            status="success",
            documents_ingested=len(documents),
            message=f"Successfully ingested {len(documents)} document(s) to the bronze layer",
        )

    except Exception as exc:
        log.exception(f"Ingest endpoint failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
