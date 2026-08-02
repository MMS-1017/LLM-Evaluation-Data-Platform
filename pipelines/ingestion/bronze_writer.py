"""
BronzeWriter — persists RawDocuments to the Bronze layer as Parquet files.

Each invocation writes a time-stamped batch file:
    data/bronze/batch_20240101_120000.parquet
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.common.exceptions import IngestionError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings
from src.models.document import RawDocument
from src.utils.file_utils import ensure_dir, write_parquet

log = get_logger(__name__)


class BronzeWriter:
    """
    Serializes a batch of RawDocuments to Bronze-layer Parquet.

    The bronze layer stores raw content verbatim — no transformations,
    only type-safe serialization with provenance metadata.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.bronze_dir = self.settings.resolve_path(self.settings.data_bronze_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def write(self, documents: list[RawDocument]) -> Path:
        """
        Write *documents* to a timestamped Parquet file in the bronze directory.

        Args:
            documents: Non-empty list of RawDocument instances.

        Returns:
            Path to the written Parquet file.

        Raises:
            IngestionError: If documents list is empty.
        """
        if not documents:
            raise IngestionError("Cannot write empty document list to bronze layer")

        ensure_dir(self.bronze_dir)
        batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = self.bronze_dir / f"batch_{batch_id}.parquet"

        df = pd.DataFrame([self._serialize(doc) for doc in documents])
        write_parquet(df, output_path)

        log.info(f"Bronze write: {len(documents)} docs → {output_path}")
        return output_path

    def read_all(self) -> pd.DataFrame:
        """
        Concatenate all bronze Parquet files into a single DataFrame.
        Returns an empty DataFrame if the bronze directory has no files.
        """
        files = sorted(self.bronze_dir.glob("*.parquet"))
        if not files:
            log.warning("No bronze Parquet files found")
            return pd.DataFrame()

        frames = [pd.read_parquet(f) for f in files]
        result = pd.concat(frames, ignore_index=True)
        log.info(f"Read {len(result)} bronze records from {len(files)} file(s)")
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _serialize(doc: RawDocument) -> dict:
        return {
            "doc_id": doc.doc_id,
            "source": doc.source,
            "content": doc.content,
            "file_type": doc.file_type,
            "byte_size": doc.byte_size,
            "ingested_at": doc.ingested_at.isoformat(),
            "metadata": str(doc.metadata),
        }
