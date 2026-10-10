"""
IngestStateTracker — records which files have already been ingested.

The tracker stores a JSON state file alongside the Bronze layer.  On each
ingestion run it:

1. Reads the state file (or starts fresh if it does not exist).
2. Filters out any source paths that were already ingested in a previous run.
3. After a successful Bronze write, updates the state file with the new
   entries.

This enables *incremental ingestion*: only files added or modified since the
last run are processed, so unchanged documents are never re-ingested and
Bronze files never accumulate duplicates of the same content.

Usage::

    from pipelines.ingestion.state_tracker import IngestStateTracker

    tracker = IngestStateTracker(settings)
    new_docs = tracker.filter_new(all_docs)   # discard already-seen docs
    if new_docs:
        path = writer.write(new_docs)
        tracker.mark_ingested(new_docs)        # persist state
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.exceptions import IngestionError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings
from src.models.document import RawDocument

log = get_logger(__name__)

_STATE_FILENAME = "ingestion_state.json"


class IngestStateTracker:
    """
    Tracks which source files have been ingested via a JSON state file.

    The state file lives at ``<bronze_dir>/.ingestion_state.json`` and maps
    ``doc_id`` → ``{source, ingested_at, batch}``.  Using the content-
    addressed ``doc_id`` (SHA-256 of file content) means that the same
    physical file will not be re-ingested if its content has not changed,
    even if it is renamed or moved.

    Thread safety: not guaranteed.  The tracker is designed for sequential,
    single-process pipelines.  Airflow-style task isolation (one process per
    run) handles concurrency at the orchestration level.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._bronze_dir: Path = self.settings.resolve_path(self.settings.data_bronze_dir)
        self._state_path: Path = self._bronze_dir / _STATE_FILENAME
        self._state: dict[str, Any] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the persisted state from disk (no-op if file does not exist)."""
        if self._state_path.exists():
            try:
                with self._state_path.open("r", encoding="utf-8") as fh:
                    self._state = json.load(fh)
                log.info(
                    f"Loaded ingestion state: {len(self._state)} known doc(s) "
                    f"from {self._state_path}"
                )
            except (json.JSONDecodeError, OSError) as exc:
                log.warning(
                    f"Could not read state file ({exc}) — starting with empty state"
                )
                self._state = {}
        else:
            log.info("No existing ingestion state found — all documents are new")
            self._state = {}

    def filter_new(self, documents: list[RawDocument]) -> list[RawDocument]:
        """
        Return only documents whose ``doc_id`` is not yet in the state.

        Args:
            documents: All documents loaded from the raw directory.

        Returns:
            Subset of *documents* that have not been ingested before.
        """
        new_docs = [d for d in documents if d.doc_id not in self._state]
        skipped = len(documents) - len(new_docs)
        if skipped:
            log.info(f"Incremental filter: {skipped} already-seen doc(s) skipped, {len(new_docs)} new")
        else:
            log.info(f"Incremental filter: all {len(new_docs)} doc(s) are new")
        return new_docs

    def mark_ingested(self, documents: list[RawDocument], batch_id: str = "") -> None:
        """
        Record *documents* as ingested and persist the updated state.

        Args:
            documents: Documents that were successfully written to Bronze.
            batch_id:  Identifier of the batch file they were written to
                       (used for auditability; empty string is fine).

        Raises:
            IngestionError: If the state file cannot be written.
        """
        ts = datetime.now(timezone.utc).isoformat()
        for doc in documents:
            self._state[doc.doc_id] = {
                "source": doc.source,
                "ingested_at": ts,
                "batch": batch_id,
            }
        self._save()
        log.info(f"State updated: {len(documents)} doc(s) marked as ingested")

    def is_ingested(self, doc_id: str) -> bool:
        """Return True if *doc_id* has been recorded as ingested."""
        return doc_id in self._state

    @property
    def known_doc_ids(self) -> frozenset[str]:
        """Frozen set of all doc_ids recorded in the state."""
        return frozenset(self._state.keys())

    @property
    def count(self) -> int:
        """Number of documents recorded in the state."""
        return len(self._state)

    def reset(self) -> None:
        """
        Clear all state in memory and delete the state file from disk.

        Use with caution — this forces a full re-ingestion on the next run.
        """
        self._state = {}
        if self._state_path.exists():
            self._state_path.unlink()
            log.warning(f"Ingestion state file deleted: {self._state_path}")
        log.info("Ingestion state reset (full re-ingest will occur on next run)")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist the current in-memory state to the JSON file on disk."""
        self._bronze_dir.mkdir(parents=True, exist_ok=True)
        try:
            with self._state_path.open("w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2, ensure_ascii=False)
            log.debug(f"State file written: {self._state_path} ({len(self._state)} entries)")
        except OSError as exc:
            raise IngestionError(
                f"Failed to write ingestion state: {self._state_path}",
                details={"error": str(exc)},
            ) from exc
