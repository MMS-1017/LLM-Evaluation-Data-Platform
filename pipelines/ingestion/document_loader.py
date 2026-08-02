"""
DocumentLoader — loads raw files from the raw-data directory into RawDocument objects.

Supported formats: .txt, .md, .json, .pdf, .csv
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

from src.common.exceptions import IngestionError
from src.common.logging import get_logger
from src.config.settings import Settings, get_settings
from src.models.document import RawDocument

log = get_logger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".json", ".pdf", ".csv"})


class DocumentLoader:
    """
    Loads raw documents from a directory, dispatching to format-specific
    readers. Returns a flat list of RawDocument objects.

    Example:
        loader = DocumentLoader()
        docs = loader.load_directory(Path("data/raw"))
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_directory(self, raw_dir: Path | None = None) -> list[RawDocument]:
        """
        Load all supported documents from *raw_dir*.
        Defaults to settings.data_raw_dir if not specified.
        """
        directory = raw_dir or self.settings.resolve_path(self.settings.data_raw_dir)
        log.info(f"Loading documents from: {directory}")

        docs = list(self._iter_documents(directory))
        log.info(f"Loaded {len(docs)} document(s) from {directory}")
        return docs

    # ── Private helpers ────────────────────────────────────────────────────────

    def _iter_documents(self, directory: Path) -> Iterator[RawDocument]:
        if not directory.exists():
            log.warning(f"Raw directory does not exist: {directory}")
            return

        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                log.debug(f"Skipping unsupported file: {path.name}")
                continue
            try:
                doc = self._load_file(path)
                log.debug(f"Loaded {path.name} ({doc.byte_size} bytes)")
                yield doc
            except IngestionError:
                raise
            except Exception as exc:
                raise IngestionError(
                    f"Failed to load file: {path}",
                    details={"path": str(path), "error": str(exc)},
                ) from exc

    def _load_file(self, path: Path) -> RawDocument:
        ext = path.suffix.lower()
        loaders = {
            ".txt": self._load_text,
            ".md": self._load_text,
            ".json": self._load_json,
            ".pdf": self._load_pdf,
            ".csv": self._load_csv,
        }
        loader = loaders.get(ext)
        if loader is None:
            raise IngestionError(f"No loader for extension: {ext}")
        return loader(path)

    # ── Format-specific readers ────────────────────────────────────────────────

    def _load_text(self, path: Path) -> RawDocument:
        """Load plain text or Markdown files."""
        return RawDocument.from_file(path)

    def _load_json(self, path: Path) -> RawDocument:
        """
        Load JSON files. Extracts 'content' or 'text' keys if present,
        otherwise serializes the entire JSON structure.
        """
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            content = raw  # Fall back to raw string
        else:
            if isinstance(data, dict):
                content = (
                    data.get("content")
                    or data.get("text")
                    or data.get("body")
                    or json.dumps(data, indent=2, ensure_ascii=False)
                )
            elif isinstance(data, list):
                content = "\n".join(
                    item.get("content", str(item)) if isinstance(item, dict) else str(item)
                    for item in data
                )
            else:
                content = str(data)

        return RawDocument(
            doc_id=hashlib.sha256(content.encode()).hexdigest(),
            source=str(path),
            content=content,
            file_type="json",
            byte_size=path.stat().st_size,
            metadata={"filename": path.name},
        )

    def _load_pdf(self, path: Path) -> RawDocument:
        """Load PDF files using pypdf (optional dependency)."""
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n\n".join(p for p in pages if p.strip())
        except ImportError:
            log.warning("pypdf not installed — falling back to raw bytes for PDF")
            content = path.read_bytes().decode("utf-8", errors="replace")
        except Exception as exc:
            raise IngestionError(
                f"PDF read failed: {path.name}", details={"error": str(exc)}
            ) from exc

        return RawDocument(
            doc_id=hashlib.sha256(content.encode()).hexdigest(),
            source=str(path),
            content=content,
            file_type="pdf",
            byte_size=path.stat().st_size,
            metadata={"filename": path.name},
        )

    def _load_csv(self, path: Path) -> RawDocument:
        """Load CSV files; converts to a text representation."""
        try:
            import pandas as pd

            df = pd.read_csv(path)
            content = df.to_string(index=False)
            extra_meta = {"rows": len(df), "columns": list(df.columns)}
        except Exception as exc:
            raise IngestionError(
                f"CSV read failed: {path.name}", details={"error": str(exc)}
            ) from exc

        return RawDocument(
            doc_id=hashlib.sha256(content.encode()).hexdigest(),
            source=str(path),
            content=content,
            file_type="csv",
            byte_size=path.stat().st_size,
            metadata={"filename": path.name, **extra_meta},
        )
