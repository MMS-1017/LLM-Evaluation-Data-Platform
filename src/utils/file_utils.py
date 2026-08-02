"""
File system and Parquet I/O utilities.
All functions log their operations for full observability.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.logging import get_logger

log = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory (and all parents) if it does not exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: Path, extensions: list[str] | None = None) -> list[Path]:
    """
    Recursively list all files in *directory*, optionally filtered by extension.

    Args:
        directory:  Root directory to search.
        extensions: List of extensions to keep, e.g. ['.txt', '.pdf'].
                    If None, all files are returned.

    Returns:
        Sorted list of matching Path objects.
    """
    if not directory.exists():
        log.warning(f"Directory does not exist: {directory}")
        return []

    files = [p for p in directory.rglob("*") if p.is_file()]

    if extensions:
        allowed = {e.lstrip(".").lower() for e in extensions}
        files = [f for f in files if f.suffix.lstrip(".").lower() in allowed]

    log.debug(f"Found {len(files)} files in {directory}")
    return sorted(files)


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    partition_cols: list[str] | None = None,
) -> None:
    """
    Write a pandas DataFrame to Parquet.

    Args:
        df:             DataFrame to write.
        path:           Output path. If partition_cols given, treated as a directory root.
        partition_cols: Optional list of column names for Hive-style partitioning.
    """
    ensure_dir(path.parent if path.suffix else path)
    table = pa.Table.from_pandas(df, preserve_index=False)

    if partition_cols:
        pq.write_to_dataset(table, root_path=str(path), partition_cols=partition_cols)
        log.info(f"Written {len(df)} rows to partitioned dataset at {path}")
    else:
        pq.write_table(table, str(path))
        log.info(f"Written {len(df)} rows to {path}")


def read_parquet(path: Path) -> pd.DataFrame:
    """
    Read a Parquet file or directory of Parquet files into a DataFrame.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")
    df = pd.read_parquet(str(path))
    log.debug(f"Read {len(df)} rows from {path}")
    return df
