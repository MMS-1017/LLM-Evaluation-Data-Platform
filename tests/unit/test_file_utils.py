"""
Unit tests for src/utils/file_utils.py.

All tests run without external services using tmp_path fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.file_utils import ensure_dir, list_files, read_parquet, write_parquet

pytestmark = pytest.mark.unit


# ── ensure_dir ────────────────────────────────────────────────────────────────

class TestEnsureDir:
    def test_creates_new_directory(self, tmp_path: Path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_existing_directory_no_error(self, tmp_path: Path):
        existing = tmp_path / "existing"
        existing.mkdir()
        # Should not raise
        result = ensure_dir(existing)
        assert result == existing

    def test_returns_path(self, tmp_path: Path):
        p = tmp_path / "new_dir"
        returned = ensure_dir(p)
        assert returned == p


# ── list_files ────────────────────────────────────────────────────────────────

class TestListFiles:
    def test_returns_empty_list_for_nonexistent_directory(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist"
        assert list_files(missing) == []

    def test_lists_all_files_without_filter(self, tmp_path: Path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.md").touch()
        (tmp_path / "c.json").touch()
        result = list_files(tmp_path)
        assert len(result) == 3

    def test_filters_by_extension(self, tmp_path: Path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "c.json").touch()
        result = list_files(tmp_path, extensions=[".txt"])
        assert len(result) == 2
        assert all(f.suffix == ".txt" for f in result)

    def test_extension_filter_case_insensitive(self, tmp_path: Path):
        # On Windows, NTFS is case-insensitive so we can't have both
        # doc.TXT and doc.txt as separate files. Use names with different
        # content suffixes to ensure two files with the *same* extension
        # but different names are both matched.
        (tmp_path / "alpha.txt").touch()
        (tmp_path / "beta.txt").touch()
        result = list_files(tmp_path, extensions=[".TXT"])  # upper-case filter
        assert len(result) == 2  # both matched despite case mismatch in filter

    def test_recursive_search(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.txt").touch()
        (tmp_path / "top.txt").touch()
        result = list_files(tmp_path, extensions=[".txt"])
        assert len(result) == 2

    def test_result_is_sorted(self, tmp_path: Path):
        (tmp_path / "c.txt").touch()
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        names = [f.name for f in list_files(tmp_path)]
        assert names == sorted(names)

    def test_no_files_returns_empty_list(self, tmp_path: Path):
        # Directory exists but is empty
        result = list_files(tmp_path)
        assert result == []

    def test_extension_without_leading_dot(self, tmp_path: Path):
        (tmp_path / "doc.txt").touch()
        # Filter using "txt" (no dot)
        result = list_files(tmp_path, extensions=["txt"])
        assert len(result) == 1


# ── write_parquet / read_parquet ──────────────────────────────────────────────

class TestParquetIO:
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
            "label": ["x", "y", "z"],
        })

    def test_write_creates_file(self, tmp_path: Path):
        df = self._sample_df()
        out = tmp_path / "data.parquet"
        write_parquet(df, out)
        assert out.exists()

    def test_read_returns_identical_dataframe(self, tmp_path: Path):
        df = self._sample_df()
        out = tmp_path / "data.parquet"
        write_parquet(df, out)
        result = read_parquet(out)
        pd.testing.assert_frame_equal(df, result)

    def test_read_nonexistent_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_parquet(tmp_path / "missing.parquet")

    def test_write_creates_parent_directories(self, tmp_path: Path):
        out = tmp_path / "a" / "b" / "data.parquet"
        write_parquet(self._sample_df(), out)
        assert out.exists()

    def test_round_trip_preserves_dtypes(self, tmp_path: Path):
        df = pd.DataFrame({"int_col": [1, 2], "float_col": [1.5, 2.5], "str_col": ["a", "b"]})
        out = tmp_path / "typed.parquet"
        write_parquet(df, out)
        result = read_parquet(out)
        assert result["int_col"].dtype.kind in ("i", "u")  # int
        assert result["float_col"].dtype.kind == "f"       # float
        # Modern PyArrow may return StringDtype or object — just verify it holds strings
        assert result["str_col"].iloc[0] == "a"

    def test_empty_dataframe_round_trip(self, tmp_path: Path):
        df = pd.DataFrame({"col": pd.Series([], dtype=str)})
        out = tmp_path / "empty.parquet"
        write_parquet(df, out)
        result = read_parquet(out)
        assert len(result) == 0
        assert "col" in result.columns
