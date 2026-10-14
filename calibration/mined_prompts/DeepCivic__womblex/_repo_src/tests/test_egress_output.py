"""Tests for the egress bundle's source_index.parquet (store/egress_output.py)."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from womblex.store.egress_output import (
    SOURCE_INDEX_FILENAME,
    SOURCE_INDEX_SCHEMA,
    raw_key_for,
    read_source_index,
    write_source_index,
)
from womblex.store.run_stamp import RunStamp, read_footer_stamp


def _row(source_hash: str, doc_id: str, filename: str, ext: str, status: str) -> dict:
    raw_key = raw_key_for(source_hash, ext) if status == "resolved" else None
    return {
        "source_hash": source_hash, "doc_id": doc_id, "filename": filename,
        "ext": ext, "raw_key": raw_key, "status": status,
    }


_ROWS = [
    _row("abc123", "doc-1", "report.pdf", ".pdf", "resolved"),
    _row("def456", "doc-2", "register.csv", ".csv", "not_found"),
]

_STAMP = RunStamp("run-A", "1.2.3", "c" * 40, "sha256:abc", "extract", preset="corpus")


def _stamp_of(path: Path) -> dict[str, str]:
    return read_footer_stamp(pq.read_schema(str(path)).metadata)


def test_raw_key_is_relative_to_the_bundle_root():
    assert raw_key_for("abc123", ".pdf") == "sources/abc123.pdf"
    assert raw_key_for("abc123", "") == "sources/abc123"


def test_write_and_read_roundtrip(tmp_path: Path):
    target = write_source_index(_ROWS, tmp_path, stamp=None)
    assert target == tmp_path / SOURCE_INDEX_FILENAME

    table = read_source_index(tmp_path)
    assert table.schema.equals(SOURCE_INDEX_SCHEMA)
    rows = {r["doc_id"]: r for r in table.to_pylist()}
    assert rows["doc-1"]["raw_key"] == "sources/abc123.pdf"
    assert rows["doc-1"]["status"] == "resolved"
    assert rows["doc-2"]["raw_key"] is None
    assert rows["doc-2"]["status"] == "not_found"


def test_raw_key_opens_as_written_from_the_bundle_root(tmp_path: Path):
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "abc123.pdf").write_bytes(b"%PDF-")
    write_source_index(_ROWS, tmp_path, stamp=None)

    key = read_source_index(tmp_path).to_pylist()[0]["raw_key"]
    assert (tmp_path / key).read_bytes() == b"%PDF-"


def test_read_source_index_from_file_path(tmp_path: Path):
    target = write_source_index(_ROWS, tmp_path, stamp=None)
    assert read_source_index(target).num_rows == 2


def test_duplicate_source_hash_keeps_both_rows(tmp_path: Path):
    """Two documents sharing one hash (same bytes, two names) each get a row,
    both naming the same raw_key — the index is not deduplicated by hash."""
    rows = [
        _row("shared", "doc-a", "a.pdf", ".pdf", "resolved"),
        _row("shared", "doc-b", "b.pdf", ".pdf", "resolved"),
    ]
    table = read_source_index(write_source_index(rows, tmp_path, stamp=None))
    assert table.num_rows == 2
    assert {r["raw_key"] for r in table.to_pylist()} == {"sources/shared.pdf"}


def test_empty_rows_is_schema_correct(tmp_path: Path):
    table = read_source_index(write_source_index([], tmp_path, stamp=None))
    assert table.num_rows == 0
    assert table.schema.equals(SOURCE_INDEX_SCHEMA)


def test_write_creates_missing_bundle_dir(tmp_path: Path):
    bundle_dir = tmp_path / "run-123"
    write_source_index(_ROWS, bundle_dir, stamp=None)
    assert (bundle_dir / SOURCE_INDEX_FILENAME).exists()


def test_read_rejects_missing_columns(tmp_path: Path):
    bad_schema = pa.schema([("source_hash", pa.string())])
    target = tmp_path / SOURCE_INDEX_FILENAME
    pq.write_table(pa.table({"source_hash": ["x"]}, schema=bad_schema), str(target))
    with pytest.raises(ValueError, match="missing columns"):
        read_source_index(target)


class TestRunStamp:
    def test_the_index_names_the_exported_run_as_stage_egress(self, tmp_path: Path):
        stamp = _stamp_of(write_source_index(_ROWS, tmp_path, stamp=_STAMP))

        assert stamp["run_id"] == "run-A"
        assert stamp["config_digest"] == "sha256:abc"
        assert stamp["preset"] == "corpus"
        assert stamp["stage"] == "egress"

    def test_attribution_survives_the_index_leaving_the_bundle(self, tmp_path: Path):
        written = write_source_index(_ROWS, tmp_path / "bundle", stamp=_STAMP)
        elsewhere = tmp_path / "elsewhere.parquet"
        elsewhere.write_bytes(written.read_bytes())

        assert _stamp_of(elsewhere)["run_id"] == "run-A"

    def test_an_unnamed_run_writes_the_index_unstamped(self, tmp_path: Path):
        assert _stamp_of(write_source_index(_ROWS, tmp_path, stamp=None)) == {}
