"""Tests for G-NAF PSV → Parquet ingest."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from womblex.ingest.gnaf import _parse_filename, discover_psv_files, ingest_gnaf_directory, ingest_psv
from womblex.ingest.gnaf_schema import ALL_TABLES, SCHEMA_VERSION

# ── Filename parsing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "stem, expected_state, expected_table",
    [
        ("ACT_ADDRESS_DETAIL_psv", "ACT", "ADDRESS_DETAIL"),
        ("NSW_STREET_LOCALITY_psv", "NSW", "STREET_LOCALITY"),
        ("Authority_Code_ADDRESS_TYPE_AUT_psv", None, "ADDRESS_TYPE_AUT"),
        ("Authority_Code_FLAT_TYPE_AUT_psv", None, "FLAT_TYPE_AUT"),
        ("OT_MB_2021_psv", "OT", "MB_2021"),
        ("random_file", None, None),
    ],
)
def test_parse_filename(stem: str, expected_state: str | None, expected_table: str | None):
    state, table = _parse_filename(stem)
    assert state == expected_state
    assert table == expected_table


# ── Single file ingest ──────────────────────────────────────────────────────


def _write_psv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_ingest_authority_code_psv(tmp_path: Path):
    """Authority Code file → Parquet with correct schema and metadata."""
    psv = tmp_path / "Authority_Code_ADDRESS_TYPE_AUT_psv.psv"
    _write_psv(psv, [
        "R|RURAL|RURAL ADDRESS",
        "U|URBAN|URBAN ADDRESS",
    ])

    out_dir = tmp_path / "out"
    result = ingest_psv(psv, out_dir, compute_md5=True)

    assert result is not None
    assert result.exists()

    table = pq.read_table(str(result))
    assert table.num_rows == 2
    assert table.column_names == ["code", "name", "description"]

    # Check provenance metadata.
    meta = table.schema.metadata
    assert meta[b"gnaf.schema_version"] == SCHEMA_VERSION.encode()
    assert meta[b"gnaf.table_name"] == b"ADDRESS_TYPE_AUT"
    assert b"gnaf.state" not in meta  # Authority Code has no state
    assert meta[b"gnaf.row_count"] == b"2"
    assert b"gnaf.source_md5" in meta

    # Zero semantic mutation: values preserved exactly.
    assert table.column("code").to_pylist() == ["R", "U"]
    assert table.column("name").to_pylist() == ["RURAL", "URBAN"]


def test_ingest_standard_psv(tmp_path: Path):
    """Standard state file → Parquet with state metadata."""
    psv = tmp_path / "ACT_STATE_psv.psv"
    _write_psv(psv, [
        "STATE1|2020-01-01||Australian Capital Territory|ACT",
    ])

    out_dir = tmp_path / "out"
    result = ingest_psv(psv, out_dir)

    assert result is not None
    table = pq.read_table(str(result))
    assert table.num_rows == 1
    assert table.column_names == ALL_TABLES["STATE"]

    meta = table.schema.metadata
    assert meta[b"gnaf.state"] == b"ACT"
    assert meta[b"gnaf.table_name"] == b"STATE"


def test_ingest_skips_header_row(tmp_path: Path):
    """A G-NAF header row is detected and dropped, not read as data.

    The distribution ships each PSV with an uppercase header naming its
    columns; it must never leak in as data row 0. A leading BOM on the header
    (as ACT_ADDRESS_SITE carries) does not defeat detection.
    """
    psv = tmp_path / "ACT_STATE_psv.psv"
    _write_psv(psv, [
        "﻿STATE_PID|DATE_CREATED|DATE_RETIRED|STATE_NAME|STATE_ABBREVIATION",
        "8|2021-11-23||Australian Capital Territory|ACT",
    ])

    result = ingest_psv(psv, tmp_path / "out")
    assert result is not None

    table = pq.read_table(str(result))
    assert table.num_rows == 1  # header dropped, one data row remains
    assert table.column("state_pid").to_pylist() == ["8"]
    assert table.column("state_abbreviation").to_pylist() == ["ACT"]


def test_ingest_headerless_file_keeps_first_row(tmp_path: Path):
    """A headerless PSV is read as-is — the first line is data, not a header."""
    psv = tmp_path / "ACT_STATE_psv.psv"
    _write_psv(psv, [
        "8|2021-11-23||Australian Capital Territory|ACT",
    ])

    result = ingest_psv(psv, tmp_path / "out")
    assert result is not None

    table = pq.read_table(str(result))
    assert table.num_rows == 1
    assert table.column("state_pid").to_pylist() == ["8"]


def test_ingest_unknown_filename_skipped(tmp_path: Path):
    """Unrecognised filename pattern returns None."""
    psv = tmp_path / "random_data.psv"
    _write_psv(psv, ["a|b|c"])

    result = ingest_psv(psv, tmp_path / "out")
    assert result is None


def test_ingest_unknown_table_skipped(tmp_path: Path):
    """Recognised pattern but unknown table name returns None."""
    psv = tmp_path / "ACT_NONEXISTENT_TABLE_psv.psv"
    _write_psv(psv, ["a|b|c"])

    result = ingest_psv(psv, tmp_path / "out")
    assert result is None


def test_ingest_no_md5(tmp_path: Path):
    """MD5 computation can be disabled."""
    psv = tmp_path / "Authority_Code_FLAT_TYPE_AUT_psv.psv"
    _write_psv(psv, ["APT|APARTMENT|APARTMENT"])

    result = ingest_psv(psv, tmp_path / "out", compute_md5=False)
    assert result is not None

    meta = pq.read_table(str(result)).schema.metadata
    assert b"gnaf.source_md5" not in meta


# ── Directory ingest ────────────────────────────────────────────────────────


def test_discover_psv_files(tmp_path: Path):
    """discover_psv_files finds .psv files recursively."""
    (tmp_path / "sub").mkdir()
    _write_psv(tmp_path / "a.psv", ["x"])
    _write_psv(tmp_path / "sub" / "b.psv", ["y"])
    (tmp_path / "c.txt").write_text("not a psv")

    found = discover_psv_files(tmp_path)
    assert len(found) == 2
    assert all(p.suffix == ".psv" for p in found)


def test_ingest_gnaf_directory(tmp_path: Path):
    """End-to-end directory ingest."""
    src = tmp_path / "gnaf"
    src.mkdir()
    _write_psv(src / "Authority_Code_STREET_CLASS_AUT_psv.psv", [
        "C|CONFIRMED|CONFIRMED STREET",
    ])
    _write_psv(src / "ACT_STATE_psv.psv", [
        "STATE1|2020-01-01||Australian Capital Territory|ACT",
    ])
    # This one should be skipped (unknown pattern).
    _write_psv(src / "readme.psv", ["some|data"])

    out = tmp_path / "out"
    written = ingest_gnaf_directory(src, out)

    assert len(written) == 2
    assert all(p.suffix == ".parquet" for p in written)
    assert (out / "Authority_Code_STREET_CLASS_AUT_psv.parquet").exists()
    assert (out / "ACT_STATE_psv.parquet").exists()


def test_empty_values_preserved(tmp_path: Path):
    """Empty strings remain empty strings, not nulls."""
    psv = tmp_path / "ACT_ADDRESS_ALIAS_psv.psv"
    # 7 columns: address_alias_pid, date_created, date_retired,
    #            principal_pid, alias_pid, alias_type_code, alias_comment
    _write_psv(psv, [
        "ALIAS1|2020-01-01||PRINC1|ALIAS2|SYN|",
    ])

    result = ingest_psv(psv, tmp_path / "out")
    assert result is not None

    table = pq.read_table(str(result))
    row = table.to_pydict()
    # date_retired and alias_comment are empty — must be "" not None.
    assert row["date_retired"] == [""]
    assert row["alias_comment"] == [""]


def test_ingest_column_count_mismatch(tmp_path: Path):
    """Column count mismatch between data and schema returns None."""
    psv = tmp_path / "ACT_STATE_psv.psv"
    # STATE expects 5 columns, provide only 3.
    _write_psv(psv, ["STATE1|2020-01-01|ACT"])

    result = ingest_psv(psv, tmp_path / "out")
    assert result is None


def test_ingest_corrupt_file(tmp_path: Path):
    """Unreadable file returns None rather than raising."""
    psv = tmp_path / "ACT_STATE_psv.psv"
    psv.write_bytes(b"\x00\x01\x02\xff\xfe")

    result = ingest_psv(psv, tmp_path / "out")
    assert result is None


def test_ingest_empty_directory(tmp_path: Path):
    """Directory with no PSV files returns empty list."""
    src = tmp_path / "empty"
    src.mkdir()
    (src / "readme.txt").write_text("not a psv")

    written = ingest_gnaf_directory(src, tmp_path / "out")
    assert written == []
