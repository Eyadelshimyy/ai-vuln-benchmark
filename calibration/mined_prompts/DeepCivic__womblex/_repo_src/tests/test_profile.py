"""Tests for column-level schema inference."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from womblex.profile import ColumnProfile, profile_dataframe, profile_file


def _col(profile, name: str) -> ColumnProfile:
    return next(c for c in profile.columns if c.name == name)


def test_infers_integer_column():
    df = pd.DataFrame({"id": ["1", "2", "3", "4"]})
    profile = profile_dataframe(df)
    col = _col(profile, "id")
    assert col.inferred_type == "integer"
    assert col.min_value == "1"
    assert col.max_value == "4"
    assert col.is_unique is True
    assert col.null_count == 0


def test_infers_float_column():
    df = pd.DataFrame({"score": ["0.1", "1.5", "-2.0", "1e3"]})
    col = _col(profile_dataframe(df), "score")
    assert col.inferred_type == "float"
    assert float(col.min_value) == -2.0
    assert float(col.max_value) == 1000.0


def test_infers_boolean_column():
    df = pd.DataFrame({"active": ["true", "false", "True", "FALSE", "yes", "no"]})
    col = _col(profile_dataframe(df), "active")
    assert col.inferred_type == "boolean"


def test_zero_one_stays_integer_not_boolean():
    # Conservative: 0/1 is more useful as integer; user can refine.
    df = pd.DataFrame({"flag": ["0", "1", "0", "1"]})
    col = _col(profile_dataframe(df), "flag")
    assert col.inferred_type == "integer"


def test_infers_iso_date_column():
    df = pd.DataFrame({"created": ["2024-01-01", "2024-02-15", "2023-12-31"]})
    col = _col(profile_dataframe(df), "created")
    assert col.inferred_type == "date"
    assert col.min_value == "2023-12-31"
    assert col.max_value == "2024-02-15"


def test_infers_datetime_column():
    df = pd.DataFrame(
        {"ts": ["2024-01-01T12:00:00", "2024-01-02T08:30:00Z", "2024-03-01T00:00:00"]}
    )
    col = _col(profile_dataframe(df), "ts")
    assert col.inferred_type == "datetime"


def test_string_column_reports_max_length():
    df = pd.DataFrame({"name": ["alice", "bob", "alexandria"]})
    col = _col(profile_dataframe(df), "name")
    assert col.inferred_type == "string"
    assert col.max_length == len("alexandria")


def test_nullability():
    df = pd.DataFrame({"x": ["1", "", "3", ""]})
    col = _col(profile_dataframe(df), "x")
    assert col.null_count == 2
    assert col.null_fraction == 0.5
    assert col.inferred_type == "integer"
    # is_unique requires zero nulls
    assert col.is_unique is False


def test_empty_column():
    df = pd.DataFrame({"empty": ["", "", ""]})
    col = _col(profile_dataframe(df), "empty")
    assert col.inferred_type == "empty"
    assert col.is_constant is True
    assert col.unique_count == 0


def test_constant_column():
    df = pd.DataFrame({"const": ["yes", "yes", "yes"]})
    col = _col(profile_dataframe(df), "const")
    assert col.is_constant is True
    assert col.unique_count == 1


def test_unique_with_no_nulls_is_pk_candidate():
    df = pd.DataFrame({"id": ["a", "b", "c"]})
    col = _col(profile_dataframe(df), "id")
    assert col.is_unique is True


def test_mixed_column_falls_back_to_string():
    df = pd.DataFrame({"messy": ["1", "2.5", "hello", "2024-01-01"]})
    col = _col(profile_dataframe(df), "messy")
    assert col.inferred_type == "string"


def test_profile_csv(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text("id,name,age\n1,alice,30\n2,bob,\n3,carol,25\n")
    profiles = profile_file(p)
    assert len(profiles) == 1
    tp = profiles[0]
    assert tp.row_count == 3
    assert tp.column_count == 3
    assert _col(tp, "id").inferred_type == "integer"
    assert _col(tp, "name").inferred_type == "string"
    age = _col(tp, "age")
    assert age.inferred_type == "integer"
    assert age.null_count == 1


def test_profile_csv_sampling(tmp_path: Path):
    p = tmp_path / "big.csv"
    rows = ["id,n"]
    rows.extend(f"{i},x" for i in range(1, 1001))
    p.write_text("\n".join(rows) + "\n")
    profiles = profile_file(p, sample_rows=100)
    tp = profiles[0]
    assert tp.row_count == 1000
    assert tp.sampled_rows == 100


def test_profile_excel_multi_sheet(tmp_path: Path):
    pytest.importorskip("openpyxl")
    p = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}).to_excel(w, sheet_name="people", index=False)
        pd.DataFrame({"x": [0.1, 0.2, 0.3]}).to_excel(w, sheet_name="metrics", index=False)
    profiles = profile_file(p)
    assert {p.sheet_name for p in profiles} == {"people", "metrics"}
    people = next(p for p in profiles if p.sheet_name == "people")
    assert _col(people, "id").inferred_type == "integer"


def test_profile_parquet_uses_native_types(tmp_path: Path):
    pytest.importorskip("pyarrow")
    p = tmp_path / "x.parquet"
    pd.DataFrame(
        {
            "id": pd.array([1, 2, 3], dtype="int64"),
            "score": pd.array([0.5, 1.5, 2.5], dtype="float64"),
            "name": ["a", "b", "c"],
        }
    ).to_parquet(p)
    profiles = profile_file(p)
    tp = profiles[0]
    assert _col(tp, "id").inferred_type == "integer"
    assert _col(tp, "score").inferred_type == "float"
    assert _col(tp, "name").inferred_type == "string"


def test_profile_ndjson(tmp_path: Path):
    p = tmp_path / "x.ndjson"
    p.write_text('{"id":1,"name":"a"}\n{"id":2,"name":"b"}\n{"id":3,"name":"c"}\n')
    profiles = profile_file(p)
    tp = profiles[0]
    assert tp.row_count == 3
    assert _col(tp, "id").inferred_type == "integer"


def test_profile_unsupported_extension(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        profile_file(p)


def test_profile_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        profile_file(tmp_path / "nope.csv")


def test_cli_profile_human_output(tmp_path: Path, capsys):
    from womblex.cli import main as cli_main

    p = tmp_path / "x.csv"
    p.write_text("id,name\n1,alice\n2,bob\n")
    rc = cli_main(["profile", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "id" in out
    assert "integer" in out
    assert "string" in out


def test_cli_profile_json_output(tmp_path: Path, capsys):
    from womblex.cli import main as cli_main

    p = tmp_path / "x.csv"
    p.write_text("id,name\n1,alice\n2,bob\n")
    rc = cli_main(["profile", str(p), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["row_count"] == 2
    cols = {c["name"]: c for c in parsed[0]["columns"]}
    assert cols["id"]["inferred_type"] == "integer"
