"""Interoperability regressions for repository bounded-JSON readers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT))
_bounded_json = importlib.import_module("scripts._bounded_json")
parse_json_bounded = _bounded_json.parse_json_bounded
read_json_object = _bounded_json.read_json_object


@pytest.mark.parametrize(
    "payload",
    (
        '{"status":"ok","status":"failed"}',
        '{"outer":{"id":1,"id":2}}',
    ),
    ids=("root", "nested"),
)
def test_parse_json_bounded_rejects_duplicate_object_members(payload: str) -> None:
    """Direct bounded parsing must reject ambiguous duplicate member names."""
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        parse_json_bounded(payload)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_parse_json_bounded_rejects_nonfinite_numeric_constants(constant: str) -> None:
    """Direct bounded parsing must reject non-standard JSON numeric constants."""
    with pytest.raises(ValueError, match="non-finite JSON numeric value"):
        parse_json_bounded(f'{{"value":{constant}}}')


def test_read_json_object_uses_same_strict_value_semantics(tmp_path: Path) -> None:
    """File-backed automation parsing must share duplicate/non-finite rejection."""
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"id":1,"id":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        read_json_object(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON numeric value"):
        read_json_object(nonfinite)
