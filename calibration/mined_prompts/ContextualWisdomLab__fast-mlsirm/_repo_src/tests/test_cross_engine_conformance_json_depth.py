"""Focused regressions for bounded conformance-manifest JSON depth parsing."""

from __future__ import annotations

import json

import pytest

import fast_mlsirm.cross_engine_conformance as conformance


def _nested_array(depth: int) -> str:
    """Return a syntactically valid JSON array nested to ``depth`` containers."""
    return "[" * depth + "0" + "]" * depth


def test_raw_manifest_depth_rejects_before_json_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-budget nesting must fail before the recursive stdlib decoder runs."""

    def fail_decoder(*args: object, **kwargs: object) -> object:
        raise AssertionError("json.loads must not run for over-budget nesting")

    monkeypatch.setattr(conformance.json, "loads", fail_decoder)

    with pytest.raises(ValueError, match="manifest JSON nesting is too deep"):
        conformance.ConformanceInventory.from_json(
            _nested_array(conformance.MAX_MANIFEST_NESTING + 1)
        )


def test_raw_manifest_depth_accepts_exact_budget() -> None:
    """The raw preflight accepts exactly the documented nesting budget."""
    conformance._validate_raw_manifest_depth(
        _nested_array(conformance.MAX_MANIFEST_NESTING)
    )


def test_raw_manifest_depth_ignores_brackets_inside_strings() -> None:
    """Quoted bracket-like text must not consume the structural depth budget."""
    payload = json.dumps(
        "prefix\\\""
        + "[" * (conformance.MAX_MANIFEST_NESTING + 8)
        + "{" * (conformance.MAX_MANIFEST_NESTING + 8)
        + "suffix"
    )

    conformance._validate_raw_manifest_depth(payload)
