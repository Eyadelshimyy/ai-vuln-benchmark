"""Tests for fail-closed replay of nested essay scoring result contracts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import runpy
from typing import Any

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.essay import build_essay_score_report

_REPORT_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_essay_reporting.py"))
)
essay_request = _REPORT_FIXTURES["essay_request"]
result_bundle = _REPORT_FIXTURES["result_bundle"]


def assert_error(code: str, callback: Callable[[], object]) -> None:
    """Assert one stable nested-replay error code."""
    with pytest.raises(AssessmentSpecError) as caught:
        callback()
    assert caught.value.code == code


def build_report_after_mutation(
    request: Any,
    result: Any,
    descriptor: Any,
) -> object:
    """Attempt report construction after a deliberate contract mutation."""
    return build_essay_score_report(
        report_id="essay_score_report",
        request=request,
        result=result,
        engine=descriptor,
    )


@pytest.mark.parametrize(
    ("field_name", "replacement", "error_code"),
    (
        ("criterion_id", "undeclared_criterion", "unknown_criterion_id"),
        ("score_category", 999, "unknown_score_category"),
        ("reason_code", "unexpected_reason", "unexpected_reason_code"),
    ),
)
def test_report_replays_nested_observation_semantics(
    field_name: str,
    replacement: Any,
    error_code: str,
) -> None:
    """Mutated criterion and score semantics fail through shared builders."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    object.__setattr__(result.observations[0], field_name, replacement)

    assert_error(
        error_code,
        lambda: build_report_after_mutation(request, result, descriptor),
    )


def test_report_rejects_non_tuple_observation_collection() -> None:
    """A result cannot replace its immutable observation tuple after creation."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    object.__setattr__(result, "observations", list(result.observations))

    assert_error(
        "invalid_essay_report_observations",
        lambda: build_report_after_mutation(request, result, descriptor),
    )


def test_report_rejects_untyped_nested_observation() -> None:
    """Every nested observation must remain a governed observation value."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    object.__setattr__(
        result,
        "observations",
        (object(), result.observations[1]),
    )

    assert_error(
        "invalid_essay_report_observation",
        lambda: build_report_after_mutation(request, result, descriptor),
    )


def test_report_rejects_observation_schema_mutation() -> None:
    """A nested schema mutation cannot survive canonical observation replay."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    object.__setattr__(result.observations[0], "schema_version", "9.9")

    assert_error(
        "essay_report_observation_replay_mismatch",
        lambda: build_report_after_mutation(request, result, descriptor),
    )


def test_report_rejects_result_schema_mutation() -> None:
    """A result schema mutation cannot survive canonical result replay."""
    request = essay_request()
    _engine, descriptor, result = result_bundle(request)
    object.__setattr__(result, "schema_version", "9.9")

    assert_error(
        "essay_report_result_replay_mismatch",
        lambda: build_report_after_mutation(request, result, descriptor),
    )
