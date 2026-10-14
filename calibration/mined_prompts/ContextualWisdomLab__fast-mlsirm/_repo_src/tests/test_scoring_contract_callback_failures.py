"""Redacted callback-failure contracts for untrusted scoring inputs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import (
    AssessmentSpecError,
    EnginePolicy,
    ValidationPolicy,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_contract_fixtures.py"))
)
assessment = _FIXTURES["assessment"]


class _RuntimeIndex:
    """Numeric-like value whose conversion fails with arbitrary runtime text."""

    def __index__(self) -> int:
        """Raise a runtime failure that must not escape or be reflected."""
        raise RuntimeError("private numeric callback payload")


class _RuntimeIterable:
    """Collection-like value whose iterator creation fails unexpectedly."""

    def __iter__(self):
        """Raise a runtime failure that must become a stable collection error."""
        raise RuntimeError("private iterator callback payload")


class _RuntimeMapping(Mapping):
    """Mapping-like metadata whose entry iteration fails unexpectedly."""

    def __iter__(self):
        """Raise a runtime failure that must become a stable mapping error."""
        raise RuntimeError("private mapping callback payload")

    def __len__(self) -> int:
        """Advertise one entry without materializing it."""
        return 1

    def __getitem__(self, key):
        """Return an inert value if an implementation requests a key."""
        return "value"


def test_numeric_callback_runtime_errors_are_redacted() -> None:
    """An arbitrary `__index__` failure becomes the documented domain error."""
    with pytest.raises(AssessmentSpecError) as captured:
        EnginePolicy(
            policy_id="engine_policy",
            engine_ids=(),
            allow_human_raters=True,
            allow_automated_raters=False,
            minimum_raters_per_response=_RuntimeIndex(),  # type: ignore[arg-type]
        )

    assert captured.value.code == "invalid_minimum_raters_per_response"
    assert captured.value.path == "$.minimum_raters_per_response"
    assert "private numeric callback payload" not in str(captured.value)


def test_iterable_callback_runtime_errors_are_redacted() -> None:
    """An arbitrary iterator failure cannot leak through policy construction."""
    with pytest.raises(AssessmentSpecError) as captured:
        ValidationPolicy(
            policy_id="validation_policy",
            metric_ids=_RuntimeIterable(),  # type: ignore[arg-type]
            construct_ids=("argument_quality",),
        )

    assert captured.value.code == "invalid_metric_ids"
    assert captured.value.path == "$.metric_ids"
    assert "private iterator callback payload" not in str(captured.value)


def test_mapping_callback_runtime_errors_are_redacted() -> None:
    """An arbitrary mapping-iteration failure becomes a stable metadata error."""
    with pytest.raises(AssessmentSpecError) as captured:
        assessment(metadata=_RuntimeMapping())

    assert captured.value.code == "invalid_metadata_mapping"
    assert captured.value.path == "$.metadata"
    assert "private mapping callback payload" not in str(captured.value)
