"""Regression tests for observed-score equating evidence admission."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.equating as E
import fast_mlsirm.fitstats as fitstats


_REAL = np.array([0.0, 1.0, 2.0], dtype=np.float64)


class _ExecutableFloat:
    """Record forbidden numeric conversion from object-backed evidence."""

    callbacks: list[str] = []

    def __float__(self) -> float:
        type(self).callbacks.append("__float__")
        raise AssertionError("CALLER_FLOAT_CALLBACK_MUST_NOT_RUN")


_CASES: dict[str, Callable[[object], object]] = {
    "equate_observed_scores": lambda value: E.equate_observed_scores(
        value, _REAL, k_x=2, k_y=2
    ),
    "equate_neat": lambda value: E.equate_neat(
        value, _REAL, _REAL, _REAL, k_x=2, k_y=2, k_v=2
    ),
    "equate_neat_linear": lambda value: E.equate_neat_linear(
        value, _REAL, _REAL, _REAL, k_x=2, k_y=2
    ),
    "loglinear_smooth": lambda value: E.loglinear_smooth(value, degree=1),
    "equate_observed_scores_kernel": lambda value: E.equate_observed_scores_kernel(
        value, _REAL, k_x=2, k_y=2
    ),
    "equating_standard_errors": lambda value: E.equating_standard_errors(
        value,
        _REAL,
        method="mean",
        route="analytic",
        k_x=2,
        k_y=2,
        n_boot=2,
    ),
}


def _forbid_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record any native discovery attempted before evidence admission."""
    calls: list[str] = []

    def forbidden_core():
        calls.append("_core_module")
        raise AssertionError("RUST_DISCOVERY_MUST_NOT_RUN")

    monkeypatch.setattr(fitstats, "_core_module", forbidden_core)
    return calls


@pytest.mark.parametrize("case", tuple(_CASES))
def test_complex_evidence_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Imaginary evidence is never silently projected onto the real line."""
    core_calls = _forbid_core_discovery(monkeypatch)
    evidence = np.array([0.0 + 0.0j, 1.0 + 0.25j, 2.0 + 0.0j])

    with pytest.raises(ValueError, match="real numeric"):
        _CASES[case](evidence)

    assert core_calls == []


@pytest.mark.parametrize("case", tuple(_CASES))
def test_object_evidence_does_not_execute_numeric_conversion(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Object-backed cells fail before caller-defined ``__float__`` runs."""
    core_calls = _forbid_core_discovery(monkeypatch)
    _ExecutableFloat.callbacks.clear()
    evidence = np.array([0.0, _ExecutableFloat(), 2.0], dtype=object)

    with pytest.raises(ValueError, match="real numeric"):
        _CASES[case](evidence)

    assert _ExecutableFloat.callbacks == []
    assert core_calls == []


def test_text_score_evidence_fails_before_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Textual score labels are not reinterpreted as numeric observations."""
    core_calls = _forbid_core_discovery(monkeypatch)

    with pytest.raises(ValueError, match="real numeric"):
        E.equate_observed_scores(["0", "1", "2"], _REAL, k_x=2, k_y=2)

    assert core_calls == []
