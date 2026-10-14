"""Fail-first safety contracts for model-comparison semantic controls."""

from __future__ import annotations

import pytest

from fast_mlsirm.model_comparison import compare_nonnested_models


class _HostileStringControl:
    """A semantic-control probe that must never be stringified by validation."""

    def __str__(self) -> str:
        raise RuntimeError("MODEL_COMPARISON_RELATION_STR_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("MODEL_COMPARISON_RELATION_REPR_SENTINEL")


class _HostileFloatControl:
    """A numeric-control probe that must never be coerced through ``float``."""

    def __float__(self) -> float:
        raise RuntimeError("MODEL_COMPARISON_FLOAT_SENTINEL")

    def __repr__(self) -> str:
        raise RuntimeError("MODEL_COMPARISON_FLOAT_REPR_SENTINEL")


def _valid_inputs() -> dict[str, object]:
    """Return the smallest otherwise-valid comparison request."""
    return {
        "loglik_a": [0.0, 0.1],
        "loglik_b": [-0.1, 0.0],
        "k_a": 1,
        "k_b": 1,
        "relation": "strictly_non_nested",
    }


def test_relation_rejects_non_string_without_invoking_representation() -> None:
    """Invalid relation objects must fail at the finite-vocabulary boundary."""
    request = _valid_inputs()
    request["relation"] = _HostileStringControl()

    with pytest.raises(ValueError, match=r"^relation must be one of "):
        compare_nonnested_models(**request)


def test_alpha_rejects_custom_numeric_coercion_without_callback() -> None:
    """Alpha validation must not execute caller-defined numeric conversion hooks."""
    request = _valid_inputs()

    with pytest.raises(ValueError, match=r"^alpha must be finite and in \(0, 1\)$"):
        compare_nonnested_models(**request, alpha=_HostileFloatControl())


def test_omega_tolerance_rejects_custom_numeric_coercion_without_callback() -> None:
    """Variance-tolerance validation must not execute caller conversion hooks."""
    request = _valid_inputs()

    with pytest.raises(
        ValueError, match=r"^omega_tol must be finite and non-negative$"
    ):
        compare_nonnested_models(**request, omega_tol=_HostileFloatControl())
