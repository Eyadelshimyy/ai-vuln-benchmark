"""Fail-first privacy contract for exposure-control integer validation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import exposure


class _LeakyIntegerControl:
    """Invalid control value whose representation must never reach package errors."""

    def __repr__(self) -> str:
        return "EXPOSURE_CONTROL_SECRET_SENTINEL"


class _ExplodingReprIntegerControl:
    """Invalid control whose representation fails if validation tries to inspect it."""

    def __repr__(self) -> str:
        raise AssertionError("caller __repr__ must not execute during validation")


def _sympson_hetter_with_test_length(value: object) -> None:
    """Reach the public integer-validation boundary with one hostile value."""
    exposure.sympson_hetter(
        np.ones(4, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        test_length=value,
        n_simulees=4,
        max_iter=1,
        q_theta=5,
    )


def test_sympson_hetter_invalid_integer_control_does_not_reflect_value() -> None:
    """Public validation identifies the field without reflecting hostile content."""
    with pytest.raises(ValueError) as captured:
        _sympson_hetter_with_test_length(_LeakyIntegerControl())

    message = str(captured.value)
    assert "test_length must be an integer" in message
    assert "EXPOSURE_CONTROL_SECRET_SENTINEL" not in message


def test_sympson_hetter_invalid_integer_control_never_calls_repr() -> None:
    """Unsupported caller objects must be rejected without invoking ``__repr__``."""
    with pytest.raises(ValueError, match="test_length must be an integer"):
        _sympson_hetter_with_test_length(_ExplodingReprIntegerControl())


@pytest.mark.parametrize("value", [1.25, np.float64(2.5), np.nan, np.inf, True])
def test_as_int_rejects_non_integral_controls_without_reflecting_value(value: object) -> None:
    """Non-integral controls expose only the package-owned field contract."""
    with pytest.raises(ValueError) as captured:
        exposure._as_int("test_length", value, minimum=1, maximum=8)

    assert str(captured.value) == "test_length must be an integer"


def test_as_int_range_error_exposes_bounds_not_rejected_value() -> None:
    """Range errors may describe owned bounds but must not echo caller input."""
    rejected = 987_654_321
    with pytest.raises(ValueError) as captured:
        exposure._as_int("test_length", rejected, minimum=1, maximum=8)

    message = str(captured.value)
    assert "test_length" in message
    assert "1" in message and "8" in message
    assert str(rejected) not in message


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1), (8, 8), (np.int64(4), 4), (np.float64(6.0), 6)],
)
def test_as_int_preserves_accepted_integer_boundaries(value: object, expected: int) -> None:
    """Redaction must not change accepted integer and exact-integral-float semantics."""
    assert exposure._as_int("test_length", value, minimum=1, maximum=8) == expected
