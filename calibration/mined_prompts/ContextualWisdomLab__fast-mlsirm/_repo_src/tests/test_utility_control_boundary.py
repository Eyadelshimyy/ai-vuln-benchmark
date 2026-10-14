"""Trust-boundary regressions for classical selection utility controls."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_mlsirm.utility import selection_utility, taylor_russell


class _HostileFloat:
    """Arbitrary float-protocol object whose callback must stay unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller __float__ callback must not execute")


class _HostileFloatSubclass(float):
    """Float subclass whose conversion callback must stay unreachable."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller float-subclass __float__ callback must not execute")


_UTILITY_CONTROL_CALLS = (
    lambda value: selection_utility(value, 1.0, 0.5, 0.5),
    lambda value: selection_utility(1.0, value, 0.5, 0.5),
    lambda value: selection_utility(1.0, 1.0, value, 0.5),
    lambda value: selection_utility(1.0, 1.0, 0.5, value),
    lambda value: selection_utility(1.0, 1.0, 0.5, 0.5, value, 1.0),
    lambda value: selection_utility(1.0, 1.0, 0.5, 0.5, 0.0, value),
    lambda value: taylor_russell(value, 0.5, 0.5),
    lambda value: taylor_russell(0.4, value, 0.5),
    lambda value: taylor_russell(0.4, 0.5, value),
)


@pytest.mark.parametrize("call", _UTILITY_CONTROL_CALLS)
@pytest.mark.parametrize("hostile_type", (_HostileFloat, _HostileFloatSubclass))
def test_utility_rejects_float_protocol_objects_without_callbacks(
    call, hostile_type
) -> None:
    """Every scalar position rejects hostile conversion without callback execution."""
    hostile_type.reset()

    with pytest.raises(ValueError, match="must be a finite real number"):
        call(hostile_type())

    assert hostile_type.calls == 0


@pytest.mark.parametrize("call", _UTILITY_CONTROL_CALLS)
@pytest.mark.parametrize("value", (True, False, math.inf, -math.inf, math.nan))
def test_utility_rejects_boolean_and_nonfinite_controls(call, value) -> None:
    """Every scalar position rejects boolean and non-finite control values."""
    with pytest.raises(ValueError, match="must be a finite real number"):
        call(value)


@pytest.mark.parametrize("call", _UTILITY_CONTROL_CALLS)
def test_utility_rejects_unrepresentable_builtin_integers(call) -> None:
    """Exact huge integers must become stable package-owned validation errors."""
    with pytest.raises(ValueError, match="must be a finite real number"):
        call(10**400)


def test_numpy_real_scalars_preserve_native_results() -> None:
    """Trusted NumPy real scalars produce the same Rust-owned results as floats."""
    py_utility = selection_utility(10.0, 2.0, 0.4, 0.5, 1.0, 2.0)
    np_utility = selection_utility(
        np.float64(10.0),
        np.float64(2.0),
        np.float64(0.4),
        np.float64(0.5),
        np.float64(1.0),
        np.float64(2.0),
    )
    assert np_utility == py_utility

    py_tr = taylor_russell(0.4, 0.5, 0.6)
    np_tr = taylor_russell(np.float64(0.4), np.float64(0.5), np.float64(0.6))
    assert np_tr == py_tr
