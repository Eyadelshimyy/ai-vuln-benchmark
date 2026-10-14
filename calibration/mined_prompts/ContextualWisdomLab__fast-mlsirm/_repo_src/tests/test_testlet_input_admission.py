"""Trust-boundary regressions for the public testlet estimator."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.testlet import fit_testlet


class _ArrayProbe:
    """Caller-owned array provider that must not run for invalid controls."""

    calls = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        type(self).calls += 1
        raise AssertionError("caller array protocol executed")


class _HostileNumber:
    """Object-array element whose numeric conversion is caller controlled."""

    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller __float__ executed")


def _tid() -> np.ndarray:
    return np.array([0, 0], dtype=np.int64)


def test_invalid_semantic_control_is_rejected_before_response_materialization() -> None:
    """Package-owned controls must fail before caller array protocols run."""

    _ArrayProbe.calls = 0
    with patch(
        "fast_mlsirm.fitstats._core_module",
        side_effect=AssertionError("native core discovery must not run"),
    ):
        with pytest.raises(ValueError, match="model must be either 'rasch' or '2pl'"):
            fit_testlet(_ArrayProbe(), _tid(), model="invalid", q_gamma=7)
    assert _ArrayProbe.calls == 0


def test_complex_responses_are_rejected_before_float_narrowing_or_native_work() -> None:
    """Imaginary response evidence must never be projected onto the real axis."""

    responses = np.array([[0.0 + 1.0j, 1.0], [1.0, 0.0]], dtype=np.complex128)
    with patch(
        "fast_mlsirm.fitstats._core_module",
        side_effect=AssertionError("native core discovery must not run"),
    ):
        with pytest.raises(ValueError, match="responses must be real-valued"):
            fit_testlet(responses, _tid(), q_gamma=7)


def test_object_response_storage_is_rejected_before_element_conversion() -> None:
    """Object storage must fail closed without invoking element conversion hooks."""

    _HostileNumber.calls = 0
    responses = np.array(
        [[_HostileNumber(), 1.0], [0.0, 1.0]],
        dtype=object,
    )
    with patch(
        "fast_mlsirm.fitstats._core_module",
        side_effect=AssertionError("native core discovery must not run"),
    ):
        with pytest.raises(ValueError, match="responses must use real numeric storage"):
            fit_testlet(responses, _tid(), q_gamma=7)
    assert _HostileNumber.calls == 0
