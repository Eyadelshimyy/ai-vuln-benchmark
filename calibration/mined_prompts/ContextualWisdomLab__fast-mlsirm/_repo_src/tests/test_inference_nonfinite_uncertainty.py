"""Fail-first scientific contracts for non-finite covariance uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm._core as core
from fast_mlsirm.inference import standard_errors_from_vcov, vcov_from_hessian


def test_standard_errors_preserve_invalid_and_infinite_uncertainty() -> None:
    """Undefined/infinite covariance must never be converted into zero uncertainty."""
    vcov = np.diag(
        np.array([4.0, 0.0, -1.0, np.nan, np.inf, -np.inf], dtype=np.float64)
    )

    result = standard_errors_from_vcov(vcov)

    assert np.array_equal(result[:3], np.array([2.0, 0.0, 0.0]))
    assert np.isnan(result[3])
    assert np.isposinf(result[4])
    assert np.isneginf(result[5])


def test_direct_rust_standard_errors_match_public_nonfinite_semantics() -> None:
    """Rust and public wrappers must agree on non-finite uncertainty semantics."""
    vcov = np.diag(np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float64))

    rust = np.asarray(core.standard_errors_from_vcov(vcov), dtype=np.float64)
    public = standard_errors_from_vcov(vcov)

    assert np.array_equal(rust[:1], public[:1])
    assert np.isnan(rust[1]) and np.isnan(public[1])
    assert np.isposinf(rust[2]) and np.isposinf(public[2])
    assert np.isneginf(rust[3]) and np.isneginf(public[3])


@pytest.mark.parametrize(
    "hessian",
    [
        np.array([[np.nan]], dtype=np.float64),
        np.array([[np.inf]], dtype=np.float64),
        np.array([[-np.inf]], dtype=np.float64),
    ],
)
def test_vcov_nonfinite_hessian_fails_closed(hessian: np.ndarray) -> None:
    """Non-finite observed information must not yield an uncontrolled covariance artifact."""
    with pytest.raises(ValueError, match=r"hessian.*finite"):
        vcov_from_hessian(hessian)
