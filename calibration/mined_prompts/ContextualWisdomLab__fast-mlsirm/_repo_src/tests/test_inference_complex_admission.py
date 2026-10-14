"""Regression coverage for complex curvature/covariance admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import _core
from fast_mlsirm.inference import (
    second_order_test,
    standard_errors_from_vcov,
    vcov_from_hessian,
)


def _unexpected_native_dispatch(*args, **kwargs):
    """Fail if an invalid complex matrix reaches Rust-owned arithmetic."""
    raise AssertionError("complex matrix reached native inference arithmetic")


def test_second_order_test_rejects_complex_hessian_before_native_dispatch(monkeypatch):
    """Complex Hessians must not be narrowed to real values before eigendiagnostics."""
    monkeypatch.setattr(_core, "second_order_test", _unexpected_native_dispatch)
    hessian = np.array([[2.0 + 1.0j, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="hessian must be real-valued"):
        second_order_test(hessian)


def test_vcov_from_hessian_rejects_complex_hessian_before_native_dispatch(monkeypatch):
    """Complex Hessians must fail before Rust inversion/pseudoinversion."""
    monkeypatch.setattr(_core, "vcov_from_hessian", _unexpected_native_dispatch)
    hessian = np.array([[2.0, 0.0], [0.0, 1.0 + 1.0j]])

    with pytest.raises(ValueError, match="hessian must be real-valued"):
        vcov_from_hessian(hessian)


def test_standard_errors_reject_complex_vcov_before_native_dispatch(monkeypatch):
    """Complex covariance values must fail before Rust diagonal reduction."""
    monkeypatch.setattr(_core, "standard_errors_from_vcov", _unexpected_native_dispatch)
    vcov = np.array([[0.25, 0.0], [0.0, 1.0 + 1.0j]])

    with pytest.raises(ValueError, match="vcov must be real-valued"):
        standard_errors_from_vcov(vcov)
