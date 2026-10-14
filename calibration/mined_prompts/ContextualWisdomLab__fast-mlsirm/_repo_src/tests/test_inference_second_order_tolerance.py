"""Public fail-closed contracts for observed-information PD diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.inference import second_order_test


def test_second_order_rejects_negative_tolerance() -> None:
    """A caller cannot redefine positive definiteness with a negative tolerance."""
    information = np.diag(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ValueError, match="tol must be a finite non-negative float"):
        second_order_test(information, tol=-1.0e-8)


def test_second_order_zero_tolerance_keeps_strict_positive_definition() -> None:
    """Zero tolerance still requires every information eigenvalue to be positive."""
    positive = np.diag(np.array([1.0, 2.0], dtype=np.float64))
    boundary = np.diag(np.array([0.0, 2.0], dtype=np.float64))

    positive_result = second_order_test(positive, tol=0.0)
    boundary_result = second_order_test(boundary, tol=0.0)

    assert positive_result["passed"] is True
    assert positive_result["min_eigenvalue"] == 1.0
    assert boundary_result["passed"] is False
    assert boundary_result["min_eigenvalue"] == 0.0
