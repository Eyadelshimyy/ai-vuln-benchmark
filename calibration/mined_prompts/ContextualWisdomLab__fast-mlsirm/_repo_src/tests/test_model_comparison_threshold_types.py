"""Threshold-type contracts for fail-closed model comparison."""

from __future__ import annotations

import pytest

from fast_mlsirm.model_comparison import compare_nonnested_models


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": "not-a-number"}, "alpha must be finite and in"),
        ({"alpha": object()}, "alpha must be finite and in"),
        ({"omega_tol": "not-a-number"}, "omega_tol must be finite and non-negative"),
        ({"omega_tol": object()}, "omega_tol must be finite and non-negative"),
    ],
)
def test_threshold_values_fail_with_stable_public_errors(kwargs, message):
    """Non-numeric threshold values never leak ``math.isfinite`` TypeErrors."""
    with pytest.raises(ValueError, match=message):
        compare_nonnested_models([0.0, 0.1], [0.2, 0.3], 1, 1, **kwargs)
