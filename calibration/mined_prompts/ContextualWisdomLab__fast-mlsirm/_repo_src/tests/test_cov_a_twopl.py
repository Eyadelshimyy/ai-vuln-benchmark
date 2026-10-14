"""Coverage for the compensatory 2PL fitter validation paths (twopl.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.models import confirmatory
from fast_mlsirm.twopl import TwoPlFit, fit_2pl


def _binary(seed=0, n_persons=60, n_items=6):
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(float)


def test_fit_2pl_happy_path_marshals_result():
    y = _binary()
    fit = fit_2pl(y, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert isinstance(fit, TwoPlFit)
    assert fit.loading.shape == (6, 1)
    assert fit.intercept.shape == (6,)
    assert fit.theta.shape == (60, 1)
    assert fit.corr.shape == (1, 1)
    assert fit.n_dims == 1
    assert fit.termination_reason in ("converged", "max_iter_reached")


def test_fit_2pl_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_2pl(_binary(), 1, q=7, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_2pl(np.zeros(6), 1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_dims_over_gh_cap():
    pattern = np.eye(4, dtype=np.int64)
    pattern = np.vstack([pattern, pattern[:2]])  # 6 items x 4 dims
    with pytest.raises(ValueError):
        fit_2pl(_binary(n_items=6), confirmatory(pattern), q=7, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_infinite_responses():
    y = _binary(n_items=6)
    y[0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_2pl(y, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_non_scalar_or_complex_q():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=np.array([7, 7]), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7j, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_non_integer_q():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7.5, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_zero_q():
    # #1929: no node-count cap; q=8 is now accepted, only q < 1 is rejected.
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=0, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_max_iter_out_of_range():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, max_iter=0, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, max_iter=200_000, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_xi_points_out_of_range_for_qmc():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, node_rule="qmc", xi_points=0, max_iter=500, tol=1e-6, xi_seed=0x9E3779B97F4A7C15)


def test_fit_2pl_rejects_bad_xi_seed():
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=True, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=1.5, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=-1, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_2pl(_binary(), 1, q=7, xi_seed=2**64, max_iter=500, tol=1e-6, xi_points=4000)
