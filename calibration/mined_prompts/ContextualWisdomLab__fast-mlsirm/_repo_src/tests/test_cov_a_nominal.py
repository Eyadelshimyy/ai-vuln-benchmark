"""Coverage for the nominal response model fitter (nominal.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.models import confirmatory
from fast_mlsirm.nominal import NominalResponseFit, fit_nominal


def _poly(n_cat=3, seed=2, n_persons=45, n_items=4):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    for k in range(n_cat):
        y[k, :] = k
    return y


def test_fit_nominal_happy_path_marshals_result():
    fit = fit_nominal(_poly(), 3, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert isinstance(fit, NominalResponseFit)
    assert fit.slope.shape == (4, 3, 1)
    assert fit.intercept.shape == (4, 3)
    assert fit.theta.shape == (45, 1)
    assert fit.n_cat == 3
    assert fit.n_dims == 1


def test_fit_nominal_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_nominal(_poly(), 3, 1, q=7, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_nominal(np.zeros(4), 3, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_dims_over_gh_cap():
    with pytest.raises(ValueError):
        fit_nominal(_poly(n_items=4), 3, confirmatory(np.eye(4, dtype=np.int64)), q=7, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_bad_n_cat_type():
    with pytest.raises(ValueError):
        fit_nominal(_poly(), np.array([3, 3]), model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3j, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3.5, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_n_cat_out_of_range():
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 1, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 65, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_zero_q():
    # #1929: no node-count cap; q=8 is now accepted, only q < 1 is rejected.
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, q=0, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_max_iter_out_of_range():
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, q=7, max_iter=0, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, q=7, max_iter=200_000, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_xi_points_out_of_range_for_qmc():
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, node_rule="qmc", xi_points=0, max_iter=500, tol=1e-6, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, node_rule="qmc", xi_points=200_001, max_iter=500, tol=1e-6, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_bad_xi_seed():
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, q=7, xi_seed=True, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, q=7, xi_seed=1.5, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, q=7, xi_seed=-1, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_nominal(_poly(), 3, 1, q=7, xi_seed=2**64, max_iter=500, tol=1e-6, xi_points=4000)


def test_fit_nominal_all_missing_skips_category_check():
    y = np.full((10, 4), np.nan)
    with pytest.raises(ValueError):
        fit_nominal(y, 3, 1, q=7, max_iter=3, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_non_integer_category():
    y = _poly(3)
    y[5, 0] = 0.5
    with pytest.raises(ValueError):
        fit_nominal(y, 3, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_nominal_rejects_category_over_range():
    y = _poly(3)
    y[5, 0] = 3
    with pytest.raises(ValueError):
        fit_nominal(y, 3, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
