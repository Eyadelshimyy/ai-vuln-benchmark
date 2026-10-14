"""Coverage for the graded response model fitter validation paths (grm.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.grm import GrmFit, fit_grm
from fast_mlsirm.models import confirmatory


def _poly(n_cat=3, seed=0, n_persons=45, n_items=4):
    """Random integer-category responses with every category present per item."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    # deterministically guarantee each category appears in every column
    for k in range(n_cat):
        y[k, :] = k
    return y


def test_fit_grm_happy_path_marshals_result():
    fit = fit_grm(_poly(), 3, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert isinstance(fit, GrmFit)
    assert fit.slope.shape == (4, 1)
    assert fit.threshold.shape == (4, 2)
    assert fit.theta.shape == (45, 1)
    assert fit.n_cat == 3
    assert fit.n_dims == 1


def test_fit_grm_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_grm(_poly(), 3, 1, q=7, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_grm(np.zeros(4), 3, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_dims_over_gh_cap():
    with pytest.raises(ValueError):
        fit_grm(_poly(n_items=4), 3, confirmatory(np.eye(4, dtype=np.int64)), q=7, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_bad_n_cat_type():
    with pytest.raises(ValueError):
        fit_grm(_poly(), np.array([3, 3]), model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3j, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3.5, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_n_cat_out_of_range():
    with pytest.raises(ValueError):
        fit_grm(_poly(), 1, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 65, model=1, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_zero_q():
    # #1929: no node-count cap; q=8 is now accepted, only q < 1 is rejected.
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=0, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_max_iter_out_of_range():
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, max_iter=0, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, max_iter=200_000, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_xi_points_out_of_range():
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, xi_points=0, max_iter=500, tol=1e-6, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, xi_points=200_001, max_iter=500, tol=1e-6, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_bad_xi_seed():
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, xi_seed=True, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, xi_seed=1.5, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, xi_seed=-1, max_iter=500, tol=1e-6, xi_points=4000)
    with pytest.raises(ValueError):
        fit_grm(_poly(), 3, 1, q=7, xi_seed=2**64, max_iter=500, tol=1e-6, xi_points=4000)


def test_fit_grm_all_missing_skips_category_check():
    # exercises the "no observed responses" branch before the core rejects it
    y = np.full((10, 4), np.nan)
    with pytest.raises(ValueError):
        fit_grm(y, 3, 1, q=7, max_iter=3, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_rejects_bad_category_values():
    y = _poly(3)
    y[5, 0] = 0.5  # non-integer observed category
    with pytest.raises(ValueError):
        fit_grm(y, 3, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    y2 = _poly(3)
    y2[5, 0] = 3  # category >= n_cat
    with pytest.raises(ValueError):
        fit_grm(y2, 3, 1, q=7, max_iter=5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
