"""Coverage for the linear logistic test model fitter (lltm.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.lltm import LltmFit, fit_lltm


def _design():
    return np.array(
        [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, 2]], dtype=np.float64
    )


def _binary(seed=0, n_persons=40, n_items=6):
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(float)


def test_fit_lltm_happy_path_marshals_result():
    fit = fit_lltm(_binary(), _design(), max_iter=5, tol=1e-6)
    assert isinstance(fit, LltmFit)
    assert fit.eta.shape == (2,)
    assert fit.b.shape == (6,)
    assert fit.theta.shape == (40,)


def test_fit_lltm_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_lltm(_binary(), _design(), max_iter=500, tol=1e-6)


def test_fit_lltm_rejects_non_2d_responses():
    with pytest.raises(ValueError):
        fit_lltm(np.zeros(6), _design(), max_iter=500, tol=1e-6)


def test_fit_lltm_rejects_non_2d_design():
    with pytest.raises(ValueError):
        fit_lltm(_binary(), np.zeros(6), max_iter=500, tol=1e-6)


def test_fit_lltm_rejects_design_row_mismatch():
    with pytest.raises(ValueError):
        fit_lltm(_binary(n_items=6), _design()[:5], max_iter=500, tol=1e-6)
