"""Coverage for the testlet response model fitter validation paths (testlet.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm import testlet as testlet_mod
from fast_mlsirm.testlet import TestletFit as _TestletFit
from fast_mlsirm.testlet import fit_testlet


def _binary(seed=0, n_persons=40, n_items=6):
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(float)


def _tid():
    return np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)


def test_fit_testlet_converged_rasch_reduction():
    # sigma fixed to zero reduces to ordinary Rasch, which converges
    fit = fit_testlet(
        _binary(), _tid(), model="rasch", max_iter=300, q_gamma=7,
        estimate_sigma=False, init_sigma2=0.0,
    )
    assert isinstance(fit, _TestletFit)
    assert fit.sigma2.shape == (3,)
    assert fit.converged


def test_fit_testlet_non_convergence_warns():
    with pytest.warns(RuntimeWarning):
        fit_testlet(_binary(), _tid(), model="rasch", max_iter=1, q_gamma=7)


def test_fit_testlet_non_convergence_raises_when_required():
    with pytest.raises(RuntimeError):
        fit_testlet(
            _binary(), _tid(), model="rasch", max_iter=1, q_gamma=7,
            require_convergence=True,
        )


def test_fit_testlet_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_testlet(_binary(), _tid(), q_gamma=7)


def test_fit_testlet_rejects_non_2d_responses():
    with pytest.raises(ValueError):
        fit_testlet(np.zeros(6), _tid())


def test_fit_testlet_rejects_non_1d_testlet_id():
    with pytest.raises(ValueError):
        fit_testlet(_binary(), np.zeros((2, 3), dtype=np.int64))


def test_fit_testlet_rejects_empty_items():
    with pytest.raises(ValueError):
        fit_testlet(np.zeros((5, 0)), np.zeros(0, dtype=np.int64))


def test_fit_testlet_rejects_empty_persons():
    with pytest.raises(ValueError):
        fit_testlet(np.zeros((0, 6)), _tid())


def test_fit_testlet_rejects_cell_limit(monkeypatch):
    monkeypatch.setattr(testlet_mod, "MAX_TESTLET_RESPONSE_CELLS", 5)
    with pytest.raises(ValueError):
        fit_testlet(_binary(), _tid(), q_gamma=7)


def test_fit_testlet_rejects_non_numeric_responses():
    with pytest.raises(ValueError):
        fit_testlet(np.array([["a", "b"], ["c", "d"]]), np.array([0, 0]))


def test_fit_testlet_rejects_non_binary_values():
    y = _binary()
    y[0, 0] = 2.0
    with pytest.raises(ValueError):
        fit_testlet(y, _tid())


def test_fit_testlet_rejects_testlet_id_length_mismatch():
    with pytest.raises(ValueError):
        fit_testlet(_binary(n_items=6), np.array([0, 0, 1], dtype=np.int64))


def test_fit_testlet_rejects_non_integer_testlet_id():
    with pytest.raises(ValueError):
        fit_testlet(_binary(), np.array([0.0, 0, 1, 1, 2, 2]))
    with pytest.raises(ValueError):
        fit_testlet(_binary(), np.array([False, False, True, True, False, True]))


def test_fit_testlet_rejects_testlet_id_out_of_range():
    with pytest.raises(ValueError):
        fit_testlet(_binary(), np.array([0, 0, 1, 1, 2, 6], dtype=np.int64))
    with pytest.raises(ValueError):
        fit_testlet(_binary(), np.array([0, 0, 1, 1, 2, -1], dtype=np.int64))


def test_fit_testlet_rejects_bad_max_iter():
    for bad in (True, 1.5, 0, 200_000):
        with pytest.raises(ValueError):
            fit_testlet(_binary(), _tid(), q_gamma=7, max_iter=bad)


def test_fit_testlet_rejects_bad_tol():
    for bad in (True, "x", np.nan, -1.0):
        with pytest.raises(ValueError):
            fit_testlet(_binary(), _tid(), q_gamma=7, tol=bad)


def test_fit_testlet_rejects_bad_q_gamma():
    # #1929: no node-count cap; q_gamma=8 is now accepted, only < 1 is not.
    for bad in (True, 7.5, 0):
        with pytest.raises(ValueError):
            fit_testlet(_binary(), _tid(), q_gamma=bad)


def test_fit_testlet_rejects_bad_init_sigma2():
    for bad in (True, "x", np.nan, -1.0):
        with pytest.raises(ValueError):
            fit_testlet(_binary(), _tid(), q_gamma=7, init_sigma2=bad)
