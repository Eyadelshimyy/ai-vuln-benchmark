"""Coverage for Rasch CML estimation and Andersen's LR test (rasch_cml.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.rasch_cml import andersen_lr_test, fit_rasch_cml


def _binary(seed=0, n_persons=80, n_items=5):
    rng = np.random.default_rng(seed)
    return (rng.random((n_persons, n_items)) < 0.5).astype(float)


def test_fit_rasch_cml_happy_path():
    out = fit_rasch_cml(_binary(), max_iter=100, tol=1e-8)
    assert out["beta"].shape == (5,)
    assert out["se"].shape == (5,)
    assert isinstance(out["converged"], bool)
    assert out["n_used"] <= 80


def test_fit_rasch_cml_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_rasch_cml(_binary(), max_iter=100, tol=1e-8)


def test_fit_rasch_cml_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_rasch_cml(np.zeros(5), max_iter=100, tol=1e-8)


def test_fit_rasch_cml_rejects_too_few_items():
    with pytest.raises(ValueError):
        fit_rasch_cml(np.zeros((5, 1)), max_iter=100, tol=1e-8)


def test_fit_rasch_cml_rejects_non_binary():
    y = _binary()
    y[0, 0] = 2.0
    with pytest.raises(ValueError):
        fit_rasch_cml(y, max_iter=100, tol=1e-8)


def test_fit_rasch_cml_rejects_bad_tol():
    with pytest.raises(ValueError):
        fit_rasch_cml(_binary(), tol=0.0, max_iter=100)
    with pytest.raises(ValueError):
        fit_rasch_cml(_binary(), tol=np.inf, max_iter=100)


def test_andersen_lr_happy_path():
    y = _binary()
    group = np.arange(80) % 2
    out = andersen_lr_test(y, group, max_iter=100, tol=1e-8)
    assert out["df"] == 4
    assert out["p_value"] >= 0.0
    assert out["n_used"].shape[0] == 2


def test_andersen_lr_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            andersen_lr_test(_binary(), np.arange(80) % 2, max_iter=100, tol=1e-8)


def test_andersen_lr_rejects_bad_group_shape():
    with pytest.raises(ValueError):
        andersen_lr_test(_binary(), np.zeros((80, 1)), max_iter=100, tol=1e-8)
    with pytest.raises(ValueError):
        andersen_lr_test(_binary(), np.arange(10), max_iter=100, tol=1e-8)


def test_andersen_lr_rejects_non_integer_group():
    with pytest.raises(ValueError):
        andersen_lr_test(_binary(), np.full(80, 0.5), max_iter=100, tol=1e-8)
    with pytest.raises(ValueError):
        andersen_lr_test(_binary(), np.where(np.arange(80) % 2, -1.0, 0.0), max_iter=100, tol=1e-8)


def test_andersen_lr_rejects_single_group():
    with pytest.raises(ValueError):
        andersen_lr_test(_binary(), np.zeros(80, dtype=np.int64), max_iter=100, tol=1e-8)


def test_andersen_lr_rejects_bad_tol():
    with pytest.raises(ValueError):
        andersen_lr_test(_binary(), np.arange(80) % 2, tol=-1.0, max_iter=100)
