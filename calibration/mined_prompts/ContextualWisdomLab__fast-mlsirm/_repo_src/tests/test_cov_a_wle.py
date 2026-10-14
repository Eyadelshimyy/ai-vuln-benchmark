"""Coverage for Warm's weighted-likelihood ability scoring (wle.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.wle import score_wle, score_wle_poly


def test_score_wle_happy_path_matrix_and_vector():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    out = score_wle(a, b, np.array([[1, 0, 1], [0, 1, 0]], dtype=float))
    assert out["theta"].shape == (2,)
    assert out["se"].shape == (2,)
    assert out["boundary"].dtype == bool
    # single-vector reshape path plus explicit c/d
    single = score_wle(a, b, np.array([1.0, 0.0, 1.0]), c=np.zeros(3), d=np.ones(3))
    assert single["theta"].shape == (1,)


def test_score_wle_explicit_observed_mask():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-0.5, 0.0, 0.5])
    y = np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 1.0]])
    observed = np.array([[True, True, False], [True, True, True]])
    out = score_wle(a, b, y, observed=observed)
    assert out["theta"].shape == (2,)


def test_score_wle_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            score_wle(np.array([1.0]), np.array([0.0]), np.array([[1.0]]))


def test_score_wle_rejects_empty_items():
    with pytest.raises(ValueError):
        score_wle(np.array([]), np.array([]), np.array([[1.0]]))


def test_score_wle_rejects_b_length_mismatch():
    with pytest.raises(ValueError):
        score_wle(np.array([1.0, 1.0, 1.0]), np.array([0.0, 1.0]), np.zeros((2, 3)))


def test_score_wle_rejects_cd_length_mismatch():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        score_wle(a, b, np.zeros((2, 3)), c=np.zeros(2))
    with pytest.raises(ValueError):
        score_wle(a, b, np.zeros((2, 3)), d=np.ones(2))


def test_score_wle_rejects_bad_response_shape():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        score_wle(a, b, np.zeros((2, 4)))


def test_score_wle_rejects_observed_shape():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        score_wle(a, b, np.zeros((2, 3)), observed=np.ones((2, 4), dtype=bool))


def test_score_wle_rejects_non_binary_values():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        score_wle(a, b, np.array([[1.0, 2.0, 0.0]]))


def _poly_cat_params(n_items=3, n_cat=3):
    # strictly decreasing thresholds work for both grm and gpcm smoke calls
    return np.tile(np.array([0.5, -0.5]), (n_items, 1))


def test_score_wle_poly_happy_path():
    slope = np.array([1.0, 1.0, 1.0])
    cat = _poly_cat_params()
    y = np.array([[0, 1, 2], [2, 1, 0]], dtype=float)
    out = score_wle_poly(y, slope, cat, n_cat=3, model="grm")
    assert out["theta"].shape == (2,)
    # vector reshape path + gpcm branch
    out2 = score_wle_poly(np.array([1.0, 0.0, 2.0]), slope, cat, n_cat=3, model="gpcm")
    assert out2["theta"].shape == (1,)


def test_score_wle_poly_explicit_observed():
    slope = np.array([1.0, 1.0, 1.0])
    cat = _poly_cat_params()
    y = np.array([[0.0, 1.0, np.nan]])
    observed = np.array([[True, True, False]])
    out = score_wle_poly(y, slope, cat, n_cat=3, observed=observed)
    assert out["theta"].shape == (1,)


def test_score_wle_poly_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            score_wle_poly(np.array([[0.0]]), np.array([1.0]), np.zeros((1, 2)), n_cat=3)


def test_score_wle_poly_rejects_bad_n_cat():
    with pytest.raises(ValueError):
        score_wle_poly(np.zeros((2, 3)), np.ones(3), np.zeros((3, 2)), n_cat=1)


def test_score_wle_poly_rejects_empty_items():
    with pytest.raises(ValueError):
        score_wle_poly(np.zeros((2, 0)), np.array([]), np.zeros((0, 2)), n_cat=3)


def test_score_wle_poly_rejects_cat_params_shape():
    with pytest.raises(ValueError):
        score_wle_poly(np.zeros((2, 3)), np.ones(3), np.zeros((3, 5)), n_cat=3)


def test_score_wle_poly_rejects_bad_response_shape():
    with pytest.raises(ValueError):
        score_wle_poly(np.zeros((2, 4)), np.ones(3), np.zeros((3, 2)), n_cat=3)


def test_score_wle_poly_rejects_observed_shape():
    with pytest.raises(ValueError):
        score_wle_poly(
            np.zeros((2, 3)), np.ones(3), np.zeros((3, 2)), n_cat=3,
            observed=np.ones((2, 4), dtype=bool),
        )


def test_score_wle_poly_rejects_bad_category_values():
    slope = np.array([1.0, 1.0, 1.0])
    cat = _poly_cat_params()
    with pytest.raises(ValueError):
        score_wle_poly(np.array([[0.0, 1.0, 5.0]]), slope, cat, n_cat=3)
