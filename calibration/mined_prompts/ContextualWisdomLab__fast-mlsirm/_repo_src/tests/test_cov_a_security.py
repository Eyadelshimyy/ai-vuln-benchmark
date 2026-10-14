"""Coverage for answer-copying detection statistics (security.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.security import (
    GbtResult,
    KIndexResult,
    KVariantsResult,
    WollackOmegaResult,
    gbt,
    k_index,
    k_variants,
    wollack_omega,
)


def _copier():
    return np.array([0, 1, 2, 0, 1])


def _source():
    return np.array([0, 1, 0, 1, 2])


def _probs():
    return np.full((5, 3), 1.0 / 3.0)


def _responses(seed=0, n_persons=60, n_items=20):
    rng = np.random.default_rng(seed)
    y = (rng.random((n_persons, n_items)) < 0.6).astype(float)
    y[1, :] = 0.0  # source has all-incorrect answers (ws > 0)
    return y


# --------------------------- wollack_omega ---------------------------

def test_wollack_omega_happy_path_matrix_and_flat():
    res = wollack_omega(_copier(), _source(), _probs(), 3)
    assert isinstance(res, WollackOmegaResult)
    flat = wollack_omega(_copier(), _source(), _probs().ravel(), 3)
    assert flat.observed_matches == res.observed_matches


def test_wollack_omega_rejects_bad_n_options():
    with pytest.raises(ValueError):
        wollack_omega(_copier(), _source(), _probs(), 3.0)
    with pytest.raises(ValueError):
        wollack_omega(_copier(), _source(), _probs(), 0)


def test_wollack_omega_index_vector_validation():
    with pytest.raises(ValueError):
        wollack_omega(np.zeros((2, 2)), _source(), _probs(), 3)  # not 1-D
    with pytest.raises(ValueError):
        wollack_omega(np.array([]), np.array([]), _probs(), 3)  # empty
    with pytest.raises(ValueError):
        wollack_omega(np.full(5, 1j), _source(), _probs(), 3)  # complex
    with pytest.raises(ValueError):
        wollack_omega(np.array(["a"] * 5), _source(), _probs(), 3)  # non-numeric
    with pytest.raises(ValueError):
        wollack_omega(np.array([0, 1, np.inf, 0, 1]), _source(), _probs(), 3)
    with pytest.raises(ValueError):
        wollack_omega(np.array([0.5, 1, 2, 0, 1]), _source(), _probs(), 3)
    with pytest.raises(ValueError):
        wollack_omega(np.array([0, 1, 9, 0, 1]), _source(), _probs(), 3)


def test_wollack_omega_rejects_length_mismatch():
    with pytest.raises(ValueError):
        wollack_omega(np.array([0, 1, 2]), np.array([0, 1, 2, 0]), np.full((3, 3), 1 / 3), 3)


def test_wollack_omega_rejects_bad_probs():
    with pytest.raises(ValueError):
        wollack_omega(_copier(), _source(), np.full((5, 3), 1j), 3)  # complex
    with pytest.raises(ValueError):
        wollack_omega(_copier(), _source(), np.array([["a"] * 3] * 5), 3)  # non-numeric
    with pytest.raises(ValueError):
        wollack_omega(_copier(), _source(), np.full((5, 2), 0.5), 3)  # wrong 2-D shape
    with pytest.raises(ValueError):
        wollack_omega(_copier(), _source(), np.full(4, 0.5), 3)  # wrong 1-D size
    with pytest.raises(ValueError):
        wollack_omega(_copier(), _source(), np.full((5, 3, 1), 0.5), 3)  # 3-D


def test_wollack_omega_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            wollack_omega(_copier(), _source(), _probs(), 3)


# --------------------------- k_index ---------------------------

def test_k_index_happy_path():
    res = k_index(_responses(), 0, 1)
    assert isinstance(res, KIndexResult)
    assert res.ws > 0


def test_k_index_rejects_bad_indices():
    y = _responses()
    with pytest.raises(ValueError):
        k_index(y, 0.5, 1)
    with pytest.raises(ValueError):
        k_index(y, True, 1)
    with pytest.raises(ValueError):
        k_index(y, -1, 1)


def test_k_index_rejects_bad_responses():
    with pytest.raises(ValueError):
        k_index(np.zeros(5), 0, 1)  # not 2-D
    with pytest.raises(ValueError):
        k_index(np.zeros((1, 5)), 0, 1)  # < 2 persons
    with pytest.raises(ValueError):
        k_index(np.zeros((5, 0)), 0, 1)  # < 1 item
    with pytest.raises(ValueError):
        k_index(np.zeros((5, 5), dtype=complex), 0, 1)  # complex
    with pytest.raises(ValueError):
        k_index(np.array([["a", "b"], ["c", "d"]]), 0, 1)  # non-numeric
    y = _responses()
    y[0, 0] = 2.0
    with pytest.raises(ValueError):
        k_index(y, 0, 1)  # not 0/1


def test_k_index_rejects_index_relations():
    y = _responses()
    with pytest.raises(ValueError):
        k_index(y, 999, 1)  # out of range
    with pytest.raises(ValueError):
        k_index(y, 3, 3)  # not distinct


def test_k_index_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            k_index(_responses(), 0, 1)


# --------------------------- gbt ---------------------------

def test_gbt_happy_path():
    res = gbt(np.array([1, 0, 1, 1, 0]), np.full(5, 0.5))
    assert isinstance(res, GbtResult)
    assert res.observed_matches == 3


def test_gbt_rejects_bad_arrays():
    with pytest.raises(ValueError):
        gbt(np.zeros((2, 2)), np.zeros((2, 2)))  # not 1-D
    with pytest.raises(ValueError):
        gbt(np.array([]), np.array([]))  # empty
    with pytest.raises(ValueError):
        gbt(np.full(3, 1j), np.full(3, 0.5))  # complex
    with pytest.raises(ValueError):
        gbt(np.array(["a", "b"]), np.array([0.5, 0.5]))  # non-numeric


def test_gbt_rejects_length_and_values():
    with pytest.raises(ValueError):
        gbt(np.array([1, 0, 1]), np.array([0.5, 0.5]))  # length mismatch
    with pytest.raises(ValueError):
        gbt(np.array([1, 2, 0]), np.array([0.5, 0.5, 0.5]))  # matches not 0/1
    with pytest.raises(ValueError):
        gbt(np.array([1, 0, 1]), np.array([0.5, 1.5, 0.5]))  # prob out of [0,1]


def test_gbt_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            gbt(np.array([1, 0]), np.array([0.5, 0.5]))


# --------------------------- k_variants ---------------------------

def test_k_variants_happy_path():
    res = k_variants(_responses(), 0, 1)
    assert isinstance(res, KVariantsResult)
    assert res.ws > 0


def test_k_variants_rejects_bad_indices():
    y = _responses()
    with pytest.raises(ValueError):
        k_variants(y, 0.5, 1)
    with pytest.raises(ValueError):
        k_variants(y, True, 1)
    with pytest.raises(ValueError):
        k_variants(y, -1, 1)


def test_k_variants_rejects_bad_responses():
    with pytest.raises(ValueError):
        k_variants(np.zeros(5), 0, 1)
    with pytest.raises(ValueError):
        k_variants(np.zeros((1, 5)), 0, 1)
    with pytest.raises(ValueError):
        k_variants(np.zeros((5, 0)), 0, 1)
    with pytest.raises(ValueError):
        k_variants(np.zeros((5, 5), dtype=complex), 0, 1)
    with pytest.raises(ValueError):
        k_variants(np.array([["a", "b"], ["c", "d"]]), 0, 1)
    y = _responses()
    y[0, 0] = 2.0
    with pytest.raises(ValueError):
        k_variants(y, 0, 1)


def test_k_variants_rejects_index_relations():
    y = _responses()
    with pytest.raises(ValueError):
        k_variants(y, 999, 1)
    with pytest.raises(ValueError):
        k_variants(y, 3, 3)


def test_k_variants_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            k_variants(_responses(), 0, 1)
