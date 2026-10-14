"""Coverage for minres factor analysis, omega, glb, and MAP (factor.py)."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.factor as F


def _data(seed=0, n=200, p=5):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    return f @ (rng.standard_normal((1, p)) * 0.8) + rng.standard_normal((n, p)) * 0.5


def _corr(seed=0):
    return np.corrcoef(_data(seed).T)


def test_minres_fa_happy_and_guard():
    res = F.minres_fa(_corr(), 1)
    assert res.loadings.shape == (5, 1)
    assert res.uniquenesses.shape == (5,)
    with pytest.raises(ValueError):
        F.minres_fa(np.zeros((3, 2)), 1)
    with pytest.raises(ValueError):
        F.minres_fa(np.zeros(5), 1)


def test_minres_fa_from_data_happy_and_guard():
    res = F.minres_fa_from_data(_data(), 1)
    assert res.loadings.shape == (5, 1)
    with pytest.raises(ValueError):
        F.minres_fa_from_data(np.zeros(5), 1)


def test_omega_total_1f_happy_and_guard():
    res = F.omega_total_1f(_corr())
    assert 0.0 <= res.omega_total <= 1.0
    assert res.fa.loadings.shape == (5, 1)
    with pytest.raises(ValueError):
        F.omega_total_1f(np.zeros((3, 2)))


def test_omega_total_1f_from_data_happy_and_guard():
    res = F.omega_total_1f_from_data(_data())
    assert 0.0 <= res.omega_total <= 1.0
    with pytest.raises(ValueError):
        F.omega_total_1f_from_data(np.zeros(5))


def test_glb_fa_happy_and_guard():
    res = F.glb_fa(_corr())
    assert res.nf >= 1
    assert res.communalities.shape == (5,)
    with pytest.raises(ValueError):
        F.glb_fa(np.zeros((3, 2)))


def test_glb_fa_from_data_happy_and_guard():
    res = F.glb_fa_from_data(_data())
    assert res.nf >= 1
    with pytest.raises(ValueError):
        F.glb_fa_from_data(np.zeros(5))


def test_velicer_map_happy_default_and_explicit_and_guard():
    default = F.velicer_map(_corr())
    explicit = F.velicer_map(_corr(), max_m=3)
    assert default.f2.shape[0] >= 1
    assert explicit.f2.shape[0] >= 1
    with pytest.raises(ValueError):
        F.velicer_map(np.zeros((3, 2)))


def test_velicer_map_from_data_happy_default_and_explicit_and_guard():
    default = F.velicer_map_from_data(_data())
    explicit = F.velicer_map_from_data(_data(), max_m=3)
    assert default.retained_f2 >= 0
    assert explicit.retained_f2 >= 0
    with pytest.raises(ValueError):
        F.velicer_map_from_data(np.zeros(5))
