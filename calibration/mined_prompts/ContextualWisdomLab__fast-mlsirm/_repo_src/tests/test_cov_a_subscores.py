"""Coverage for Haberman subscore added-value analysis (subscores.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.subscores import SubscoreResult, subscore_analysis


def _valid(seed=0, n=200):
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    cols = [f1 + rng.standard_normal(n) * 0.7 for _ in range(3)]
    cols += [f2 + rng.standard_normal(n) * 0.7 for _ in range(3)]
    return np.column_stack(cols)


def _groups():
    return np.array([0, 0, 0, 1, 1, 1])


def test_subscore_analysis_happy_path():
    res = subscore_analysis(_valid(), _groups())
    assert isinstance(res, SubscoreResult)
    assert res.alpha.shape == (2,)
    assert res.corr.shape == (3, 3)
    assert res.disattenuated_corr.shape == (2, 2)
    assert res.subscore_s.shape == (200, 2)


def test_subscore_analysis_accepts_float_integer_groups():
    res = subscore_analysis(_valid(), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    assert res.alpha.shape == (2,)


def test_subscore_analysis_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            subscore_analysis(_valid(), _groups())


def test_subscore_analysis_rejects_non_2d():
    with pytest.raises(ValueError):
        subscore_analysis(np.zeros(6), _groups())


def test_subscore_analysis_rejects_too_small():
    with pytest.raises(ValueError):
        subscore_analysis(np.zeros((2, 6)), _groups())
    with pytest.raises(ValueError):
        subscore_analysis(np.zeros((5, 3)), np.array([0, 0, 1]))


def test_subscore_analysis_rejects_missing_values():
    y = _valid()
    y[0, 0] = np.nan
    with pytest.raises(ValueError):
        subscore_analysis(y, _groups())


def test_subscore_analysis_rejects_group_length_mismatch():
    with pytest.raises(ValueError):
        subscore_analysis(_valid(), np.array([0, 0, 1]))


def test_subscore_analysis_rejects_non_integer_groups():
    with pytest.raises(ValueError):
        subscore_analysis(_valid(), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.5]))


def test_subscore_analysis_rejects_negative_groups():
    with pytest.raises(ValueError):
        subscore_analysis(_valid(), np.array([-1, 0, 0, 1, 1, 1]))


def test_subscore_analysis_rejects_group_index_out_of_range():
    with pytest.raises(ValueError):
        subscore_analysis(_valid(), np.array([0, 0, 0, 1, 1, 6]))
