"""Coverage-B: core-absent and object-dtype coercion branches of classification.py."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm import classification


def _patch_core_none(monkeypatch):
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)


def test_core_or_raise_reports_missing_core(monkeypatch):
    _patch_core_none(monkeypatch)
    with pytest.raises(RuntimeError, match="livingston_correlation requires the compiled Rust core"):
        classification.livingston_correlation(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), 1.0, 1.0
        )


def test_subkoviak_rejects_non_numeric_cuts():
    with pytest.raises(ValueError, match="cuts must be numeric"):
        classification.subkoviak_agreement(
            np.array([1, 2, 3, 4, 5]), 10, np.array(["a"], dtype=object)
        )


def test_livingston_k2_coerces_object_dtype_inputs():
    result = classification.livingston_k2(
        np.array([0, 1, 2, 3, 4], dtype=object),
        cut=2.0,
        reliability=0.8,
        n_lengths=np.array([1.0], dtype=object),
    )
    assert isinstance(result, classification.LivingstonResult)
    assert result.k2.shape == (1,)


def test_livingston_correlation_coerces_object_dtype_inputs():
    value = classification.livingston_correlation(
        np.array([0, 1, 2, 3], dtype=object),
        np.array([1, 2, 3, 4], dtype=object),
        cut_x=2.0,
        cut_y=2.0,
    )
    assert isinstance(value, float)


def test_livingston_correlation_rejects_non_numeric_object_inputs():
    with pytest.raises(ValueError, match="x must be numeric"):
        classification.livingston_correlation(
            np.array(["a", "b"], dtype=object), np.array([1.0, 2.0]), 1.0, 1.0
        )
