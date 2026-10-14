"""Regression coverage for lossless WLE scoring input admission."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fast_mlsirm.wle import score_wle, score_wle_poly


def _unexpected_dispatch(*args, **kwargs):
    """Fail if rejected caller evidence reaches Rust-owned WLE arithmetic."""
    raise AssertionError("invalid WLE evidence reached native scoring arithmetic")


def _core():
    """Return a fake native surface whose scoring entry points must not run."""
    return SimpleNamespace(score_wle=_unexpected_dispatch, score_wle_poly=_unexpected_dispatch)


@pytest.mark.parametrize(
    ("field", "a", "b", "responses", "c", "d", "message"),
    [
        (
            "responses",
            np.array([1.0]),
            np.array([0.0]),
            np.array([[1.0 + 1.0j]]),
            None,
            None,
            "responses must be real-valued",
        ),
        (
            "a",
            np.array([1.0 + 1.0j]),
            np.array([0.0]),
            np.array([[1.0]]),
            None,
            None,
            "a must be real-valued",
        ),
        (
            "b",
            np.array([1.0]),
            np.array([1.0 + 1.0j]),
            np.array([[1.0]]),
            None,
            None,
            "b must be real-valued",
        ),
        (
            "c",
            np.array([1.0]),
            np.array([0.0]),
            np.array([[1.0]]),
            np.array([0.1 + 0.1j]),
            None,
            "c must be real-valued",
        ),
        (
            "d",
            np.array([1.0]),
            np.array([0.0]),
            np.array([[1.0]]),
            None,
            np.array([0.9 + 0.1j]),
            "d must be real-valued",
        ),
    ],
)
def test_score_wle_rejects_complex_evidence_before_native_dispatch(
    monkeypatch, field, a, b, responses, c, d, message
):
    """Dichotomous WLE must never discard imaginary scoring evidence."""
    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: _core())

    with pytest.raises(ValueError, match=message):
        score_wle(a, b, responses, c=c, d=d)


@pytest.mark.parametrize(
    ("responses", "slope", "cat_params", "message"),
    [
        (
            np.array([[0.0 + 1.0j]]),
            np.array([1.0]),
            np.array([[0.5, -0.5]]),
            "responses must be real-valued",
        ),
        (
            np.array([[0.0]]),
            np.array([1.0 + 1.0j]),
            np.array([[0.5, -0.5]]),
            "slope must be real-valued",
        ),
        (
            np.array([[0.0]]),
            np.array([1.0]),
            np.array([[0.5 + 0.1j, -0.5]]),
            "cat_params must be real-valued",
        ),
    ],
)
def test_score_wle_poly_rejects_complex_evidence_before_native_dispatch(
    monkeypatch, responses, slope, cat_params, message
):
    """Polytomous WLE must never discard imaginary scoring evidence."""
    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: _core())

    with pytest.raises(ValueError, match=message):
        score_wle_poly(responses, slope, cat_params, n_cat=3)
