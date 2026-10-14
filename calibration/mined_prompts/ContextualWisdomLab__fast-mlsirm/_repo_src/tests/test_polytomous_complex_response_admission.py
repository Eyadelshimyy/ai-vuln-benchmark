"""Regression coverage for lossless polytomous response admission."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import polytomous


def _unexpected_core_discovery():
    """Fail if invalid complex response data reach compiled-core discovery."""
    raise AssertionError("complex responses reached native-core discovery")


def _valid_fit() -> polytomous.PolytomousFit:
    """Return the smallest valid two-category fit record for scoring admission."""
    return polytomous.PolytomousFit(
        model="grm",
        slope=np.array([1.0], dtype=np.float64),
        cat_params=np.array([[0.0]], dtype=np.float64),
        loglik=0.0,
        n_iter=1,
    )


def test_fit_polytomous_rejects_complex_responses_before_native_discovery(monkeypatch):
    """A non-zero imaginary category must not be narrowed into a valid real category."""
    monkeypatch.setattr(polytomous, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0 + 1.0j], [1.0 + 0.0j]])

    with pytest.raises(ValueError, match="responses must be real-valued"):
        polytomous.fit_polytomous(responses, n_cat=2, model="grm", q_theta=21, max_iter=80, tol=1e-6)


def test_score_polytomous_rejects_complex_responses_before_native_discovery(monkeypatch):
    """Scoring must reject complex categories before Rust EAP dispatch is considered."""
    monkeypatch.setattr(polytomous, "_core_module", _unexpected_core_discovery)
    responses = np.array([[0.0 + 1.0j], [1.0 + 0.0j]])

    with pytest.raises(ValueError, match="responses must be real-valued"):
        polytomous.score_polytomous(responses, _valid_fit(), q_theta=21)


def test_polytomous_response_admission_preserves_real_missingness_contract():
    """Real integer categories and both supported missing markers remain unchanged."""
    responses = np.array([[0.0, np.nan], [-1.0, 1.0]], dtype=np.float32)

    values, observed = polytomous._poly_int_and_mask(responses, n_cat=2)

    assert np.array_equal(values, np.array([[0, 0], [0, 1]], dtype=np.int64))
    assert np.array_equal(
        observed,
        np.array([[True, False], [False, True]], dtype=bool),
    )
