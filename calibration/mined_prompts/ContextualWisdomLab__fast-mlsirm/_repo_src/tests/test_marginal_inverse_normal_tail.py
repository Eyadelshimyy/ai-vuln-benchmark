"""Regression contracts for the marginal QMC inverse-normal transform."""

from __future__ import annotations

from statistics import NormalDist

import pytest

import fast_mlsirm.estimators.marginal as marginal


@pytest.mark.parametrize("probability", (1e-6, 1e-3, 1e-2))
def test_inverse_normal_tail_matches_reference_and_symmetry(probability: float) -> None:
    """Acklam's lower tail must retain its denominator and mirror the upper tail."""

    actual = marginal._inv_normal_cdf(probability)
    expected = NormalDist().inv_cdf(probability)

    assert actual == pytest.approx(expected, rel=0.0, abs=1e-8)
    assert actual == pytest.approx(
        -marginal._inv_normal_cdf(1.0 - probability),
        rel=0.0,
        abs=1e-8,
    )
