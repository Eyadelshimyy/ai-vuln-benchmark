"""Regression tests for zero-score and full-score (extreme raw score) robustness.

A respondent whose response vector is all-correct (full raw score) or all-incorrect
(zero raw score) drives the unpenalised joint-maximum-likelihood ability estimate to
+/-infinity, because the likelihood is monotone in that person's latent trait (Lord,
1980, *Applications of Item Response Theory to Practical Testing Problems*; Baker &
Kim, 2004, *Item Response Theory: Parameter Estimation Techniques*, 2nd ed.). The
MAP/penalised objective used here must keep every estimate finite and preserve the
raw-score ordering of the extreme persons. ``tests/test_irt_stability.py`` exercises
extreme scores only for a single optimiser step and mixed with missing-by-design
axes; these tests pin the fully observed, fit-to-convergence behaviour.
"""

import numpy as np

from fast_mlsirm import FitConfig, PenaltyConfig
from fast_mlsirm.fit import fit


def _extreme_score_responses() -> np.ndarray:
    """Return a fully observed matrix whose first two persons are the extremes.

    Person 0 answers every item correctly (full raw score) and person 1 answers
    every item incorrectly (zero raw score); the remaining persons span the
    intermediate score range so the item parameters remain identified.
    """
    return np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # full raw score
            [0.0, 0.0, 0.0, 0.0],  # zero raw score
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )


def test_fit_converges_with_full_and_zero_score_persons():
    """Penalised fit stays finite when persons hold extreme raw scores."""
    responses = _extreme_score_responses()
    factors = np.zeros(responses.shape[1], dtype=int)

    result = fit(
        responses,
        factors,
        config=FitConfig(
            model="MIRT",
            optimizer="adam",
            max_iter=200,
            n_restarts=1,
            latent_dim=1,
            seed=11,
            penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
        ),
    )

    assert np.isfinite(result.objective)
    assert np.all(np.isfinite(result.params.theta))
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.alpha))
    # The regularised estimate must still order the extreme persons correctly:
    # the full-score respondent's trait exceeds the zero-score respondent's.
    assert result.params.theta[0, 0] > result.params.theta[1, 0]


def test_fit_is_finite_when_an_item_has_zero_response_variance():
    """A degenerate item answered identically by everyone must not break the fit."""
    responses = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )  # column 0 is constant (every person correct) -> zero variance
    factors = np.zeros(responses.shape[1], dtype=int)

    result = fit(
        responses,
        factors,
        config=FitConfig(
            model="MIRT",
            optimizer="adam",
            max_iter=200,
            n_restarts=1,
            latent_dim=1,
            seed=7,
            penalty=PenaltyConfig(lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0),
        ),
    )

    assert np.isfinite(result.objective)
    assert np.all(np.isfinite(result.params.theta))
    assert np.all(np.isfinite(result.params.b))
    assert np.all(np.isfinite(result.params.alpha))
