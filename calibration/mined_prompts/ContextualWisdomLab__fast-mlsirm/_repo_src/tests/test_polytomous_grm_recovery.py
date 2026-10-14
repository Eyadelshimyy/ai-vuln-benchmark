"""Theta-recovery study for the Rust-backed unidimensional GRM estimator.

Responses are generated from Samejima's cumulative graded-response model and
then passed through the public ``fit_polytomous(..., model="grm")`` and
``score_polytomous`` surfaces. The test therefore checks true-person-parameter
recovery rather than only validation or execution.

This simulator uses the package's native GRM parameterization directly:
``P(Y >= k | theta) = logistic(a * theta + beta_k)`` with strictly decreasing
boundary intercepts ``beta``. This is the same form documented by
``PolytomousFit.cat_params`` and ``mlsirm_core::poly::grm_logprobs``.

Reference
---------
Samejima, F. (1969). Estimation of latent ability using a response pattern of
graded scores. *Psychometrika, 34*(S1), 1-97.
https://doi.org/10.1007/BF03372160
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import fit_polytomous, score_polytomous, validate_irt_response_matrix

N_PERSONS = 400
N_ITEMS = 12
N_CAT = 4
SEED = 20260101

# Reasonable bounds for a 12-item, 4-category GRM test at this sample size:
# a real run with these exact parameters/seed measures RMSE ~0.38 and
# correlation ~0.92. Bias/MAE and normal-approximation coverage based on the
# Rust-returned posterior SD are asserted separately so correlation remains
# supplementary rather than standing in for calibration.
MAX_THETA_RMSE = 0.6
MAX_THETA_MAE = 0.5
MAX_ABS_THETA_BIAS = 0.15
MIN_THETA_NORMAL_APPROX_COVERAGE = 0.80
MIN_THETA_CORRELATION = 0.75


def _grm_category_probs(
    theta: float,
    discrimination: float,
    boundary_intercepts: np.ndarray,
) -> np.ndarray:
    """Samejima GRM probabilities in the package-native ``a*theta + beta`` form."""
    cumulative = np.concatenate(
        (
            [1.0],
            1.0 / (1.0 + np.exp(-(discrimination * theta + boundary_intercepts))),
            [0.0],
        )
    )
    return -np.diff(cumulative)


def test_grm_recovers_true_theta_within_expected_rmse() -> None:
    rng = np.random.default_rng(SEED)
    true_theta = rng.normal(0.0, 1.0, N_PERSONS)
    true_discrimination = rng.uniform(0.8, 2.0, N_ITEMS)
    # Preserve the predecessor test's deterministic DGP while making the
    # library-native additive boundary intercept explicit. These beta values
    # are the negatives of that test's additive threshold terms; they are not
    # classical threshold locations b, for which beta = -a * b.
    true_boundary_intercepts = -np.sort(
        rng.normal(0.0, 1.0, (N_ITEMS, N_CAT - 1)), axis=1
    )

    responses = np.zeros((N_PERSONS, N_ITEMS))
    for item in range(N_ITEMS):
        for person in range(N_PERSONS):
            probs = _grm_category_probs(
                true_theta[person],
                true_discrimination[item],
                true_boundary_intercepts[item],
            )
            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum()
            responses[person, item] = rng.choice(N_CAT, p=probs)

    responses = validate_irt_response_matrix(
        responses, item_type="polytomous", n_categories=N_CAT
    )
    fit = fit_polytomous(responses, n_cat=N_CAT, model="grm", q_theta=21, max_iter=80, tol=1e-6)
    assert fit.converged

    scored = score_polytomous(responses, fit, q_theta=21)
    theta_eap = scored["theta_eap"]
    theta_sd = scored["theta_sd"]

    error = theta_eap - true_theta
    bias = float(np.mean(error))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    normal_approx_coverage = float(np.mean(np.abs(error) <= 1.96 * theta_sd))
    correlation = float(np.corrcoef(theta_eap, true_theta)[0, 1])

    assert np.all(np.isfinite(theta_sd))
    assert np.all(theta_sd > 0.0)
    assert abs(bias) < MAX_ABS_THETA_BIAS, (
        f"theta recovery absolute bias {abs(bias):.3f} exceeded {MAX_ABS_THETA_BIAS}"
    )
    assert mae < MAX_THETA_MAE, (
        f"theta recovery MAE {mae:.3f} exceeded {MAX_THETA_MAE}"
    )
    assert rmse < MAX_THETA_RMSE, (
        f"theta recovery RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    )
    assert normal_approx_coverage >= MIN_THETA_NORMAL_APPROX_COVERAGE, (
        f"theta mean±1.96 posterior-SD coverage {normal_approx_coverage:.3f} below "
        f"{MIN_THETA_NORMAL_APPROX_COVERAGE}"
    )
    assert correlation > MIN_THETA_CORRELATION, (
        f"theta recovery correlation {correlation:.3f} below {MIN_THETA_CORRELATION}"
    )
