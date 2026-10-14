"""Fixed-item-parameter calibration recovery study over a two-period GRM design.

Period 1 free-calibrates an item bank. Period 2 contains different examinees
from the same item parameters but a deliberate ability-mean shift and is EAP
scored on the period-1 fit without re-estimating its item parameters. The study
checks both person recovery and preservation of the population shift, then
contrasts that fixed-bank result with a separately converged free refit that
re-centers the second period.

Kim (2006) describes fixed-parameter calibration methods based on marginal
maximum-likelihood/EM calibration and evaluates recovery when fixed reference
item parameters are carried into groups with shifted latent distributions. The
present fit-then-score test is a bounded public-API recovery analogue: it does
not claim to reimplement Kim's five EM variants in Python.

Reference
---------
Kim, S. (2006). A comparative study of IRT fixed parameter calibration
methods. *Journal of Educational Measurement, 43*(4), 355-381.
https://doi.org/10.1111/j.1745-3984.2006.00021.x
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import fit_polytomous, score_polytomous

N_ITEMS = 15
N_CAT = 4
N_PERSONS_PERIOD_1 = 400
N_PERSONS_PERIOD_2 = 250
SEED = 20260101
PERIOD_2_TRUE_MEAN_SHIFT = 0.3

# A real run with these exact parameters/seed measures period-2 FIPC theta
# RMSE ~0.33 and correlation ~0.94, with the FIPC-estimated period-2 mean
# landing at ~0.37 against a true realized mean of ~0.43 (the shift is
# substantially recovered, not perfectly -- sampling noise on 250 people).
# Bias/MAE and normal-approximation coverage based on the Rust-returned
# posterior SD are required independently of correlation so fixed-bank
# recovery cannot pass on rank ordering alone.
MAX_THETA_RMSE = 0.6
MAX_THETA_MAE = 0.5
MAX_ABS_THETA_BIAS = 0.20
MIN_THETA_NORMAL_APPROX_COVERAGE = 0.80
MIN_THETA_CORRELATION = 0.75
MIN_FIPC_MEAN_SHIFT_DETECTED = PERIOD_2_TRUE_MEAN_SHIFT * 0.5
MAX_INDEPENDENT_REFIT_MEAN_SHIFT_DETECTED = PERIOD_2_TRUE_MEAN_SHIFT * 0.35


def _grm_category_probs(
    theta: float, discrimination: float, thresholds: np.ndarray
) -> np.ndarray:
    """Samejima graded-response probabilities using threshold locations."""
    cumulative = np.concatenate(
        (
            [1.0],
            1.0 / (1.0 + np.exp(-discrimination * (theta - thresholds))),
            [0.0],
        )
    )
    return -np.diff(cumulative)


def _simulate_responses(
    theta: np.ndarray,
    discrimination: np.ndarray,
    thresholds: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_persons = len(theta)
    n_items = len(discrimination)
    responses = np.zeros((n_persons, n_items))
    for item in range(n_items):
        for person in range(n_persons):
            probs = _grm_category_probs(
                theta[person], discrimination[item], thresholds[item]
            )
            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum()
            responses[person, item] = rng.choice(N_CAT, p=probs)
    return responses


def test_fipc_recovers_period_two_mean_shift_that_an_independent_refit_would_hide() -> None:
    param_rng = np.random.default_rng(SEED)
    true_discrimination = param_rng.uniform(0.8, 2.0, N_ITEMS)
    true_thresholds = np.sort(
        param_rng.normal(0.0, 1.0, (N_ITEMS, N_CAT - 1)), axis=1
    )

    theta_period_1 = param_rng.normal(0.0, 1.0, N_PERSONS_PERIOD_1)
    theta_period_2 = param_rng.normal(
        PERIOD_2_TRUE_MEAN_SHIFT, 1.0, N_PERSONS_PERIOD_2
    )

    responses_period_1 = _simulate_responses(
        theta_period_1, true_discrimination, true_thresholds, SEED + 1
    )
    period_1_fit = fit_polytomous(
        responses_period_1, n_cat=N_CAT, model="grm", q_theta=21, max_iter=80, tol=1e-6
    )
    assert period_1_fit.converged

    responses_period_2 = _simulate_responses(
        theta_period_2, true_discrimination, true_thresholds, SEED + 2
    )

    # Fixed-bank scoring: period 2 uses period 1's item parameters unchanged.
    fipc_scored = score_polytomous(responses_period_2, period_1_fit, q_theta=21)
    fipc_theta_eap = fipc_scored["theta_eap"]
    fipc_theta_sd = fipc_scored["theta_sd"]

    error = fipc_theta_eap - theta_period_2
    bias = float(np.mean(error))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    normal_approx_coverage = float(
        np.mean(np.abs(error) <= 1.96 * fipc_theta_sd)
    )
    correlation = float(np.corrcoef(fipc_theta_eap, theta_period_2)[0, 1])
    fipc_detected_shift = float(fipc_theta_eap.mean())

    assert np.all(np.isfinite(fipc_theta_sd))
    assert np.all(fipc_theta_sd > 0.0)
    assert abs(bias) < MAX_ABS_THETA_BIAS, (
        f"FIPC period-2 theta absolute bias {abs(bias):.3f} exceeded "
        f"{MAX_ABS_THETA_BIAS}"
    )
    assert mae < MAX_THETA_MAE, (
        f"FIPC period-2 theta MAE {mae:.3f} exceeded {MAX_THETA_MAE}"
    )
    assert rmse < MAX_THETA_RMSE, (
        f"FIPC period-2 theta RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    )
    assert normal_approx_coverage >= MIN_THETA_NORMAL_APPROX_COVERAGE, (
        f"FIPC period-2 theta mean±1.96 posterior-SD coverage "
        f"{normal_approx_coverage:.3f} below {MIN_THETA_NORMAL_APPROX_COVERAGE}"
    )
    assert correlation > MIN_THETA_CORRELATION, (
        f"FIPC period-2 theta correlation {correlation:.3f} "
        f"below {MIN_THETA_CORRELATION}"
    )
    assert fipc_detected_shift > MIN_FIPC_MEAN_SHIFT_DETECTED, (
        f"FIPC-detected period-2 mean shift {fipc_detected_shift:.3f} did not clear "
        f"{MIN_FIPC_MEAN_SHIFT_DETECTED:.3f} -- fixed-bank scoring should "
        "preserve real between-period movement"
    )

    # The comparison is admissible only if the free refit itself converged.
    independent_fit = fit_polytomous(
        responses_period_2, n_cat=N_CAT, model="grm", q_theta=21, max_iter=80, tol=1e-6
    )
    assert independent_fit.converged
    independent_scored = score_polytomous(responses_period_2, independent_fit, q_theta=21)
    independent_detected_shift = float(independent_scored["theta_eap"].mean())

    assert abs(independent_detected_shift) < MAX_INDEPENDENT_REFIT_MEAN_SHIFT_DETECTED, (
        f"independent refit detected shift {independent_detected_shift:.3f}, expected it to "
        f"re-center near 0 (below {MAX_INDEPENDENT_REFIT_MEAN_SHIFT_DETECTED:.3f}) -- "
        "if it also recovers the shift, this test no longer demonstrates fixed-bank "
        "scoring's value over a naive independent refit"
    )
