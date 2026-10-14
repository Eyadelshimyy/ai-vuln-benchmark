"""Accuracy, uncertainty-calibration, and efficiency recovery for polytomous CAT.

A GRM bank is calibrated from synthetic responses and then passed to
``cat_simulate_polytomous`` for examinees with known true abilities. The test
requires bounded bias/MAE/RMSE, calibrated Rust-returned posterior uncertainty,
and materially fewer administered items than the full bank. Correlation remains
supplementary: a highly correlated CAT can still be biased or understate score
uncertainty, while a silent fallback to non-adaptive item order can still meet
score-recovery bounds.

Dodd, De Ayala, and Koch (1995) evaluate computerized adaptive testing with
polytomous IRT items and motivate measuring adaptive efficiency together with
score quality. This regression uses the package's Rust-owned item-information,
selection, scoring, posterior-SD, and stopping path; Python only generates the
synthetic GRM responses and summarizes recovery against known simulation truth.
The interval ``theta_eap ± 1.96 * theta_sd`` is an explicit normal approximation
based on the returned posterior mean and posterior SD, not a claim of an exact
posterior credible interval.

Reference
---------
Dodd, B. G., De Ayala, R. J., & Koch, W. R. (1995). Computerized adaptive
testing with polytomous items. *Applied Psychological Measurement, 19*(1),
5-22. https://doi.org/10.1177/014662169501900103
"""

from __future__ import annotations

import numpy as np
from fast_mlsirm import cat_simulate_polytomous, fit_polytomous

N_PERSONS = 400
N_ITEMS = 40
N_CAT = 4
SEED = 20260101

CAT_MIN_ITEMS = 5
CAT_MAX_ITEMS = N_ITEMS
CAT_SE_THRESHOLD = 0.4

# A real run with these exact parameters/seed measures theta RMSE ~0.40 and
# correlation ~0.91 using a mean of ~8.7 of 40 items -- comparable accuracy
# to the full-bank GRM recovery test (~0.38/~0.92) at roughly a fifth of
# the items. Margins are loose enough to tolerate a minor fast-mlsirm
# version bump while still catching a real regression in recovery,
# uncertainty calibration, or the adaptive-selection efficiency CAT exists
# to provide.
#
# MAX_MEAN_ITEMS_USED is deliberately close to the measured ~8.7 (not a
# loose N_ITEMS * 0.5): the same fixture/seed with adaptive=False (random
# item order) measures mean_items_used ~14.97, which still clears rmse/
# correlation bounds -- so a bound of 12 is what actually catches a silent
# fallback to non-adaptive selection.
MAX_THETA_RMSE = 0.65
MAX_THETA_MAE = 0.52
MAX_ABS_THETA_BIAS = 0.20
MIN_NORMAL_APPROX_COVERAGE = 0.80
MIN_THETA_CORRELATION = 0.7
MAX_MEAN_ITEMS_USED = 12


def _grm_category_probs(
    theta: float, discrimination: float, thresholds: np.ndarray
) -> np.ndarray:
    """Samejima graded-response category probabilities."""
    cumulative = np.concatenate(
        (
            [1.0],
            1.0 / (1.0 + np.exp(-discrimination * (theta - thresholds))),
            [0.0],
        )
    )
    return -np.diff(cumulative)


def test_cat_recovers_theta_using_substantially_fewer_items_than_the_full_bank() -> None:
    rng = np.random.default_rng(SEED)
    true_theta = rng.normal(0.0, 1.0, N_PERSONS)
    true_discrimination = rng.uniform(0.8, 2.0, N_ITEMS)
    true_thresholds = np.sort(
        rng.normal(0.0, 1.0, (N_ITEMS, N_CAT - 1)), axis=1
    )

    responses = np.zeros((N_PERSONS, N_ITEMS))
    for item in range(N_ITEMS):
        for person in range(N_PERSONS):
            probs = _grm_category_probs(
                true_theta[person], true_discrimination[item], true_thresholds[item]
            )
            probs = np.clip(probs, 0.0, None)
            probs = probs / probs.sum()
            responses[person, item] = rng.choice(N_CAT, p=probs)

    bank = fit_polytomous(responses, n_cat=N_CAT, model="grm", q_theta=21, max_iter=80, tol=1e-6)
    assert bank.converged

    cat_result = cat_simulate_polytomous(
        true_theta,
        bank,
        min_items=CAT_MIN_ITEMS,
        max_items=CAT_MAX_ITEMS,
        se_threshold=CAT_SE_THRESHOLD,
        adaptive=True,
        seed=SEED,
    )
    theta_eap = cat_result["theta_eap"]
    theta_sd = cat_result["theta_sd"]
    n_used = cat_result["n_used"]

    assert np.all(np.isfinite(theta_sd))
    assert np.all(theta_sd > 0.0)

    error = theta_eap - true_theta
    bias = float(np.mean(error))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    normal_approx_coverage = float(
        np.mean(np.abs(error) <= 1.96 * theta_sd)
    )
    correlation = float(np.corrcoef(theta_eap, true_theta)[0, 1])
    mean_items_used = float(n_used.mean())

    assert abs(bias) < MAX_ABS_THETA_BIAS, (
        f"CAT theta bias {bias:.3f} exceeded ±{MAX_ABS_THETA_BIAS}"
    )
    assert mae < MAX_THETA_MAE, (
        f"CAT theta MAE {mae:.3f} exceeded {MAX_THETA_MAE}"
    )
    assert rmse < MAX_THETA_RMSE, (
        f"CAT theta RMSE {rmse:.3f} exceeded {MAX_THETA_RMSE}"
    )
    assert normal_approx_coverage >= MIN_NORMAL_APPROX_COVERAGE, (
        "CAT theta normal-approximation coverage "
        f"{normal_approx_coverage:.3f} below {MIN_NORMAL_APPROX_COVERAGE}"
    )
    assert correlation > MIN_THETA_CORRELATION, (
        f"CAT theta correlation {correlation:.3f} below {MIN_THETA_CORRELATION}"
    )
    assert mean_items_used < MAX_MEAN_ITEMS_USED, (
        f"CAT used a mean of {mean_items_used:.2f} of {N_ITEMS} items, "
        "not meaningfully fewer than the full bank -- adaptive item selection "
        "isn't providing efficiency"
    )
