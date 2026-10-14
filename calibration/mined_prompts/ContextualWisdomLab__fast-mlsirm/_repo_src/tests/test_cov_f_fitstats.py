"""Residual coverage top-ups (batch F) for ``fast_mlsirm.fitstats``.

These tests reach the last few uncovered lines/branches that the existing
``test_fitstats.py`` / ``test_cov_c_fitstats.py`` and the paper-feature suite do
not: the continued-fraction exhaustion of the incomplete-gamma helper, the
non-finite elementary-symmetric denominator skip, the M2 degrees-of-freedom
guards for the CMLE/multigroup/multilevel estimators, the degenerate item-pair
SRMSR skips, and the CFI/TLI ``nan`` fall-through. They assert real numeric
behaviour and the exact exception types, not merely line execution.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm.fitstats as fm
from fast_mlsirm.types import MLSIRMParams


def _mirt_carrier(n_items, n_persons, seed=0, alpha=None, b=None):
    """Build a MIRT parameter carrier plus matching responses (no fitting)."""
    rng = np.random.default_rng(seed)
    alpha = np.zeros(n_items) if alpha is None else np.asarray(alpha, float)
    b = np.linspace(-1.0, 1.0, n_items) if b is None else np.asarray(b, float)
    theta = rng.standard_normal((n_persons, 1))
    eta = np.exp(alpha)[None, :] * theta + b[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    params = SimpleNamespace(
        alpha=alpha, b=b, zeta=np.zeros((n_items, 1)), tau=-30.0,
        theta=theta, xi=np.zeros((n_persons, 1)),
    )
    return y, np.zeros(n_items, dtype=np.int64), params


# ---------------------------------------------------------------------------
# incomplete-gamma continued fraction: loop exhaustion (191->205)
# ---------------------------------------------------------------------------


def test_gammainc_continued_fraction_exhausts_without_converging():
    # At a very large shape parameter sitting on the x == a + 1 boundary the
    # modified-Lentz continued fraction does not meet the 1e-15 convergence test
    # within 499 iterations, so the loop runs to completion and returns the
    # accumulated value (exercising the for->return exit arc).
    value = fm._gammainc_upper_reg(250000.0, 250001.0)
    assert math.isfinite(value)
    assert 0.0 < value < 1.0
    # chi2_sf routes through the same helper (a = df/2, x = stat/2).
    assert fm.chi2_sf(500002.0, 500000.0) == pytest.approx(value)


# ---------------------------------------------------------------------------
# elementary-symmetric denominator: non-finite skip (1973->1971)
# ---------------------------------------------------------------------------


def test_rasch_conditional_set_probabilities_skips_nonfinite_denominator():
    # A non-finite item easiness makes the centred log-weights non-finite, so
    # every elementary-symmetric denominator entry is non-finite and the
    # assignment guard (`... and isfinite(denominator[score])`) is False for each
    # score: those cells are left at their initial zero rather than assigned.
    with np.errstate(all="ignore"):
        out = fm._rasch_conditional_set_probabilities(
            np.array([np.inf, 0.0, 0.0, 0.0, 0.0]), [[0]]
        )
    assert out.shape == (6, 1)
    # no score cell was assigned, so the column stays all-zero.
    assert np.array_equal(out, np.zeros((6, 1)))


# ---------------------------------------------------------------------------
# dimensionality_residuals: predictor/response shape mismatch (1186)
# ---------------------------------------------------------------------------


def test_dimensionality_residuals_rejects_predictor_shape_mismatch():
    # theta carries more persons than the response matrix, so the model linear
    # predictor is taller than y and the shape guard fires.
    y = np.zeros((20, 4))
    fid = np.zeros(4, dtype=np.int64)
    params = MLSIRMParams(
        theta=np.zeros((25, 1)), alpha=np.zeros(4), b=np.zeros(4),
        xi=np.zeros((25, 1)), zeta=np.zeros((4, 1)), tau=-30.0,
    )
    with pytest.raises(ValueError, match="parameter dimensions must match"):
        fm.dimensionality_residuals(y, fid, params, "MIRT")


# ---------------------------------------------------------------------------
# CMLE Rasch M2: degrees-of-freedom guard and degenerate-pair SRMSR skip
# ---------------------------------------------------------------------------


def test_m2_cmle_rasch_df_guard_too_few_cases():
    # Exactly one respondent per raw score (0..5) satisfies the "every category
    # represented" precondition but leaves only 6 cases < p + 2 = 11.
    patterns = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    with pytest.raises(ValueError, match="needs more moments/cases than parameters"):
        fm.m2_cmle_rasch(patterns, np.linspace(-1.0, 1.0, 5))


def test_m2_cmle_rasch_skips_degenerate_item_pair():
    # A large well-fit sample whose two end items are each all-but-one constant
    # drives that pair's observed second-moment denominator below 1e-12, so the
    # SRMSR accumulator skips it (the dobs/dmod guard's False arc). The model
    # otherwise fits, so M2 stays finite and the run completes.
    n = 1_050_000
    rng = np.random.default_rng(1)
    b_true = np.array([-20.0, -0.5, 0.0, 0.5, 20.0])
    theta = rng.standard_normal(n)
    y = (
        rng.random((n, 5)) < 1.0 / (1.0 + np.exp(-(theta[:, None] + b_true[None, :])))
    ).astype(float)
    y[0, :] = 0.0  # guarantees a raw score of 0
    y[1, :] = 1.0  # guarantees a raw score of 5
    marginals = y.mean(axis=0)
    dobs_end_pair = (
        marginals[0] * (1 - marginals[0]) * marginals[4] * (1 - marginals[4])
    )
    assert dobs_end_pair <= 1e-12  # the pair that must be skipped
    result = fm.m2_cmle_rasch(y, b_true)
    assert math.isfinite(result.m2)
    assert math.isfinite(result.srmsr)


# ---------------------------------------------------------------------------
# _m2_numpy reference: degenerate pair skip (2329->2324) and CFI/TLI nan (2362)
# ---------------------------------------------------------------------------


def test_m2_numpy_constant_item_skips_pair_then_null_is_singular():
    # A fully constant observed item gives its pairs a zero observed
    # second-moment denominator (dobs == 0), so the SRMSR loop skips them. The
    # observed-independence null covariance is then singular, which is the
    # documented failure for a degenerate item.
    y, fid, params = _mirt_carrier(
        5, 300, seed=6, alpha=np.log(np.linspace(0.8, 1.4, 5))
    )
    y[:, 0] = 1.0
    with pytest.raises(np.linalg.LinAlgError):
        fm._m2_numpy(y, ~np.isnan(y), fid, params, "MIRT", 7, 7, 1e-8)


def test_m2_numpy_returns_nan_cfi_tli_when_null_below_df():
    # Independent Bernoulli responses make the independence null model fit about
    # as well as its own degrees of freedom (null_m2 <= null_df), so the CFI/TLI
    # incremental-fit indices fall through to nan.
    rng = np.random.default_rng(1001)
    n_items = 5
    y = (rng.random((60, n_items)) < 0.5).astype(float)
    assert not np.any(y.sum(0) == 0) and not np.any(y.sum(0) == 60)
    fid = np.zeros(n_items, dtype=np.int64)
    params = SimpleNamespace(
        alpha=np.zeros(n_items), b=np.zeros(n_items), zeta=np.zeros((n_items, 1)),
        tau=-30.0, theta=np.zeros((60, 1)), xi=np.zeros((60, 1)),
    )
    result = fm._m2_numpy(y, ~np.isnan(y), fid, params, "MIRT", 7, 7, 1e-8)
    assert result.null_m2 <= result.null_df
    assert math.isnan(result.cfi)
    assert math.isnan(result.tli)


# ---------------------------------------------------------------------------
# _m2_indices: CFI/TLI nan fall-through (2645)
# ---------------------------------------------------------------------------


def test_m2_indices_returns_nan_cfi_tli_when_null_not_worse():
    # When the null M2 is not larger than the model M2, the incremental indices
    # are undefined and returned as nan.
    _, _, ci_lower, ci_upper, cfi, tli = fm._m2_indices(
        m2_value=5.0, df=3.0, null_m2=2.0, null_df=4.0, n=100
    )
    assert math.isfinite(ci_lower) and math.isfinite(ci_upper)
    assert math.isnan(cfi)
    assert math.isnan(tli)


# ---------------------------------------------------------------------------
# multigroup / multilevel M2 degrees-of-freedom guards (2797, 2935)
# ---------------------------------------------------------------------------


def test_m2_multigroup_df_non_positive_guard():
    # Three one-item-per-dimension factors with two groups make the stacked
    # moment count equal the parameter count (2 * 6 == 12), so df <= 0.
    rng = np.random.default_rng(9)
    n_items, n_dims = 3, 3
    factor_id = np.array([0, 1, 2], dtype=np.int64)
    per_group = 40
    group_id = np.repeat(np.arange(2), per_group)
    y = (rng.random((2 * per_group, n_items)) < 0.5).astype(float)
    params = MLSIRMParams(
        theta=np.zeros((2 * per_group, n_dims)), alpha=np.zeros(n_items),
        b=np.zeros(n_items), xi=np.zeros((2 * per_group, 1)),
        zeta=np.zeros((n_items, 1)), tau=-30.0,
    )
    means = np.zeros((2, n_dims))
    sds = np.ones((2, n_dims))
    with pytest.raises(ValueError, match="multigroup M2 df non-positive"):
        fm.m2_multigroup(
            y, factor_id, params, "MIRT", group_id, means, sds, q_theta=7, q_xi=7
        )


def test_m2_multilevel_df_non_positive_guard():
    # Three items on one dimension give six moments but 2 * 3 + 1 = 7 parameters,
    # so the multilevel df is non-positive.
    rng = np.random.default_rng(11)
    n_items = 3
    factor_id = np.zeros(n_items, dtype=np.int64)
    n_clusters, cluster_size = 10, 8
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    n = n_clusters * cluster_size
    y = (rng.random((n, n_items)) < 0.5).astype(float)
    params = MLSIRMParams(
        theta=np.zeros((n, 1)), alpha=np.zeros(n_items), b=np.zeros(n_items),
        xi=np.zeros((n, 1)), zeta=np.zeros((n_items, 1)), tau=-30.0,
    )
    with pytest.raises(ValueError, match="multilevel M2 df non-positive"):
        fm.m2_multilevel(
            y, factor_id, params, "MIRT", cluster_id, 0.5, q_theta=7, q_u=7, q_xi=7
        )
