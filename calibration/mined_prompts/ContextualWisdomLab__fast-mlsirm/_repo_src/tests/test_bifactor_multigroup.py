"""Python binding contract for the multiple-group polytomous bifactor GRM
(stage 2 of #1912).

Covers the public ``fast_mlsirm.bifactor_multigroup`` surface: shapes and
identification properties of a small two-group simulation with a known focal
shift, deterministic reruns, single-group equivalence with stage 1,
anchor/free handling, argument validation, the unobserved-category failure,
and the non-convergence report.

References
----------
Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
Full-information item bifactor analysis of graded response data. *Applied
Psychological Measurement, 31*(1), 4-19.
https://doi.org/10.1177/0146621606289485

Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
https://doi.org/10.1037/a0023350
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.bifactor_grm import fit_bifactor_grm
from fast_mlsirm.bifactor_multigroup import fit_bifactor_grm_multigroup

N_PER_GROUP = 300
N_ITEMS = 6
N_SPECIFIC = 2
N_CAT = 4
SEED = 20260916
FOCAL_MU = 0.50
FOCAL_SIGMA = 1.20
SPECIFIC_MAP = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
TRUE_A_G = np.array([1.4, 1.1, 1.0, 1.2, 0.9, 1.1])
TRUE_A_S = np.array([1.0, 0.9, 1.1, 1.0, 0.8, 0.9])
TRUE_D = np.array(
    [
        [1.2, 0.0, -1.2],
        [1.0, -0.1, -1.3],
        [1.3, 0.2, -1.0],
        [1.1, 0.1, -1.1],
        [0.9, -0.2, -1.4],
        [1.2, 0.0, -1.2],
    ]
)


def _simulate(seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_persons = 2 * N_PER_GROUP
    rng = np.random.default_rng(seed)
    group = np.repeat(np.array([0, 1], dtype=np.int64), N_PER_GROUP)
    theta_g = np.empty(n_persons)
    theta_g[:N_PER_GROUP] = rng.normal(0.0, 1.0, N_PER_GROUP)
    theta_g[N_PER_GROUP:] = FOCAL_MU + FOCAL_SIGMA * rng.normal(
        0.0, 1.0, N_PER_GROUP
    )
    theta_s = rng.normal(0.0, 1.0, (n_persons, N_SPECIFIC))
    y = np.zeros((n_persons, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        base = TRUE_A_G[i] * theta_g + TRUE_A_S[i] * theta_s[:, SPECIFIC_MAP[i]]
        cum = 1.0 / (1.0 + np.exp(-(base[:, None] + TRUE_D[i][None, :])))
        probs = np.concatenate(
            [1.0 - cum[:, [0]], -np.diff(cum, axis=1), cum[:, [-1]]], axis=1
        )
        draws = rng.random(n_persons)
        y[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    return y, group


def _fit(y: np.ndarray, group: np.ndarray, **overrides):
    kwargs = {
        "q_general": 7,
        "q_specific": 7,
        "max_iter": 500,
        "tol": 1e-5,
        "n_starts": 1,
        "seed": SEED,
    }
    kwargs.update(overrides)
    return fit_bifactor_grm_multigroup(
        y, group, SPECIFIC_MAP, N_CAT, N_SPECIFIC, **kwargs
    )


def test_fit_returns_identified_shaped_result() -> None:
    y, group = _simulate(SEED)
    fit = _fit(y, group)
    assert fit.converged, fit.termination_reason
    assert fit.a_general.shape == (2, N_ITEMS)
    assert fit.a_specific.shape == (2, N_ITEMS)
    assert fit.threshold.shape == (2, N_ITEMS, N_CAT - 1)
    assert fit.theta_g_eap.shape == (2 * N_PER_GROUP,)
    assert fit.theta_g_sd.shape == (2 * N_PER_GROUP,)
    assert bool((fit.theta_g_sd >= 0).all())
    assert fit.group_category_counts.shape == (2, N_ITEMS, N_CAT)
    assert int(fit.group_category_counts.sum()) == 2 * N_PER_GROUP * N_ITEMS
    # Reference pinned; focal estimated.
    assert fit.general_mean[0] == 0.0
    assert fit.general_sd[0] == 1.0
    assert np.isfinite(fit.general_mean[1])
    assert np.isfinite(fit.general_sd[1]) and fit.general_sd[1] > 0
    # Anchored rows identical; boundaries strictly decreasing.
    np.testing.assert_array_equal(fit.a_general[0], fit.a_general[1])
    assert bool((np.diff(fit.threshold, axis=2) < 0).all())
    # Reflection canonicalization over the joint anchor.
    anchor = int(np.argmax(np.abs(fit.a_general[0])))
    assert fit.a_general[0][anchor] > 0.0
    # Monotone loglik trace.
    trace = np.asarray(fit.loglik_trace)
    assert np.all(np.isfinite(trace)) and len(trace) >= 2
    assert bool((np.diff(trace) >= -1e-9).all())
    # Focal EAPs center above reference under a positive shift.
    assert float(fit.theta_g_eap[N_PER_GROUP:].mean()) > float(
        fit.theta_g_eap[:N_PER_GROUP].mean()
    )


def test_same_seed_bit_reproduces() -> None:
    y, group = _simulate(SEED)
    first = _fit(y, group)
    second = _fit(y, group)
    np.testing.assert_array_equal(first.a_general, second.a_general)
    np.testing.assert_array_equal(first.threshold, second.threshold)
    assert first.best_start == second.best_start


def test_single_group_matches_stage1_exactly() -> None:
    rng = np.random.default_rng(SEED)
    n = 200
    theta_g = rng.normal(0.0, 1.0, n)
    theta_s = rng.normal(0.0, 1.0, (n, N_SPECIFIC))
    y = np.zeros((n, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        base = TRUE_A_G[i] * theta_g + TRUE_A_S[i] * theta_s[:, SPECIFIC_MAP[i]]
        cum = 1.0 / (1.0 + np.exp(-(base[:, None] + TRUE_D[i][None, :])))
        probs = np.concatenate(
            [1.0 - cum[:, [0]], -np.diff(cum, axis=1), cum[:, [-1]]], axis=1
        )
        draws = rng.random(n)
        y[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    group = np.zeros(n, dtype=np.int64)
    single = fit_bifactor_grm(
        y,
        SPECIFIC_MAP,
        N_CAT,
        N_SPECIFIC,
        q_general=7,
        q_specific=7,
        max_iter=200,
        tol=1e-5,
        n_starts=1,
        seed=SEED,
    )
    multi = fit_bifactor_grm_multigroup(
        y,
        group,
        SPECIFIC_MAP,
        N_CAT,
        N_SPECIFIC,
        q_general=7,
        q_specific=7,
        max_iter=200,
        tol=1e-5,
        n_starts=1,
        seed=SEED,
    )
    np.testing.assert_array_equal(multi.a_general[0], single.a_general)
    np.testing.assert_array_equal(multi.threshold[0], single.threshold)
    np.testing.assert_array_equal(multi.theta_g_eap, single.theta_g_eap)


def test_free_anchor_keeps_anchored_rows_equal() -> None:
    y, group = _simulate(SEED)
    anchor = np.array([True, True, True, True, True, False])
    fit = _fit(y, group, anchor_mask=anchor)
    assert fit.converged, fit.termination_reason
    for i in range(N_ITEMS):
        if anchor[i]:
            assert fit.a_general[0, i] == fit.a_general[1, i]
    assert np.isfinite(fit.a_general).all()


def test_rejects_out_of_range_caller_arguments() -> None:
    y, group = _simulate(SEED)
    # #1929: no node-count cap; q_general=5 is now accepted, only < 1 is not.
    with pytest.raises(ValueError):
        _fit(y, group, q_general=0)
    with pytest.raises(ValueError):
        _fit(y, group, n_starts=0)
    with pytest.raises(ValueError):
        _fit(y, group, anchor_mask=np.array([True, False]))
    with pytest.raises(ValueError):
        _fit(
            y,
            group,
            anchor_mask=np.array([False] * N_ITEMS),
        )


def test_uncapped_start_budget_is_accepted() -> None:
    # No magic upper cap on n_starts (stage-1 review fix-up rule).
    y, group = _simulate(SEED)
    fit = _fit(y, group, max_iter=1, tol=1e-12, n_starts=40)
    assert not fit.converged
    assert 0 <= fit.best_start < 40


def test_unobserved_category_fails_loudly() -> None:
    y, group = _simulate(SEED)
    y[:, 0] = np.clip(y[:, 0], 0, N_CAT - 2)
    with pytest.raises(ValueError, match="never observed"):
        _fit(y, group)


def test_non_convergence_is_reported_not_substituted() -> None:
    y, group = _simulate(SEED)
    fit = _fit(y, group, max_iter=1, tol=1e-12)
    assert not fit.converged
    assert fit.termination_reason == "max_iter_reached"


def test_q_general_and_q_specific_are_required() -> None:
    """RED test for #1929: no unsourced defaults exist for the node counts."""
    y, group = _simulate(SEED)
    with pytest.raises(TypeError):
        fit_bifactor_grm_multigroup(y, group, SPECIFIC_MAP, N_CAT, N_SPECIFIC)


def test_max_iter_n_starts_seed_and_tol_are_required() -> None:
    """ADR-0028 (#1963): iteration/convergence/replicate/seed controls have no default."""
    y, group = _simulate(SEED)
    base = dict(q_general=7, q_specific=7, max_iter=500, tol=1e-5, n_starts=1, seed=SEED)
    for missing in ("max_iter", "tol", "n_starts", "seed"):
        kwargs = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(TypeError):
            fit_bifactor_grm_multigroup(
                y, group, SPECIFIC_MAP, N_CAT, N_SPECIFIC, **kwargs
            )
