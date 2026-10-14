"""Python binding contract for the single-group polytomous bifactor GRM
(stage 1 of #1912).

Covers the public ``fast_mlsirm.bifactor_grm.fit_bifactor_grm`` surface:
shapes and identification properties of a small independent simulation,
deterministic reruns from the same seed, argument validation, the
unobserved-category failure, and the non-convergence report.

Reference
---------
Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
Full-information item bifactor analysis of graded response data. *Applied
Psychological Measurement, 31*(1), 4-19.
https://doi.org/10.1177/0146621606289485
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.bifactor_grm import fit_bifactor_grm

N_PERSONS = 300
N_ITEMS = 6
N_SPECIFIC = 2
N_CAT = 4
SEED = 20260916
SPECIFIC_MAP = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
TRUE_A_G = np.array([1.4, -1.1, 1.0, 1.2, 0.9, 1.1])
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


def _simulate(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta_g = rng.normal(0.0, 1.0, N_PERSONS)
    theta_s = rng.normal(0.0, 1.0, (N_PERSONS, N_SPECIFIC))
    y = np.zeros((N_PERSONS, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        base = TRUE_A_G[i] * theta_g + TRUE_A_S[i] * theta_s[:, SPECIFIC_MAP[i]]
        cum = 1.0 / (1.0 + np.exp(-(base[:, None] + TRUE_D[i][None, :])))
        probs = np.concatenate(
            [1.0 - cum[:, [0]], -np.diff(cum, axis=1), cum[:, [-1]]], axis=1
        )
        draws = rng.random(N_PERSONS)
        y[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    return y


def _fit(y: np.ndarray, **overrides):
    kwargs = {
        "q_general": 7,
        "q_specific": 7,
        "max_iter": 500,
        "tol": 1e-5,
        "n_starts": 1,
        "seed": SEED,
    }
    kwargs.update(overrides)
    return fit_bifactor_grm(y, SPECIFIC_MAP, N_CAT, N_SPECIFIC, **kwargs)


def test_fit_returns_identified_shaped_result() -> None:
    y = _simulate(SEED)
    fit = _fit(y)
    assert fit.converged, fit.termination_reason
    assert fit.a_general.shape == (N_ITEMS,)
    assert fit.a_specific.shape == (N_ITEMS,)
    assert fit.threshold.shape == (N_ITEMS, N_CAT - 1)
    assert fit.theta_g_eap.shape == (N_PERSONS,)
    assert fit.theta_g_sd.shape == (N_PERSONS,)
    assert bool((fit.theta_g_sd >= 0).all())
    assert fit.category_counts.shape == (N_ITEMS, N_CAT)
    assert int(fit.category_counts.sum()) == N_PERSONS * N_ITEMS
    # Strictly decreasing boundary intercepts within each item.
    assert bool((np.diff(fit.threshold, axis=1) < 0).all())
    # Reflection canonicalization: largest-magnitude slopes positive.
    anchor_g = int(np.argmax(np.abs(fit.a_general)))
    assert fit.a_general[anchor_g] > 0.0
    for s in range(N_SPECIFIC):
        block = np.flatnonzero(SPECIFIC_MAP == s)
        anchor = block[int(np.argmax(np.abs(fit.a_specific[block])))]
        assert fit.a_specific[anchor] > 0.0
    # Monotone loglik trace ending at a finite value.
    trace = np.asarray(fit.loglik_trace)
    assert np.all(np.isfinite(trace)) and len(trace) >= 2
    assert bool((np.diff(trace) >= -1e-9).all())
    assert fit.n_parameters == N_ITEMS * N_CAT + N_ITEMS  # K+1 per block item


def test_same_seed_bit_reproduces() -> None:
    y = _simulate(SEED)
    first = _fit(y)
    second = _fit(y)
    np.testing.assert_array_equal(first.a_general, second.a_general)
    np.testing.assert_array_equal(first.threshold, second.threshold)
    assert first.best_start == second.best_start


def test_rejects_out_of_range_caller_arguments() -> None:
    y = _simulate(SEED)
    # #1929: no node-count cap; q_general=5/q_specific=22 are now accepted,
    # only < 1 is not.
    with pytest.raises(ValueError):
        _fit(y, q_general=0)
    with pytest.raises(ValueError):
        _fit(y, q_specific=0)
    with pytest.raises(ValueError):
        _fit(y, max_iter=0)
    with pytest.raises(ValueError):
        _fit(y, tol=0.0)
    with pytest.raises(ValueError):
        _fit(y, n_starts=0)
    with pytest.raises(ValueError):
        fit_bifactor_grm(
            y,
            np.array([0, 0, 0, 1, 1]),
            N_CAT,
            N_SPECIFIC,
            q_general=7,
            q_specific=7,
            max_iter=500,
            tol=1e-5,
            n_starts=1,
            seed=SEED,
        )
    with pytest.raises(ValueError):
        fit_bifactor_grm(
            y,
            np.array([0, 0, 0, 1, 1, 5], dtype=np.int64),
            N_CAT,
            N_SPECIFIC,
            q_general=7,
            q_specific=7,
            max_iter=500,
            tol=1e-5,
            n_starts=1,
            seed=SEED,
        )


def test_q_general_and_q_specific_are_required() -> None:
    """RED test for #1929: no unsourced defaults exist for the node counts."""
    y = _simulate(SEED)
    with pytest.raises(TypeError):
        fit_bifactor_grm(y, SPECIFIC_MAP, N_CAT, N_SPECIFIC)


def test_max_iter_n_starts_seed_and_tol_are_required() -> None:
    """ADR-0028 (#1963): iteration/convergence/replicate/seed controls have no default."""
    y = _simulate(SEED)
    base = dict(q_general=7, q_specific=7, max_iter=500, tol=1e-5, n_starts=1, seed=SEED)
    for missing in ("max_iter", "tol", "n_starts", "seed"):
        kwargs = {k: v for k, v in base.items() if k != missing}
        with pytest.raises(TypeError):
            fit_bifactor_grm(y, SPECIFIC_MAP, N_CAT, N_SPECIFIC, **kwargs)


def test_deprecated_positional_default_removed_for_fipc() -> None:
    """ADR-0028 (#1963): fit_bifactor_grm_fipc's max_iter/tol are also required."""
    from fast_mlsirm.bifactor_grm import fit_bifactor_grm_fipc

    y = _simulate(SEED)
    fit = _fit(y)
    anchor = np.zeros(N_ITEMS, dtype=bool)
    anchor[:3] = True
    with pytest.raises(TypeError):
        fit_bifactor_grm_fipc(
            y,
            SPECIFIC_MAP,
            N_CAT,
            N_SPECIFIC,
            anchor,
            fit.a_general,
            fit.a_specific,
            fit.threshold,
        )


def test_uncapped_start_budget_is_accepted() -> None:
    # Stage-1 review fix-up: no magic upper cap on n_starts.
    y = _simulate(SEED)
    fit = _fit(y, max_iter=1, tol=1e-12, n_starts=40)
    assert not fit.converged
    assert 0 <= fit.best_start < 40


def test_unobserved_category_fails_loudly() -> None:
    y = _simulate(SEED)
    y[:, 0] = np.clip(y[:, 0], 0, N_CAT - 2)  # drop the top category
    with pytest.raises(ValueError, match="never observed"):
        _fit(y)


def test_non_convergence_is_reported_not_substituted() -> None:
    y = _simulate(SEED)
    fit = _fit(y, max_iter=1, tol=1e-12)
    assert not fit.converged
    assert fit.termination_reason == "max_iter_reached"


def test_issue_1976_dense_q_never_tolerance_met_at_start_slopes() -> None:
    """#1976: q>=421 previously reported tolerance_met with a_general at 1.0."""
    smap = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1], dtype=np.int64)
    n, j, k = 340, 13, 4

    def _simulate(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        a_g = rng.uniform(0.8, 2.5, j)
        a_s = np.where(smap >= 0, rng.uniform(0.5, 1.5, j), 0.0)
        thr = np.sort(rng.normal(0, 1.2, (j, k - 1)), axis=1)[:, ::-1]
        tg = rng.standard_normal(n)
        ts = rng.standard_normal((n, 3))
        y = np.zeros((n, j), dtype=np.int64)
        for i in range(j):
            eta = a_g[i] * tg + (a_s[i] * ts[:, smap[i]] if smap[i] >= 0 else 0.0)
            cum = 1.0 / (1.0 + np.exp(-(eta[:, None] + thr[i][None, :])))
            p = np.column_stack(
                [
                    1.0 - cum[:, 0],
                    cum[:, 0] - cum[:, 1],
                    cum[:, 1] - cum[:, 2],
                    cum[:, 2],
                ]
            )
            y[:, i] = np.sum(rng.random(n)[:, None] > np.cumsum(p, axis=1), axis=1)
        return y

    y = None
    for seed in range(8, 40):
        cand = _simulate(seed)
        if all(len(np.unique(cand[:, i])) == k for i in range(j)):
            y = cand
            break
    assert y is not None, "no simulation seed with all categories observed"

    fit = fit_bifactor_grm(
        y,
        smap,
        n_cat=4,
        n_specific=3,
        q_general=481,
        q_specific=481,
        max_iter=2000,
        tol=1e-6,
        n_starts=1,
        seed=20260917,
    )
    if fit.termination_reason == "tolerance_met":
        assert fit.converged
        assert not np.allclose(fit.a_general, 1.0), (
            "tolerance_met must not leave a_general at the start"
        )
    elif np.allclose(fit.a_general, 1.0):
        assert not fit.converged
        assert fit.termination_reason == "numerical_em_stall"
