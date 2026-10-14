# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stopping rule, per-replicate reporting, validation, and CPU/GPU parity
for the joint person bootstrap.

Implementation basis: Andrews, D. W. K., & Buchinsky, M. (2000). A
three-step method for choosing the number of bootstrap repetitions.
*Econometrica, 68*(1), 23–51. https://www.jstor.org/stable/2999474
"""

import numpy as np
import pytest

from fast_mlsirm.bifactor_bootstrap import (
    _endpoint_movement,
    run_bifactor_bootstrap,
)


def _small_problem(seed=777, n_persons=80, n_items=6, n_cat=3, n_specific=2):
    smap = np.zeros(n_items, dtype=np.int64)
    smap[n_items // 2 :] = 1
    rng = np.random.default_rng(seed)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2 :] = 1
    return responses, smap, n_cat, n_specific, group_ids


def _fit_kw(**over):
    kw = dict(
        n_replicates=12,
        batch_size=4,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=1200.0,
        q_general=11,
        q_specific=11,
        n_jobs=1,
        base_seed=123,
        ci_level=0.95,
        max_iter=10,
        tol=1e-3,
        n_starts=1,
    )
    kw.update(over)
    return kw


def test_endpoint_movement_exact_values():
    """_endpoint_movement measures max endpoint shift over half-width."""
    prev_lo = np.array([0.0, 1.0, 5.0])
    prev_hi = np.array([2.0, 3.0, 5.0])  # third entry has zero half-width
    new_lo = np.array([0.1, 1.5, 5.0])
    new_hi = np.array([2.1, 3.0, 5.0])
    # entry 0: max(0.1, 0.1)/1.05; entry 1: max(0.5, 0.0)/0.75 = 2/3
    assert _endpoint_movement(prev_lo, prev_hi, new_lo, new_hi) == pytest.approx(2.0 / 3.0)
    # Identical endpoints -> zero movement.
    assert _endpoint_movement(new_lo, new_hi, new_lo, new_hi) == pytest.approx(0.0)
    # All-zero half-widths carry no MC uncertainty.
    z = np.array([1.0, 1.0])
    assert _endpoint_movement(z, z, z, z) == pytest.approx(0.0)


def test_ratio_zero_runs_all_replicates():
    """mc_stopping_ratio=0 disables early stopping (full completion)."""
    responses, smap, n_cat, n_specific, _ = _small_problem()
    res = run_bifactor_bootstrap(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        **_fit_kw(),
    )
    assert res.n_requested == 12
    assert res.n_replicates == 12
    assert res.stopped_early is False
    assert len(res.converged) == 12
    assert int(np.sum(res.converged)) == res.n_converged
    assert res.replicate_a_general.shape[0] == res.n_converged


def test_per_replicate_convergence_reported_and_never_substituted():
    """Failed replicates are flagged, excluded, never imputed."""
    responses, smap, n_cat, n_specific, group_ids = _small_problem()
    res = run_bifactor_bootstrap(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        group_ids=group_ids,
        n_groups=2,
        **_fit_kw(n_jobs=2),
    )
    assert res.converged.dtype == bool
    assert res.replicate_loglik.shape == (res.n_converged,)
    assert np.all(np.isfinite(res.replicate_loglik))
    assert res.replicate_a_general.shape[0] == res.n_converged
    assert res.replicate_threshold.shape[0] == res.n_converged
    if res.n_converged > 1:
        assert np.all(np.isfinite(res.se_a_general))
        assert np.all(np.isfinite(res.ci_lower_a_general))
        assert np.all(res.ci_lower_a_general <= res.ci_upper_a_general)


def test_stopping_rule_invariants():
    """Early stopping (if fired) preserves reporting invariants."""
    responses, smap, n_cat, n_specific, group_ids = _small_problem()
    res = run_bifactor_bootstrap(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        group_ids=group_ids,
        n_groups=2,
        **_fit_kw(n_replicates=40, batch_size=5, mc_stopping_ratio=0.5, n_jobs=2),
    )
    assert res.n_replicates <= res.n_requested
    assert len(res.converged) == res.n_replicates
    if res.stopped_early:
        assert res.n_replicates < res.n_requested


def test_bootstrap_argument_validation():
    """Caller arguments have validated ranges (no silent misconfiguration)."""
    responses, smap, n_cat, n_specific, group_ids = _small_problem()
    good = dict(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        group_ids=None,
        n_groups=1,
        n_replicates=4,
        batch_size=4,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=1200.0,
        q_general=11,
        q_specific=11,
        n_jobs=1,
        base_seed=1,
        ci_level=0.95,
        max_iter=3,
        tol=1e-3,
        n_starts=1,
    )
    bad_cases = [
        {"n_replicates": 0},
        {"n_replicates": -3},
        {"n_replicates": 4.5},
        {"n_replicates": True},
        {"batch_size": 0},
        {"batch_size": -1},
        {"mc_stopping_ratio": -0.1},
        {"mc_stopping_ratio": 1.0},
        {"mc_stopping_ratio": float("nan")},
        {"mc_stopping_ratio": float("inf")},
        {"compute_budget_seconds": 0.0},
        {"compute_budget_seconds": -2.0},
        {"compute_budget_seconds": float("nan")},
        {"device": "tpu"},
        {"ci_level": 0.0},
        {"ci_level": 1.0},
        {"ci_level": 1.5},
        {"q_general": 0},
        {"q_general": -3},
        {"q_general": 11.5},
        {"q_general": True},
        {"q_specific": 0},
        {"q_specific": -3},
        {"q_specific": 11.5},
        {"q_specific": True},
        {"n_starts": 0},
    ]
    for bad in bad_cases:
        kwargs = dict(good)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            run_bifactor_bootstrap(**kwargs)


def test_cpu_gpu_bootstrap_replicate_parity():
    """Same base_seed gives replicate-by-replicate CPU/GPU agreement."""
    responses, smap, n_cat, n_specific, _ = _small_problem(
        seed=999, n_persons=40, n_items=4
    )
    smap4 = np.array([0, 0, 1, 1], dtype=np.int64)
    kwargs = dict(
        responses=responses,
        specific_map=smap4,
        n_cat=n_cat,
        n_specific=2,
        n_replicates=6,
        batch_size=6,
        mc_stopping_ratio=0.0,
        compute_budget_seconds=1200.0,
        q_general=11,
        q_specific=11,
        n_jobs=1,
        base_seed=42,
        ci_level=0.95,
        max_iter=8,
        tol=1e-3,
        n_starts=1,
    )
    res_cpu = run_bifactor_bootstrap(**kwargs, device="cpu")
    res_gpu = run_bifactor_bootstrap(**kwargs, device="gpu")
    assert res_cpu.n_converged == res_gpu.n_converged
    assert res_cpu.n_replicates == res_gpu.n_replicates
    np.testing.assert_allclose(
        res_cpu.replicate_a_general, res_gpu.replicate_a_general, atol=1e-3
    )
    np.testing.assert_allclose(
        res_cpu.replicate_threshold, res_gpu.replicate_threshold, atol=1e-3
    )
    np.testing.assert_allclose(
        res_cpu.replicate_loglik, res_gpu.replicate_loglik, atol=1e-2
    )
