# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Joint person bootstrap replicate structure, SEs, and reproducibility."""

import numpy as np

from fast_mlsirm.bifactor_bootstrap import BifactorBootstrapResult, run_bifactor_bootstrap


def test_bifactor_bootstrap_shapes_and_se() -> None:
    """Run a 10-replicate person bootstrap and verify structure and SEs."""
    n_persons = 80
    n_items = 6
    n_cat = 3
    n_specific = 2

    smap = np.zeros(n_items, dtype=np.int64)
    smap[n_items // 2 :] = 1

    rng = np.random.default_rng(777)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2 :] = 1

    boot_res = run_bifactor_bootstrap(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        group_ids=group_ids,
        n_groups=2,
        n_replicates=10,
        batch_size=5,
        mc_stopping_ratio=0.1,
        compute_budget_seconds=1200.0,
        q_general=11,
        q_specific=11,
        n_jobs=2,
        base_seed=123,
        ci_level=0.95,
        max_iter=25,
        tol=1e-3,
        n_starts=1,
    )

    assert isinstance(boot_res, BifactorBootstrapResult)
    assert boot_res.n_requested == 10
    assert boot_res.n_replicates == 10
    assert boot_res.n_converged >= 8

    # Dimensions: B_conv x groups x items.
    assert boot_res.replicate_a_general.ndim == 3
    assert boot_res.replicate_a_general.shape[1:] == (2, 6)
    assert boot_res.replicate_threshold.shape[1:] == (2, 6, 2)
    assert boot_res.replicate_general_mean.shape[1:] == (2,)
    assert boot_res.replicate_specific_sd.shape[1:] == (2, 2)

    # Standard error shapes.
    assert boot_res.se_a_general.shape == (2, 6)
    assert boot_res.se_threshold.shape == (2, 6, 2)
    assert boot_res.se_general_mean.shape == (2,)
    assert boot_res.se_specific_sd.shape == (2, 2)

    # Reference group moments are pinned (zero SE across replicates).
    np.testing.assert_allclose(boot_res.se_general_mean[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(boot_res.se_general_sd[0], 0.0, atol=1e-10)

    # Free parameters must have finite non-negative SEs.
    assert np.all(np.isfinite(boot_res.se_a_general))
    assert np.all(boot_res.se_a_general >= 0.0)
    assert np.all(np.isfinite(boot_res.se_threshold))
    assert np.all(boot_res.se_threshold >= 0.0)
    assert boot_res.wall_clock_seconds > 0.0
    assert boot_res.throughput_replicates_per_second > 0.0


def test_bifactor_bootstrap_reproducibility() -> None:
    """Identical seeds must yield identical bootstrap replicate estimates."""
    n_persons = 40
    n_items = 4
    n_cat = 3
    n_specific = 1

    smap = np.zeros(n_items, dtype=np.int64)
    rng = np.random.default_rng(999)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)

    kwargs = dict(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        n_replicates=4,
        batch_size=4,
        mc_stopping_ratio=0.001,
        compute_budget_seconds=1200.0,
        q_general=11,
        q_specific=11,
        base_seed=42,
        ci_level=0.95,
        max_iter=8,
        tol=1e-3,
        n_starts=1,
    )

    res_run1 = run_bifactor_bootstrap(**kwargs, n_jobs=1)
    res_run2 = run_bifactor_bootstrap(**kwargs, n_jobs=2)

    assert res_run1.n_converged == res_run2.n_converged
    np.testing.assert_allclose(
        res_run1.replicate_a_general,
        res_run2.replicate_a_general,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        res_run1.replicate_threshold,
        res_run2.replicate_threshold,
        atol=1e-12,
    )


def test_base_seed_ci_level_max_iter_n_starts_tol_are_required() -> None:
    """ADR-0028 (#1963): stochastic seed and precision controls have no default."""
    n_persons, n_items, n_cat, n_specific = 40, 4, 3, 1
    smap = np.zeros(n_items, dtype=np.int64)
    rng = np.random.default_rng(999)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)

    base = dict(
        responses=responses,
        specific_map=smap,
        n_cat=n_cat,
        n_specific=n_specific,
        n_replicates=4,
        batch_size=4,
        mc_stopping_ratio=0.001,
        compute_budget_seconds=1200.0,
        q_general=11,
        q_specific=11,
        base_seed=42,
        ci_level=0.95,
        max_iter=8,
        tol=1e-3,
        n_starts=1,
    )
    for missing in ("base_seed", "ci_level", "max_iter", "n_starts", "tol"):
        kwargs = {k: v for k, v in base.items() if k != missing}
        try:
            run_bifactor_bootstrap(**kwargs)
        except TypeError:
            continue
        raise AssertionError(f"expected TypeError when {missing!r} is omitted")
