"""Python binding contract for the single-group polytomous two-tier GRM
(stage 4 of #1912).

Covers the public ``fast_mlsirm.two_tier_grm.fit_two_tier_grm`` surface:
shapes and identification properties of a small independent simulation
(two correlated primaries, cross-cutting specifics), deterministic reruns
from the same seed, argument validation, the unobserved-category failure,
and the non-convergence report.

Reference
---------
Cai, L. (2010). A two-tier full-information item factor analysis model with
applications. *Psychometrika, 75*(4), 581-612.
https://doi.org/10.1007/s11336-010-9178-0 (full text read, pp. 583-584)
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.two_tier_grm import fit_two_tier_grm, two_tier_oakes_se

N_PERSONS = 300
N_ITEMS = 6
N_PRIMARY = 2
N_SPECIFIC = 2
N_CAT = 4
SEED = 20260917
RHO = 0.35
PRIMARY_MAP = np.array(
    [
        [True, False],
        [True, False],
        [True, False],
        [False, True],
        [False, True],
        [False, True],
    ]
)
SPECIFIC_MAP = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
TRUE_A_P = np.array(
    [
        [1.4, 0.0],
        [1.1, 0.0],
        [1.0, 0.0],
        [0.0, 1.2],
        [0.0, 0.9],
        [0.0, 1.1],
    ]
)
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


def _simulate(seed: int, rho: float = RHO) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z0 = rng.normal(0.0, 1.0, N_PERSONS)
    z1 = rng.normal(0.0, 1.0, N_PERSONS)
    theta = np.stack([z0, rho * z0 + np.sqrt(1.0 - rho**2) * z1], axis=1)

    theta_s = rng.normal(0.0, 1.0, (N_PERSONS, N_SPECIFIC))
    y = np.zeros((N_PERSONS, N_ITEMS), dtype=np.int64)
    for i in range(N_ITEMS):
        base = (
            TRUE_A_P[i, 0] * theta[:, 0]
            + TRUE_A_P[i, 1] * theta[:, 1]
            + TRUE_A_S[i] * theta_s[:, SPECIFIC_MAP[i]]
        )
        cum = 1.0 / (1.0 + np.exp(-(base[:, None] + TRUE_D[i][None, :])))
        probs = np.concatenate(
            [1.0 - cum[:, [0]], -np.diff(cum, axis=1), cum[:, [-1]]], axis=1
        )
        draws = rng.random(N_PERSONS)
        y[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    return y

def test_primary_correlation_validation() -> None:
    with pytest.raises(ValueError, match="primary_correlation"):
        _fit(_simulate(SEED), primary_correlation="unknown")


def test_identity_rejects_identical_primary_support() -> None:
    y = _simulate(SEED)
    shared = np.ones_like(PRIMARY_MAP)
    with pytest.raises(ValueError, match="identical free-loading item sets"):
        fit_two_tier_grm(
            y, shared, SPECIFIC_MAP, N_CAT, N_PRIMARY, N_SPECIFIC,
            q_primary=7, q_specific=7, max_iter=500, tol=1e-5,
            n_starts=1, seed=SEED, primary_correlation="identity",
        )


# Cross-build tolerance for the stored origin/main literals below. The fixture
# stops on `delta_loglik <= 1e-2 * (1 + |loglik|)` (slack ~3.8e-1 at
# loglik = -37.15), so the pinned iterate is a trajectory point whose low-order
# bits differ between compilers; the observed cross-build spread on this fixture
# was at the 1e-9 scale, while injected behavioural regressions moved these
# values by 3.98e-4 to 1.05e-1. 1e-6 sits between those measured scales.
GOLDEN_CROSS_BUILD_ATOL = 1e-6
GOLDEN_N_ITER = 6
GOLDEN_TERMINATION = "tolerance_met"


def test_estimate_path_matches_origin_main_golden() -> None:
    """Golden from origin/main 99c228a8f50a in a separate clean s1 checkout.

    Built there with Python 3.12 and ``pip install -e .``; ran this test's
    deterministic 12 x 4 data through ``fit_two_tier_grm`` with the explicit
    arguments below, then ``two_tier_oakes_se`` at the fitted parameters.
    Cai (2010, pp. 583-584) defines the two-tier covariance; this checks that
    the existing estimated-Phi path stayed numerically unchanged.

    Tolerance contract. Two things are checked with different strictness:

    * Within one build, the default path and ``primary_correlation="estimate"``
      must agree bit-for-bit, and the iteration count, termination reason and
      trace length must match the golden exactly. Those are integer/string or
      same-arithmetic comparisons, so they carry no cross-platform slack.
    * The stored floating-point literals are compared with
      ``GOLDEN_CROSS_BUILD_ATOL``. They were produced by one build
      (linux-x86_64, pre-AVX host) and reproduce there exactly; a second build
      (macos-arm64) was reported to differ at the 1e-9 scale on this fixture,
      although that run's assertion text was not preserved. The fixture stops
      on the relative rule ``delta_loglik <= tol * (1 + |loglik|)``, which with
      ``tol=1e-2`` is a slack of about 3.8e-1 around ``loglik = -37.15``: the
      iterate this golden pins is a point on the EM trajectory, not a converged
      optimum, so ordinary floating-point reassociation between builds moves it
      far more than 1e-12. Any real change to the estimate path moves these
      values by far more than the tolerance below - deliberately injected
      regressions on this fixture (identity rewiring, one fewer Newton step, a
      10x larger ridge) moved them by 1.05e-1, 2.09e-2 and 3.98e-4 - so the
      guard keeps its purpose without asserting bitwise equality across
      compilers. Note the smallest of those is 3.98e-4: a 1e-2 tolerance would
      have missed it, which is why the bound below sits at 1e-6.

    Reference: Cai, L. (2010). A two-tier full-information item factor analysis
    model with applications. *Psychometrika, 75*(4), 581-612.
    https://doi.org/10.1007/s11336-010-9178-0
    """
    y = np.array([[(p + i) % 3 for i in range(4)] for p in range(12)], dtype=np.int64)
    pmap = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    smap = np.zeros(4, dtype=np.int64)
    kwargs = dict(q_primary=7, q_specific=7, max_iter=500, tol=1e-2, n_starts=1, seed=20260922)
    golden_phi = np.array([[1.0, -0.10480732118999127], [-0.10480732118999127, 1.0]])
    golden_trace = np.array([
        -57.878882056754186, -51.76952034453358, -47.89637529712579,
        -43.0974994038811, -38.606846492548215, -37.22539643782572,
        -37.15081748073924,
    ])
    golden_primary = np.array([
        [-0.14916623496724393, 0.0], [0.2869920793276147, 0.0],
        [0.0, 0.2869920842842997], [0.0, -0.14916624583907015],
    ])
    golden_specific = np.array([
        22.307488834567913, -0.8566330375435262,
        -0.8566330381941958, 22.307488835442,
    ])
    golden_threshold = np.array([
        [11.64599181577608, -11.591569652560967],
        [0.6385437446044752, -1.0542766103752086],
        [1.05427661081288, -0.6385437452077196],
        [11.5915696386629, -11.645991831810742],
    ])
    golden_labels = [
        "a_primary:0:0", "a_specific:0", "d:0:0", "d:0:1",
        "a_primary:1:0", "a_specific:1", "d:1:0", "d:1:1",
        "a_primary:2:1", "a_specific:2", "d:2:0", "d:2:1",
        "a_primary:3:1", "a_specific:3", "d:3:0", "d:3:1", "phi_z:0:1",
    ]
    fits = {}
    for correlation in (None, "estimate"):
        options = kwargs if correlation is None else {**kwargs, "primary_correlation": correlation}
        fit = fit_two_tier_grm(y, pmap, smap, 3, 2, 1, **options)
        fits[correlation] = fit
        # Exact, platform-independent parts of the contract.
        assert fit.n_iter == GOLDEN_N_ITER
        assert fit.termination_reason == GOLDEN_TERMINATION
        assert fit.loglik_trace.shape == golden_trace.shape
        np.testing.assert_array_equal(np.diag(fit.phi), np.ones(2))
        np.testing.assert_array_equal(fit.phi, fit.phi.T)
        # Stored literals: cross-build tolerance (see docstring).
        for got, expected in (
            (fit.phi, golden_phi), (fit.loglik_trace, golden_trace),
            (fit.a_primary, golden_primary), (fit.a_specific, golden_specific),
            (fit.threshold, golden_threshold),
        ):
            np.testing.assert_allclose(
                got, expected, rtol=0, atol=GOLDEN_CROSS_BUILD_ATOL
            )
        se = two_tier_oakes_se(
            fit.a_primary, fit.a_specific, fit.threshold, fit.phi,
            y, pmap, smap, 3, 2, 1, q_primary=7, q_specific=7, fd_step=1e-5,
            **({} if correlation is None else {"primary_correlation": correlation}),
        )
        assert se.labels == golden_labels
        assert se.information.shape == (17, 17)

    # Same build, same arithmetic: the default path and the explicit
    # "estimate" path must be bit-identical, with no tolerance at all.
    default_fit, explicit_fit = fits[None], fits["estimate"]
    for left, right in (
        (default_fit.phi, explicit_fit.phi),
        (default_fit.loglik_trace, explicit_fit.loglik_trace),
        (default_fit.a_primary, explicit_fit.a_primary),
        (default_fit.a_specific, explicit_fit.a_specific),
        (default_fit.threshold, explicit_fit.threshold),
    ):
        np.testing.assert_array_equal(left, right)


def _fit(y: np.ndarray, **overrides):
    kwargs = {
        "q_primary": 7,
        "q_specific": 7,
        "max_iter": 500,
        "tol": 1e-5,
        "n_starts": 1,
        "seed": SEED,
    }
    kwargs.update(overrides)
    return fit_two_tier_grm(y, PRIMARY_MAP, SPECIFIC_MAP, N_CAT, N_PRIMARY, N_SPECIFIC, **kwargs)


def test_fit_returns_identified_shaped_result() -> None:
    y = _simulate(SEED)
    fit = _fit(y)
    assert fit.converged, fit.termination_reason
    assert fit.a_primary.shape == (N_ITEMS, N_PRIMARY)
    assert fit.a_specific.shape == (N_ITEMS,)
    assert fit.threshold.shape == (N_ITEMS, N_CAT - 1)
    assert fit.phi.shape == (N_PRIMARY, N_PRIMARY)
    # Phi is a correlation matrix: unit diagonal, symmetric, |off-diag| < 1.
    np.testing.assert_allclose(np.diag(fit.phi), np.ones(N_PRIMARY), atol=1e-12)
    np.testing.assert_allclose(fit.phi, fit.phi.T, atol=1e-12)
    assert bool(np.all(np.abs(fit.phi - np.eye(N_PRIMARY)) < 1.0))
    assert fit.theta_p_eap.shape == (N_PERSONS, N_PRIMARY)
    assert fit.theta_p_sd.shape == (N_PERSONS, N_PRIMARY)
    assert bool((fit.theta_p_sd >= 0).all())
    assert fit.category_counts.shape == (N_ITEMS, N_CAT)
    assert int(fit.category_counts.sum()) == N_PERSONS * N_ITEMS
    # Fixed pattern positions stay exactly zero.
    assert bool((fit.a_primary[~PRIMARY_MAP] == 0.0).all())
    # Strictly decreasing boundary intercepts within each item.
    assert bool((np.diff(fit.threshold, axis=1) < 0).all())
    # Reflection canonicalization: largest-magnitude slopes positive.
    for d in range(N_PRIMARY):
        loaders = np.flatnonzero(PRIMARY_MAP[:, d])
        anchor = loaders[int(np.argmax(np.abs(fit.a_primary[loaders, d])))]
        assert fit.a_primary[anchor, d] > 0.0
    for s in range(N_SPECIFIC):
        block = np.flatnonzero(SPECIFIC_MAP == s)
        anchor = block[int(np.argmax(np.abs(fit.a_specific[block])))]
        assert fit.a_specific[anchor] > 0.0
    # Monotone loglik trace ending at a finite value.
    trace = np.asarray(fit.loglik_trace)
    assert np.all(np.isfinite(trace)) and len(trace) >= 2
    assert bool((np.diff(trace) >= -1e-9).all())
    # sum_i (k_i + has_specific + m1) + P(P-1)/2 = 6*1 + 6*1 + 6*3 + 1.
    assert fit.n_parameters == 6 + 6 + 18 + 1


def test_same_seed_bit_reproduces() -> None:
    y = _simulate(SEED)
    first = _fit(y)
    second = _fit(y)
    np.testing.assert_array_equal(first.a_primary, second.a_primary)
    np.testing.assert_array_equal(first.threshold, second.threshold)
    np.testing.assert_array_equal(first.phi, second.phi)
    assert first.best_start == second.best_start


def test_rejects_out_of_range_caller_arguments() -> None:
    y = _simulate(SEED)
    # #1929: no node-count cap; q_primary=5/q_specific=22 are now accepted,
    # only < 1 is not.
    with pytest.raises(ValueError):
        _fit(y, q_primary=0)
    with pytest.raises(ValueError):
        _fit(y, q_specific=0)
    with pytest.raises(ValueError):
        _fit(y, max_iter=0)
    with pytest.raises(ValueError):
        _fit(y, tol=0.0)
    with pytest.raises(ValueError):
        _fit(y, n_starts=0)
    with pytest.raises(ValueError, match="primary_map"):
        fit_two_tier_grm(
            y,
            np.ones((N_ITEMS, N_PRIMARY - 1), dtype=bool),
            SPECIFIC_MAP,
            N_CAT,
            N_PRIMARY,
            N_SPECIFIC,
            7,
            7, max_iter=500, tol=1e-6, n_starts=1, seed=0x9E3779B97F4A7C15
        )
    with pytest.raises(ValueError, match="specific_map"):
        fit_two_tier_grm(
            y,
            PRIMARY_MAP,
            np.array([0, 0, 0, 1, 1]),
            N_CAT,
            N_PRIMARY,
            N_SPECIFIC,
            7,
            7, max_iter=500, tol=1e-6, n_starts=1, seed=0x9E3779B97F4A7C15
        )


def test_q_primary_and_q_specific_are_required() -> None:
    """No unsourced defaults exist for the node counts (Project rule, #1929)."""
    y = _simulate(SEED)
    with pytest.raises(TypeError):
        fit_two_tier_grm(y, PRIMARY_MAP, SPECIFIC_MAP, N_CAT, N_PRIMARY, N_SPECIFIC, max_iter=500, tol=1e-6, n_starts=1, seed=0x9E3779B97F4A7C15)


def test_unobserved_category_fails_loudly() -> None:
    y = _simulate(SEED)
    y[:, 0] = np.clip(y[:, 0], 0, N_CAT - 2)  # drop the top category
    with pytest.raises(ValueError, match="never observed"):
        _fit(y)
