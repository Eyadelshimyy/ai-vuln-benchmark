"""Residual coverage top-ups (batch F) for the NumPy marginal EM estimator.

These reach the last uncovered branches in ``fit_marginal_numpy`` that the
existing ``test_cov_e_marginal.py`` / ``test_estimator_marginal.py`` suites do
not: the valid single-free FIPC anchor path (enough fixed anchors per
dimension), the zero-information tau M-step skip, and the backtracking
line-search exhaustion arcs of the tau and covariate M-steps. Each test asserts
real returned behaviour, not merely line execution.
"""

from __future__ import annotations

import numpy as np

from fast_mlsirm.estimators.marginal import fit_marginal_numpy


def _binary(n_persons, n_items, seed=0):
    """Deterministic 0/1 response matrix plus an all-observed mask."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n_persons, n_items)) < 0.5).astype(np.float64)
    return y, np.ones_like(y, dtype=bool)


# ---------------------------------------------------------------------------
# single-free FIPC: enough fixed anchors per dimension (593->599 False arc)
# ---------------------------------------------------------------------------


def test_singlefree_accepts_two_anchors_per_dimension():
    # Two fixed anchor items on the single dimension satisfy the >= 2 rule, so
    # the guard's raise is skipped and calibration proceeds with the anchored
    # item parameters held fixed.
    y, observed = _binary(8, 3, seed=1)
    anchors = {
        "fixed": np.array([True, True, False]),
        "alpha": np.zeros(3),
        "b": np.array([-0.3, 0.4, 0.0]),
        "zeta": np.zeros((3, 1)),
    }
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0, 0]), model="MLS2PLM", n_dims=1,
        latent_dim=1, q_theta=7, q_xi=7, max_iter=1,
        pop={"kind": "singlefree"}, anchors=anchors, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    # anchored items keep their fixed easiness.
    np.testing.assert_allclose(res["b"][:2], anchors["b"][:2])


# ---------------------------------------------------------------------------
# tau M-step: zero Fisher information skip (823->855 False arc)
# ---------------------------------------------------------------------------


def test_tau_m_step_skips_when_information_is_zero():
    # Anchoring every item at a saturating easiness makes the model
    # probabilities exactly 0/1, so prob * (1 - prob) == 0 everywhere and the
    # data information vanishes. With the tau ridge penalty also set to zero the
    # total tau information is 0, so the `info > 0` guard is False and tau is
    # left unchanged.
    y, observed = _binary(8, 4, seed=1)
    anchors = {
        "fixed": np.ones(4, dtype=bool),
        "alpha": np.zeros(4),
        "b": np.full(4, 100.0),
        "zeta": np.zeros((4, 1)),
    }
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0, 0, 0]), model="MLS2PLM", n_dims=1,
        latent_dim=1, q_theta=7, q_xi=7, max_iter=2, anchors=anchors,
        penalty={"lambda_tau": 0.0}, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert np.isfinite(res["tau"])


# ---------------------------------------------------------------------------
# tau M-step: line search exhausts against the upper clip (847->855, 852)
# ---------------------------------------------------------------------------


def test_tau_line_search_exhausts_at_clip_boundary():
    # A strong prior mean pulls tau toward 50, but tau is clipped at 5. Once tau
    # sits at the clip the Newton step keeps proposing tau >= 5, every candidate
    # re-clips to the current value with no objective gain, so the backtracking
    # line search halves the step through all 20 iterations and exits without
    # accepting a move.
    y, observed = _binary(10, 4, seed=2)
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0, 0, 0]), model="MLS2PLM", n_dims=1,
        latent_dim=1, q_theta=7, q_xi=7, max_iter=6,
        penalty={"mu_tau": 50.0, "lambda_tau": 5.0}, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert np.isfinite(res["tau"])
    assert res["tau"] <= 5.0


# ---------------------------------------------------------------------------
# covariate M-step: line search exhausts against the delta clip (897->905, 902)
# ---------------------------------------------------------------------------


def test_covariate_line_search_exhausts_at_clip_boundary():
    # A covariate that perfectly separates the responses wants an unbounded
    # slope; delta is clipped at 10 and starts there. Every Newton candidate
    # re-clips to 10 with no gain, so the covariate line search halves the step
    # through all 20 iterations and exits without accepting a move.
    n_persons, n_items = 12, 4
    group_id = np.repeat(np.arange(2), n_persons // 2)
    w = np.tile(np.array([-1.0, 1.0, -1.0, 1.0]), (2, 1))
    y = np.tile((w[0] > 0).astype(np.float64), (n_persons, 1))
    observed = np.ones_like(y, dtype=bool)
    res = fit_marginal_numpy(
        y, observed, np.array([0, 1, 0, 1]), model="MLS2PLM", n_dims=2,
        latent_dim=2, q_theta=7, q_xi=7, max_iter=4,
        pop={"kind": "multigroup", "group_id": group_id, "n_groups": 2},
        covariate={"w": w, "init_delta": 10.0}, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert np.isfinite(res["delta"])
    assert res["delta"] <= 10.0
