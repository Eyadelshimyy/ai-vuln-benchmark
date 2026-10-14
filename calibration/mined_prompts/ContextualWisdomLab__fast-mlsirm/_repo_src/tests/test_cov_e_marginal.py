"""Coverage: NumPy reference marginal estimator guards and edge branches.

Targets the validation raises, quadrature-node helper error paths, the frozen
``score_eap`` scorer, the softmax/GRM cell guards, and the NumPy EM branches
(latent_dim=3 init, structural-zero mixture, item covariate M-step) that the
Rust-backed public API and parity suites do not reach on the NumPy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.estimators.marginal import (
    _gh,
    _gpcm_m_step_item,
    _pca_align,
    _xi_grid,
    _xi_nodes,
    compute_category_logprobs,
    compute_gpcm_node_gradient,
    fit_gpcm_numpy,
    fit_marginal_numpy,
    compute_grm_category_logprobs,
    score_eap,
)


def _binary(n_persons, n_items, seed=0):
    """Deterministic 0/1 response matrix plus an all-observed mask."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n_persons, n_items)) < 0.5).astype(np.float64)
    observed = np.ones_like(y, dtype=bool)
    return y, observed


# --- quadrature-node helpers ---


def test_gh_rejects_zero_quadrature():
    # #1929: no node-count cap; q=9 is now accepted, only q < 1 is rejected.
    with pytest.raises(ValueError, match="q must be >= 1"):
        _gh(0)


def test_xi_grid_rejects_oversized_tensor():
    with pytest.raises(ValueError, match="tensor-grid limit"):
        _xi_grid(41, 4)


def test_xi_nodes_gh_rejects_high_latent_dim():
    with pytest.raises(ValueError, match="tensor Gauss-Hermite supports latent_dim <= 3"):
        _xi_nodes("gh", 4, 11, 0, 0)


def test_xi_nodes_qmc_rejects_zero_points():
    with pytest.raises(ValueError, match="xi_points must be >= 1"):
        _xi_nodes("qmc", 1, 11, 0, 0)


def test_xi_nodes_qmc_rejects_high_latent_dim():
    with pytest.raises(ValueError, match="Halton rule supports latent_dim"):
        _xi_nodes("qmc", 7, 11, 4, 0)


def test_xi_nodes_qmc_unshifted_low_tail_inverse_cdf():
    # xi_seed == 0 skips the Halton shift loop and enough points push a Halton
    # coordinate into the p < 0.02425 low tail of the inverse-normal CDF.
    grid, logw = _xi_nodes("qmc", 1, 11, 64, 0)
    assert grid.shape == (64, 1)
    assert np.all(np.isfinite(grid))
    assert np.allclose(logw, -np.log(64))


def test_xi_nodes_mc_rejects_zero_points():
    with pytest.raises(ValueError, match="xi_points must be >= 1"):
        _xi_nodes("mc", 1, 11, 0, 0)


def test_xi_nodes_rejects_unknown_rule():
    with pytest.raises(ValueError, match="xi_rule must be one of"):
        _xi_nodes("bogus", 1, 11, 4, 0)


# --- fit_marginal_numpy validation guards ---


def test_fit_marginal_rejects_oversized_working_set():
    y, observed = _binary(100, 2)
    with pytest.raises(ValueError, match="marginal working set"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1,
            latent_dim=1, q_theta=7, xi_rule="qmc", xi_points=100_000, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_seed=0
        )


def test_fit_marginal_rejects_unidim_model_with_multiple_dims():
    y, observed = _binary(6, 2)
    with pytest.raises(ValueError, match="unidimensional models require n_dims == 1"):
        fit_marginal_numpy(
            y, observed, np.array([0, 1]), model="ULS2PLM", n_dims=2, latent_dim=1, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_rejects_factor_id_out_of_range():
    y, observed = _binary(6, 2)
    with pytest.raises(ValueError, match="factor_id values must be in"):
        fit_marginal_numpy(
            y, observed, np.array([0, 1]), model="MLS2PLM", n_dims=1, latent_dim=1, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_rejects_bad_latent_dim():
    y, observed = _binary(6, 2)
    with pytest.raises(ValueError, match="1 <= latent_dim <= 3"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1, latent_dim=4, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_rejects_non_binary_observed():
    y = np.array([[0.0, 2.0], [1.0, 0.0]])
    observed = np.ones_like(y, dtype=bool)
    with pytest.raises(ValueError, match="observed responses must be 0 or 1"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1, latent_dim=1, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_singlefree_requires_anchors():
    y, observed = _binary(6, 2)
    with pytest.raises(ValueError, match="singlefree \\(FIPC\\) requires anchors"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1,
            latent_dim=1, pop={"kind": "singlefree"}, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_covariate_rejects_multilevel():
    y, observed = _binary(6, 2)
    with pytest.raises(ValueError, match="item covariates with a multilevel structure"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1, latent_dim=1,
            pop={"kind": "multilevel", "cluster_id": np.zeros(6, int), "n_clusters": 1},
            covariate={"w": np.zeros((1, 2))}, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_single_context_covariate_is_collinear():
    y, observed = _binary(6, 2)
    with pytest.raises(ValueError, match="collinear with b"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1, latent_dim=1,
            covariate={"w": np.zeros(2)}, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_rejects_bad_group_id():
    y, observed = _binary(4, 2)
    with pytest.raises(ValueError, match="group_id values must be in"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1, latent_dim=1,
            pop={"kind": "multigroup", "group_id": np.array([0, 0, 2, 0]), "n_groups": 2}, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_rejects_bad_cluster_id():
    y, observed = _binary(4, 2)
    with pytest.raises(ValueError, match="cluster_id values must be in"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1, latent_dim=1,
            pop={"kind": "multilevel", "cluster_id": np.array([0, 0, 3, 0]), "n_clusters": 2}, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_rejects_anchors_with_no_fixed_items():
    y, observed = _binary(6, 2)
    anchors = {
        "fixed": np.zeros(2, dtype=bool),
        "alpha": np.zeros(2),
        "b": np.zeros(2),
        "zeta": np.zeros((2, 1)),
    }
    with pytest.raises(ValueError, match="anchors must fix at least one item"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1,
            latent_dim=1, anchors=anchors, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_singlefree_requires_two_anchors_per_dim():
    y, observed = _binary(6, 2)
    anchors = {
        "fixed": np.array([True, False]),
        "alpha": np.zeros(2),
        "b": np.zeros(2),
        "zeta": np.zeros((2, 1)),
    }
    with pytest.raises(ValueError, match="at least two fixed anchor items per"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1,
            latent_dim=1, pop={"kind": "singlefree"}, anchors=anchors, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


def test_fit_marginal_rejects_nonfinite_anchor_tau():
    y, observed = _binary(6, 2)
    anchors = {
        "fixed": np.ones(2, dtype=bool),
        "alpha": np.zeros(2),
        "b": np.zeros(2),
        "zeta": np.zeros((2, 1)),
        "tau": np.nan,
    }
    with pytest.raises(ValueError, match="anchor tau must be finite"):
        fit_marginal_numpy(
            y, observed, np.array([0, 0]), model="MLS2PLM", n_dims=1,
            latent_dim=1, anchors=anchors, q_theta=21, q_xi=11, q_u=15, max_iter=200, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )


# --- NumPy EM branches ---


def test_fit_marginal_latent_dim_three_init_runs():
    y, observed = _binary(6, 3, seed=1)
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0, 0]), model="MLS2PLM", n_dims=1,
        latent_dim=3, q_theta=7, q_xi=7, max_iter=1, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert res["zeta"].shape == (3, 3)


def test_fit_marginal_anchors_without_tau_skip_branch():
    y, observed = _binary(6, 2, seed=2)
    anchors = {
        "fixed": np.ones(2, dtype=bool),
        "alpha": np.zeros(2),
        "b": np.zeros(2),
        "zeta": np.zeros((2, 1)),
    }
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0]), model="ULS2PLM", n_dims=1,
        latent_dim=1, q_theta=7, q_xi=7, max_iter=1, anchors=anchors, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    np.testing.assert_allclose(res["b"], anchors["b"])


def test_fit_marginal_zero_inflation_single_not_converged_final_pass():
    y, observed = _binary(10, 4, seed=3)
    y[:4] = 0.0  # structural-zero rows
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0, 0, 0]), model="MLSRM", n_dims=1,
        latent_dim=1, q_theta=7, q_xi=7, max_iter=1, zero_inflation=True, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert res["pi_zero"] > 0.0
    assert res["zero_responsibility"].shape == (10,)
    assert not res["converged"]


def test_fit_marginal_zero_inflation_multilevel_not_converged_final_pass():
    y, observed = _binary(12, 4, seed=4)
    cluster_id = np.repeat(np.arange(3), 4)
    y[8:12] = 0.0  # a fully structural-zero cluster
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0, 0, 0]), model="MLS2PLM", n_dims=1,
        latent_dim=1, q_theta=7, q_xi=7, q_u=7, max_iter=1, zero_inflation=True,
        pop={"kind": "multilevel", "cluster_id": cluster_id, "n_clusters": 3}, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert res["zero_responsibility"].shape == (12,)
    assert res["u_eap"].shape == (3,)
    assert not res["converged"]


def test_fit_marginal_covariate_distance_m_step_runs():
    y, observed = _binary(8, 4, seed=5)
    group_id = np.repeat(np.arange(2), 4)
    w = np.tile(np.linspace(-1.0, 1.0, 4), (2, 1))
    res = fit_marginal_numpy(
        y, observed, np.array([0, 1, 0, 1]), model="MLS2PLM", n_dims=2,
        latent_dim=2, q_theta=7, q_xi=7, max_iter=2,
        pop={"kind": "multigroup", "group_id": group_id, "n_groups": 2},
        covariate={"w": w, "init_delta": 0.0}, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert np.isfinite(res["delta"])


def test_fit_marginal_covariate_no_interaction_m_step_runs():
    y, observed = _binary(8, 4, seed=6)
    group_id = np.repeat(np.arange(2), 4)
    w = np.tile(np.linspace(-1.0, 1.0, 4), (2, 1))
    res = fit_marginal_numpy(
        y, observed, np.array([0, 1, 0, 1]), model="MIRT", n_dims=2,
        latent_dim=2, q_theta=7, q_xi=7, max_iter=2,
        pop={"kind": "multigroup", "group_id": group_id, "n_groups": 2},
        covariate={"w": w, "init_delta": 0.0}, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert np.isfinite(res["delta"])


# --- _pca_align 1-D sign flip ---


def test_pca_align_unidimensional_sign_flip():
    zeta = np.array([[-2.0], [0.5]])
    xi = np.array([[0.1], [0.2]])
    _pca_align(zeta, xi)
    # largest-|coordinate| item was negative, so both blocks flip sign
    assert zeta[0, 0] > 0.0
    assert xi[0, 0] < 0.0


# --- score_eap (frozen EAP scorer) ---


def test_score_eap_scores_response_vectors():
    y, observed = _binary(5, 4, seed=7)
    result = score_eap(
        y, observed, np.array([0, 0, 1, 1]),
        alpha=np.zeros(4), b=np.zeros(4), zeta=np.zeros((4, 2)), tau=0.0,
        model="MLS2PLM", q_theta=7, q_xi=7, eps_distance=1e-8
    )
    assert result["theta_eap"].shape == (5, 2)
    assert result["xi_eap"].shape == (5, 2)
    assert result["loglik"].shape == (5,)
    assert np.all(np.isfinite(result["theta_sd"]))


def test_score_eap_no_space_model_scores_vectors():
    y, observed = _binary(4, 3, seed=11)
    result = score_eap(
        y, observed, np.array([0, 0, 0]),
        alpha=np.zeros(3), b=np.zeros(3), zeta=np.zeros((3, 1)), tau=0.0,
        model="MIRT", n_dims=1, q_theta=7, q_xi=11, eps_distance=1e-8
    )
    assert result["theta_eap"].shape == (4, 1)
    assert result["xi_eap"].shape == (4, 1)


def test_score_eap_rejects_non_1d_factor_id():
    y, observed = _binary(3, 2)
    with pytest.raises(ValueError, match="factor_id must be a non-empty 1-D array"):
        score_eap(
            y, observed, np.zeros((2, 2)), alpha=np.zeros(2), b=np.zeros(2),
            zeta=np.zeros((2, 1)), tau=0.0, model="MIRT", q_theta=21, q_xi=11, eps_distance=1e-8
        )


def test_score_eap_rejects_non_numeric_factor_id():
    y, observed = _binary(3, 2)
    with pytest.raises(ValueError, match="factor_id must contain integer values"):
        score_eap(
            y, observed, np.array(["a", "b"], dtype=object), alpha=np.zeros(2),
            b=np.zeros(2), zeta=np.zeros((2, 1)), tau=0.0, model="MIRT", q_theta=21, q_xi=11, eps_distance=1e-8
        )


def test_score_eap_rejects_non_integer_factor_id():
    y, observed = _binary(3, 2)
    with pytest.raises(ValueError, match="finite non-negative integers"):
        score_eap(
            y, observed, np.array([0.5, 1.5]), alpha=np.zeros(2), b=np.zeros(2),
            zeta=np.zeros((2, 1)), tau=0.0, model="MIRT", q_theta=21, q_xi=11, eps_distance=1e-8
        )


# --- compute_category_logprobs / compute_grm_category_logprobs guards ---


def test_category_logprobs_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="1-D arrays of equal length K"):
        compute_category_logprobs(0.0, [0.0, 1.0], [0.0])


def test_category_logprobs_rejects_single_category():
    with pytest.raises(ValueError, match="need at least K=2 categories"):
        compute_category_logprobs(0.0, [0.0], [0.0])


def test_grm_category_logprobs_rejects_bad_thresholds():
    with pytest.raises(ValueError, match="1-D array of length K-1"):
        compute_grm_category_logprobs(np.array([0.0]), np.array([[1.0]]))


def test_deprecated_aliases_warn_and_match_new_names():
    from fast_mlsirm.estimators.marginal import (
        category_logprobs,
        gpcm_node_gradient,
        grm_category_logprobs,
    )

    with pytest.deprecated_call():
        old = category_logprobs(0.5, [0.0, 1.0], [0.0, 0.2])
    assert np.allclose(old, compute_category_logprobs(0.5, [0.0, 1.0], [0.0, 0.2]))

    with pytest.deprecated_call():
        old_g = gpcm_node_gradient(0.5, [0.0, 1.0], [0.0, 0.2], [3.0, 7.0])
    new_g = compute_gpcm_node_gradient(0.5, [0.0, 1.0], [0.0, 0.2], [3.0, 7.0])
    for a, b in zip(old_g, new_g):
        assert np.allclose(a, b)

    with pytest.deprecated_call():
        old_t = grm_category_logprobs(np.array([0.5]), np.array([1.0, -1.0]))
    assert np.allclose(old_t, compute_grm_category_logprobs(np.array([0.5]), np.array([1.0, -1.0])))


# --- fit_gpcm_numpy guards ---


def test_fit_gpcm_numpy_rejects_non_numeric_responses():
    with pytest.raises(ValueError, match="responses must be a numeric 2-D array"):
        fit_gpcm_numpy([["a", "b"], ["c", "d"]], 2, q_theta=21, max_iter=80, tol=1e-6)


def test_fit_gpcm_numpy_rejects_bad_q_theta():
    with pytest.raises(ValueError, match="q_theta must be an integer >= 1"):
        fit_gpcm_numpy(np.zeros((4, 2)), 2, q_theta=0, max_iter=80, tol=1e-6)


# --- additional NumPy EM edge branches ---


def test_fit_marginal_infers_n_dims_from_factor_id():
    y, observed = _binary(6, 3, seed=8)
    res = fit_marginal_numpy(
        y, observed, np.array([0, 1, 1]), model="MLS2PLM",
        latent_dim=1, q_theta=7, q_xi=7, max_iter=1, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert res["theta_eap"].shape == (6, 2)


def test_fit_marginal_multigroup_skips_empty_group_population_update():
    y, observed = _binary(6, 2, seed=9)
    # group id 1 is unpopulated (only 0 and 2 appear) so its population M-step
    # sees zero posterior mass and is skipped.
    res = fit_marginal_numpy(
        y, observed, np.array([0, 0]), model="ULS2PLM", n_dims=1, latent_dim=1,
        q_theta=7, q_xi=7, max_iter=2,
        pop={"kind": "multigroup", "group_id": np.array([0, 0, 0, 2, 2, 2]), "n_groups": 3}, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert res["mu"].shape == (3, 1)


def test_fit_marginal_zero_weight_covariate_skips_delta_update():
    y, observed = _binary(8, 4, seed=10)
    group_id = np.repeat(np.arange(2), 4)
    # all-zero covariate weights make the delta Fisher information exactly zero,
    # so the covariate M-step's ``if info_d > 0`` guard skips the update.
    res = fit_marginal_numpy(
        y, observed, np.array([0, 1, 0, 1]), model="MLS2PLM", n_dims=2,
        latent_dim=2, q_theta=7, q_xi=7, max_iter=2,
        pop={"kind": "multigroup", "group_id": group_id, "n_groups": 2},
        covariate={"w": np.zeros((2, 4)), "init_delta": 0.0}, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert res["delta"] == 0.0


def test_fit_marginal_mirt_item_reaches_optimum_within_m_steps():
    # A fully-converged MIRT item hits the ``slope < 1e-20`` inner-M-step break
    # once its penalized gradient vanishes; this seed/config reaches it.
    rng = np.random.default_rng(4)
    n_persons, n_items = 30, 6
    b = np.linspace(-1.0, 1.0, n_items)
    a = 0.8 + 0.6 * rng.random(n_items)
    theta = rng.standard_normal(n_persons)
    eta = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    observed = np.ones_like(y, dtype=bool)
    res = fit_marginal_numpy(
        y, observed, np.zeros(n_items, dtype=int), model="MIRT", n_dims=1,
        latent_dim=1, q_theta=7, q_xi=7, max_iter=60, m_steps=12, tol=1e-6, q_u=15, eps_distance=1e-8, xi_points=256, xi_seed=0
    )
    assert res["n_iter"] > 0


def test_gpcm_m_step_single_newton_iteration_completes():
    nodes, _ = _gh(7)
    rng = np.random.default_rng(0)
    r_counts = np.abs(rng.normal(size=(7, 3))) + 1.0
    out = _gpcm_m_step_item(np.zeros(3), nodes, r_counts, n_newton=1)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
