"""Public-API recovery tests for the marginal (MMLE-EM) latent-space estimator."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.estimators.marginal import fit_marginal_numpy
from fast_mlsirm.fit import fit


def _simulate_lsirm(
    seed=0,
    n_persons=500,
    n_items=14,
    n_dims=2,
    latent_dim=2,
    gamma=1.0,
    group_shift=None,
    cluster_sd=0.0,
    n_clusters=0,
):
    rng = np.random.default_rng(seed)
    fid = np.array([i % n_dims for i in range(n_items)])
    a = 0.8 + 0.8 * rng.random(n_items)
    b = -1.0 + 2.0 * rng.random(n_items)
    zeta = rng.standard_normal((n_items, latent_dim)) * 0.8
    group_id = (np.arange(n_persons) % 2) if group_shift is not None else None
    cluster_id = (np.arange(n_persons) % n_clusters) if n_clusters else None
    u = rng.standard_normal(n_clusters) * cluster_sd if n_clusters else None
    theta = rng.standard_normal((n_persons, n_dims))
    if group_shift is not None:
        theta += np.asarray(group_shift)[group_id][:, None]
    if n_clusters:
        theta += u[cluster_id][:, None]
    xi = rng.standard_normal((n_persons, latent_dim))
    dist = np.linalg.norm(xi[:, None, :] - zeta[None, :, :], axis=2)
    eta = a[None, :] * theta[:, fid] + b[None, :] - gamma * dist
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return y, fid, a, b, theta, group_id, cluster_id


def _cfg(**kwargs):
    defaults = dict(
        model="MLS2PLM",
        estimator="mmle",
        max_iter=100,
        q_theta=15,
        q_xi=7,
        q_u=11,
    )
    defaults.update(kwargs)
    return FitConfig(**defaults)


def test_marginal_recovers_spatial_model():
    y, fid, a, b, theta, *_ = _simulate_lsirm(seed=11)
    result = fit(y, fid, _cfg())
    trace = np.asarray(result.loglik_trace)
    assert np.all(np.diff(trace) >= -1e-6), "marginal loglik must be non-decreasing"
    assert np.corrcoef(result.params.theta[:, 0], theta[:, 0])[0, 1] > 0.6
    assert result.params.gamma > 0.3
    assert result.population["theta_sd"].shape == result.params.theta.shape


def test_marginal_multigroup_recovers_group_means():
    y, fid, *_ , group_id, _ = _simulate_lsirm(
        seed=12, n_dims=1, latent_dim=1, gamma=0.8, group_shift=[0.0, 1.0],
        n_items=12,
    )
    result = fit(y, fid, _cfg(model="ULS2PLM"), group_id=group_id)
    mu = result.population["mu"]
    assert abs(mu[0, 0]) < 1e-12, "reference group stays pinned"
    assert 0.5 < mu[1, 0] < 1.6, f"group-2 mean should recover ~1.0, got {mu[1, 0]}"


def test_marginal_multilevel_recovers_intercept_sd():
    y, fid, *_ , cluster_id = _simulate_lsirm(
        seed=13, n_persons=600, n_dims=1, latent_dim=1, gamma=0.8,
        cluster_sd=0.8, n_clusters=30, n_items=12,
    )
    result = fit(y, fid, _cfg(model="ULSRM"), cluster_id=cluster_id)
    pop = result.population
    assert 0.35 < pop["sigma_u"] < 1.4, f"sigma_u should recover ~0.8, got {pop['sigma_u']}"
    assert 0.0 < pop["icc"] < 1.0
    assert pop["u_eap"].shape == (30,)


def test_marginal_handles_missing_data():
    y, fid, *_ = _simulate_lsirm(seed=14, n_persons=200, n_items=10)
    rng = np.random.default_rng(0)
    y[rng.random(y.shape) < 0.3] = np.nan
    result = fit(y, fid, _cfg(max_iter=40))
    assert np.all(np.isfinite(result.params.theta))
    assert result.n_iter > 0


def test_numpy_trace_endpoint_matches_returned_parameters_after_max_iter():
    y = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    observed = np.ones_like(y, dtype=bool)
    factor_id = np.zeros(y.shape[1], dtype=np.int64)
    fit_kwargs = {
        "model": "MIRT",
        "n_dims": 1,
        "latent_dim": 1,
        "pop": {"kind": "single"},
        "q_theta": 7,
        "q_xi": 7,
        "q_u": 7,
        "max_iter": 1,
        "m_steps": 2,
        "tol": 1e-5,
        "eps_distance": 1e-8,
        "xi_points": 256,
        "xi_seed": 0,
    }
    result = fit_marginal_numpy(y, observed, factor_id, **fit_kwargs)
    anchors = {
        "fixed": np.ones(y.shape[1], dtype=bool),
        "alpha": result["alpha"].copy(),
        "b": result["b"].copy(),
        "zeta": result["zeta"].copy(),
        "tau": result["tau"],
    }
    reevaluated = fit_marginal_numpy(
        y, observed, factor_id, anchors=anchors, **fit_kwargs
    )

    assert result["n_iter"] == 1
    assert len(result["loglik_trace"]) == 2
    np.testing.assert_allclose(
        result["loglik_trace"][-1], reevaluated["loglik_trace"][0], atol=1e-10
    )


def test_marginal_rejects_invalid_quadrature():
    # #1929: no node-count cap; q_theta=12 is now accepted, only < 1 is not.
    with pytest.raises(ValueError, match="q_theta must be >= 1"):
        FitConfig(q_theta=0).validate()
