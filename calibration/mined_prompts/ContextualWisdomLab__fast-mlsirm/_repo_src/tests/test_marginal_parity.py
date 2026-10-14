"""Rust<->NumPy parity gate for the marginal (MMLE-EM) estimator.

Both backends implement the identical deterministic algorithm (same quadrature
tables, same E/M-step algebra, same init), so agreement is asserted at 1e-9 —
far tighter than the 1e-6 workspace contract — after a full EM run.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.config import FitConfig
from fast_mlsirm.fit import fit
from fast_mlsirm.reference import fit_reference

pytestmark = pytest.mark.skipif(
    pytest.importorskip("fast_mlsirm._core", reason="compiled core required") is None,
    reason="compiled core required",
)


def _simulate(seed=0, n_persons=250, n_items=12, n_dims=2, latent_dim=2, missing=0.0):
    rng = np.random.default_rng(seed)
    fid = np.array([i % n_dims for i in range(n_items)])
    theta = rng.standard_normal((n_persons, n_dims))
    xi = rng.standard_normal((n_persons, latent_dim))
    zeta = rng.standard_normal((n_items, latent_dim)) * 0.8
    dist = np.linalg.norm(xi[:, None, :] - zeta[None, :, :], axis=2)
    eta = theta[:, fid] + 0.2 - dist
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    if missing > 0.0:
        y[rng.random((n_persons, n_items)) < missing] = np.nan
    return y, fid


def _both(y, fid, model, n_dims, **kwargs):
    results = {}
    for backend in ("rust", "numpy"):
        cfg = FitConfig(
            model=model,
            estimator="mmle",
            max_iter=30,
            backend=backend,
            rust_device="cpu",
            q_theta=15,
            q_xi=7,
            q_u=11,
        )
        results[backend] = (
            fit(y, fid, cfg, **kwargs)
            if backend == "rust"
            else fit_reference(y, fid, cfg, **kwargs)
        )
    return results["rust"], results["numpy"]


def _assert_close(r, n, tol=1e-9):
    assert r.optimizer.endswith("/rust")
    assert n.optimizer.endswith("/numpy")
    np.testing.assert_allclose(r.params.b, n.params.b, atol=tol)
    np.testing.assert_allclose(r.params.alpha, n.params.alpha, atol=tol)
    np.testing.assert_allclose(r.params.zeta, n.params.zeta, atol=tol)
    np.testing.assert_allclose(r.params.tau, n.params.tau, atol=tol)
    np.testing.assert_allclose(r.params.theta, n.params.theta, atol=tol)
    np.testing.assert_allclose(r.params.xi, n.params.xi, atol=tol)
    np.testing.assert_allclose(
        r.loglik_trace[-1], n.loglik_trace[-1], rtol=0, atol=tol
    )


@pytest.mark.parametrize("model", ["MLS2PLM", "MLSRM", "MIRT"])
def test_marginal_parity_multidim_models(model):
    y, fid = _simulate(seed=1)
    r, n = _both(y, fid, model, n_dims=2)
    _assert_close(r, n)


@pytest.mark.parametrize("model", ["ULS2PLM", "ULSRM"])
def test_marginal_parity_unidimensional_with_grouping(model):
    # Plain ULS* routes to the legacy fast path, so exercise the marginal path
    # through the population structures.
    y, fid = _simulate(seed=2, n_dims=1)
    group_id = np.arange(len(y)) % 3
    r, n = _both(y, fid, model, n_dims=1, group_id=group_id)
    _assert_close(r, n)
    np.testing.assert_allclose(
        r.population["mu"], n.population["mu"], atol=1e-9
    )
    np.testing.assert_allclose(
        r.population["sigma"], n.population["sigma"], atol=1e-9
    )


def test_marginal_parity_multilevel():
    y, fid = _simulate(seed=3, n_dims=2)
    cluster_id = np.arange(len(y)) % 10
    r, n = _both(y, fid, "MLS2PLM", n_dims=2, cluster_id=cluster_id)
    _assert_close(r, n)
    np.testing.assert_allclose(
        r.population["sigma_u"], n.population["sigma_u"], atol=1e-9
    )
    np.testing.assert_allclose(
        r.population["u_eap"], n.population["u_eap"], atol=1e-9
    )


def test_marginal_parity_with_missing_data():
    y, fid = _simulate(seed=4, missing=0.25)
    r, n = _both(y, fid, "MLS2PLM", n_dims=2)
    _assert_close(r, n)


def test_marginal_parity_theta_sd():
    y, fid = _simulate(seed=5)
    r, n = _both(y, fid, "MLS2PLM", n_dims=2)
    np.testing.assert_allclose(
        r.population["theta_sd"], n.population["theta_sd"], atol=1e-9
    )


def test_marginal_gpu_agrees_with_cpu_loosely(capfd):
    # Compare an explicit f32 GPU E-step with the f64 CPU reference. An auto
    # request can silently fall back to CPU and would not prove GPU parity.
    y, fid = _simulate(seed=6)
    cluster_id = np.arange(len(y)) % 10
    results = {}
    for device in ("cpu", "gpu"):
        cfg = FitConfig(
            model="MLS2PLM",
            estimator="mmle",
            max_iter=15,
            backend="rust",
            rust_device=device,
            q_theta=15,
            q_xi=7,
            q_u=11,
        )
        results[device] = fit(y, fid, cfg, cluster_id=cluster_id)
    device_stderr = capfd.readouterr().err
    if "no usable GPU adapter was found" in device_stderr:
        pytest.skip("no usable GPU adapter; explicit GPU request fell back to CPU")

    r, g = results["cpu"], results["gpu"]
    np.testing.assert_allclose(g.params.b, r.params.b, atol=1e-3)
    np.testing.assert_allclose(g.params.zeta, r.params.zeta, atol=5e-3)
    np.testing.assert_allclose(g.params.theta, r.params.theta, atol=1e-3)
    np.testing.assert_allclose(
        g.population["sigma_u"], r.population["sigma_u"], atol=1e-3
    )
    np.testing.assert_allclose(
        g.loglik_trace[-1], r.loglik_trace[-1], rtol=1e-6, atol=2e-4
    )
