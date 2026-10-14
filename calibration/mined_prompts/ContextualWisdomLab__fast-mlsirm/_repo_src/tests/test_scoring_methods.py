"""EAP/MAP/EAPsum scoring, QMC/MC estimator rules, and FIPC via the public API."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.serving as serving
from fast_mlsirm.config import FitConfig
from fast_mlsirm.estimators.marginal import fit_marginal_numpy
from fast_mlsirm.fit import fit
from fast_mlsirm.reference import fit_reference
from fast_mlsirm.serving import export_serving_bundle, score_respondents, serving_prior


def _simulate(seed=0, P=400, I=12, D=2, gamma=1.0):
    rng = np.random.default_rng(seed)
    fid = np.array([i % D for i in range(I)])
    theta = rng.standard_normal((P, D))
    xi = rng.standard_normal((P, 2))
    zeta = rng.standard_normal((I, 2)) * 0.8
    eta = theta[:, fid] + 0.3 - gamma * np.linalg.norm(xi[:, None] - zeta[None], axis=2)
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    return y, fid


def _bundle(seed=0, **fit_kwargs):
    y, fid = _simulate(seed=seed)
    cfg = FitConfig(
        model="MLS2PLM",
        estimator="mmle",
        max_iter=240,
        tolerance=1e-2,
        q_theta=15,
        q_xi=7,
        **fit_kwargs,
    )
    result = fit(y, fid, cfg)
    codes = [f"I{i}" for i in range(y.shape[1])]
    return y, fid, export_serving_bundle(result, codes, fid, q_theta=15, q_xi=7), codes


def test_map_scoring_matches_eap_loosely_and_reports_se():
    y, fid, bundle, codes = _bundle(seed=1)
    payload = {codes[0]: 1, codes[2]: 1, codes[4]: 0, codes[6]: 0}
    eap = score_respondents(bundle, payload, method="eap")[0]
    mp = score_respondents(bundle, payload, method="map")[0]
    assert mp["converged"]
    for d in range(bundle["n_dims"]):
        assert abs(eap["theta"][d] - mp["theta"][d]) < 0.7
        assert mp["theta_sd"][d] > 0.0
    # MAP shrinks toward the mode of a unimodal posterior — same sign as EAP
    assert np.sign(eap["theta"][0]) == np.sign(mp["theta"][0]) or abs(eap["theta"][0]) < 0.15


def test_eapsum_tables_in_bundle_and_lookup_scoring():
    y, fid, bundle, codes = _bundle(seed=2)
    tables = bundle["eapsum_tables"]
    assert tables is not None and len(tables) == bundle["n_dims"]
    for t in tables:
        assert len(t["eap"]) == t["n_items_dim"] + 1
        assert abs(sum(t["score_prob"]) - 1.0) < 1e-8
        assert all(b >= a - 1e-9 for a, b in zip(t["eap"], t["eap"][1:]))
    # complete response vector -> lookup scoring works and tracks EAP scoring
    full = {c: int(v) for c, v in zip(codes, y[0])}
    via_table = score_respondents(bundle, full, method="eapsum", device="auto")[0]
    via_table_cpu = score_respondents(bundle, full, method="eapsum", device="cpu")[0]
    np.testing.assert_allclose(via_table["theta"], via_table_cpu["theta"], atol=2e-4)
    np.testing.assert_allclose(via_table["theta_sd"], via_table_cpu["theta_sd"], atol=2e-4)
    via_eap = score_respondents(bundle, full, method="eap")[0]
    for d in range(bundle["n_dims"]):
        # summed-score EAP loses the latent-space detail; loose agreement only
        assert abs(via_table["theta"][d] - via_eap["theta"][d]) < 0.8
    # incomplete pattern must be rejected for the lookup path
    with pytest.raises(ValueError, match="complete responses"):
        score_respondents(bundle, {codes[0]: 1}, method="eapsum")


def test_eapsum_scoring_fails_closed_without_rust(monkeypatch):
    _, _, bundle, codes = _bundle(seed=12)
    monkeypatch.setattr(serving, "_core_module", lambda: None)
    with pytest.raises(RuntimeError, match="compiled Rust core"):
        score_respondents(bundle, {code: 1 for code in codes}, method="eapsum")


def test_prior_override_conditions_scores():
    _, _, bundle, codes = _bundle(seed=3)
    payload = {codes[0]: 1, codes[1]: 0}
    n_dims = bundle["n_dims"]
    base = score_respondents(bundle, payload)[0]
    shifted = score_respondents(
        bundle, payload, prior=(np.full(n_dims, 1.0), np.ones(n_dims))
    )[0]
    assert all(s > b for s, b in zip(shifted["theta"], base["theta"]))


def test_serving_prior_widens_for_multilevel_bundles():
    y, fid = _simulate(seed=4, P=300)
    cid = np.arange(len(y)) % 10
    cfg = FitConfig(
        model="MLS2PLM",
        estimator="mmle",
        max_iter=160,
        tolerance=1e-2,
        q_theta=15,
        q_xi=7,
    )
    result = fit(y, fid, cfg, cluster_id=cid)
    bundle = export_serving_bundle(
        result, [f"I{i}" for i in range(y.shape[1])], fid, q_theta=15, q_xi=7
    )
    mean, sd = serving_prior(bundle)
    sigma_u = bundle["population"]["sigma_u"]
    assert np.allclose(mean, 0.0)
    assert np.allclose(sd, np.sqrt(1.0 + sigma_u**2))


@pytest.mark.parametrize("rule", ["qmc", "mc"])
def test_qmc_mc_rules_parity_between_backends(rule):
    y, fid = _simulate(seed=5, P=100, I=8)
    results = {}
    for backend in ("rust", "numpy"):
        cfg = FitConfig(
            model="MLS2PLM",
            estimator="mmle",
            max_iter=150,
            tolerance=1e-2,
            backend=backend,
            rust_device="cpu",
            q_theta=11,
            xi_rule=rule,
            xi_points=32,
            xi_seed=9,
        )
        results[backend] = (
            fit(y, fid, cfg)
            if backend == "rust"
            else fit_reference(y, fid, cfg)
        )
        trace = np.asarray(results[backend].loglik_trace)
        final_change = float(trace[-1] - trace[-2])
        assert results[backend].convergence_status == "converged"
        assert results[backend].n_iter < cfg.max_iter
        assert np.all(np.isfinite(trace))
        assert np.all(np.diff(trace) >= -1e-9)
        assert 0.0 <= final_change < cfg.tolerance
    np.testing.assert_allclose(
        results["rust"].params.b, results["numpy"].params.b, atol=1e-9
    )
    np.testing.assert_allclose(
        results["rust"].loglik_trace[-1], results["numpy"].loglik_trace[-1], atol=1e-9
    )


def test_fipc_public_api_freezes_anchors_and_frees_population():
    rng = np.random.default_rng(11)
    P, I = 600, 12
    fid = np.zeros(I, dtype=np.int64)
    a_true = 0.8 + 0.6 * rng.random(I)
    b_true = -1.0 + 2.0 * rng.random(I)
    theta = 0.8 + rng.standard_normal(P)  # shifted population
    eta = a_true[None, :] * theta[:, None] + b_true[None, :]
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    anchors = dict(
        fixed=np.arange(I) < 6,
        alpha=np.log(a_true),
        b=b_true,
        zeta=np.zeros((I, 1)),
        tau=-30.0,
    )
    cfg = FitConfig(
        model="ULS2PLM", estimator="mmle", max_iter=80, q_theta=15, latent_dim=1
    )
    result = fit(y, fid, cfg, anchors=anchors)
    np.testing.assert_allclose(result.params.b[:6], b_true[:6])
    np.testing.assert_allclose(np.exp(result.params.alpha[:6]), a_true[:6])
    pop = result.population
    assert pop["kind"] == "singlefree"
    np.testing.assert_array_equal(pop["fixed_items"], anchors["fixed"])
    assert pop["tau_fixed"]
    assert 0.4 < pop["mu"][0, 0] < 1.3, f"FIPC mean should recover ~0.8: {pop['mu']}"


def test_fipc_guards():
    y, fid = _simulate(seed=6, P=60, I=6)
    anchors = dict(
        fixed=np.zeros(6, dtype=bool),
        alpha=np.zeros(6),
        b=np.zeros(6),
        zeta=np.zeros((6, 2)),
    )
    with pytest.raises(ValueError):
        fit(y, fid, FitConfig(model="MLS2PLM", estimator="mmle", max_iter=3), anchors=anchors)
    with pytest.raises(ValueError, match="require estimator"):
        fit(y, fid, FitConfig(model="MLS2PLM", estimator="jmle"), anchors=anchors)


@pytest.mark.parametrize("backend", ["rust", "numpy"])
def test_fipc_rejects_unidentified_and_nonfinite_anchor_contracts(backend):
    y, fid = _simulate(seed=17, P=80, I=6)
    base = dict(
        fixed=np.array([True, True, False, False, False, False]),
        alpha=np.zeros(6),
        b=np.zeros(6),
        zeta=np.zeros((6, 2)),
    )
    cfg = FitConfig(
        model="MLS2PLM",
        estimator="mmle",
        backend=backend,
        rust_device="cpu",
        max_iter=3,
    )
    fit_call = fit if backend == "rust" else fit_reference
    with pytest.raises(ValueError, match="at least two fixed anchor items"):
        fit_call(y, fid, cfg, anchors=base)

    finite = dict(base)
    finite["fixed"] = np.ones(6, dtype=bool)
    finite["tau"] = np.nan
    with pytest.raises(ValueError, match="anchor tau must be finite"):
        fit_call(y, fid, cfg, anchors=finite)


@pytest.mark.parametrize("backend", ["rust", "numpy"])
@pytest.mark.parametrize(
    "fixed",
    [
        np.array([np.nan, np.nan, np.nan, np.nan, 0.0, 0.0]),
        np.array(["false", "false", "false", "false", "", ""]),
    ],
    ids=["nan-numeric", "false-string"],
)
def test_fipc_rejects_nonboolean_anchor_masks(backend, fixed):
    y, fid = _simulate(seed=18, P=40, I=6)
    anchors = dict(
        fixed=fixed,
        alpha=np.zeros(6),
        b=np.zeros(6),
        zeta=np.zeros((6, 2)),
    )
    cfg = FitConfig(
        model="MLS2PLM",
        estimator="mmle",
        backend=backend,
        rust_device="cpu",
        max_iter=1,
    )
    fit_call = fit if backend == "rust" else fit_reference
    with pytest.raises(ValueError, match="anchor fixed must contain only boolean"):
        fit_call(y, fid, cfg, anchors=anchors)


def test_numpy_marginal_rejects_nonboolean_anchor_mask_directly():
    y, fid = _simulate(seed=19, P=20, I=6)
    anchors = dict(
        fixed=np.array([np.nan, np.nan, np.nan, np.nan, 0.0, 0.0]),
        alpha=np.zeros(6),
        b=np.zeros(6),
        zeta=np.zeros((6, 2)),
    )
    with pytest.raises(ValueError, match="anchor fixed must contain only boolean"):
        fit_marginal_numpy(
            y,
            np.ones_like(y, dtype=bool),
            fid,
            model="MLS2PLM",
            n_dims=2,
            latent_dim=2,
            pop={"kind": "singlefree"},
            max_iter=1,
            anchors=anchors, q_theta=21, q_xi=11, q_u=15, tol=1e-5, m_steps=4, eps_distance=1e-8, xi_points=256, xi_seed=0
        )
