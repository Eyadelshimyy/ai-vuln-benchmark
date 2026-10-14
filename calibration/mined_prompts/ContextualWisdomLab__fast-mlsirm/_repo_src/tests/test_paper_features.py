"""Tests for the paper-grounded additions: zero inflation, position covariate,
validation gates, IRTree expansion, DIF analysis, Vuong, Q3/GDDM, and ICs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_mlsirm import (
    raju_area,
    velicer_map,
    velicer_map_from_data,
    FitConfig,
    dif_analysis,
    dimensionality_residuals,
    fit,
    irtree_expand,
    validate_judge,
    vuong_nonnested,
)


def _sim_2pl(seed=0, P=600, I=12, shift_by_group=None, gid=None):
    rng = np.random.default_rng(seed)
    fid = np.zeros(I, dtype=np.int64)
    a = 0.8 + 0.6 * rng.random(I)
    b = np.linspace(-1.2, 1.2, I)
    theta = rng.standard_normal(P)
    if shift_by_group is not None:
        theta = theta + np.asarray(shift_by_group)[gid]
    eta = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    return y, fid, a, b


def test_zero_inflation_via_public_api():
    y, fid, *_ = _sim_2pl(seed=1)
    y[:150] = 0.0  # structural zeros: 25%
    cfg = FitConfig(
        model="MLSRM", estimator="mmle", max_iter=60, latent_dim=1,
        q_theta=15, q_xi=7, zero_inflation=True, rust_device="cpu",
    )
    r = fit(y, fid, cfg)
    pop = r.population
    assert 0.1 < pop["pi_zero"] < 0.45
    assert pop["zero_responsibility"][:150].mean() > 0.6
    # ULSRM: (b + zeta) per item + tau + pi = 12*2 + 1 + 1
    assert r.ic is not None and r.ic["n_parameters"] == 26
    plain = fit(y, fid, FitConfig(
        model="MLSRM", estimator="mmle", max_iter=60, latent_dim=1,
        q_theta=15, q_xi=7, rust_device="cpu",
    ))
    assert r.loglik_trace[-1] > plain.loglik_trace[-1]
    # BIC prefers the mixture on mixture data (Kang-Cohen-Sung: BIC decides)
    assert r.ic["bic"] < plain.ic["bic"]


def test_zero_inflation_multilevel_final_scores_use_mixture_posteriors():
    from fast_mlsirm.estimators import marginal as marginal_ref
    from fast_mlsirm.reference import fit_reference

    rng = np.random.default_rng(20260722)
    n_clusters, persons_per_cluster, n_items = 4, 8, 6
    n_persons = n_clusters * persons_per_cluster
    cluster_id = np.repeat(np.arange(n_clusters), persons_per_cluster)
    u_true = np.array([-1.2, -0.3, 0.6, 1.4])
    theta = rng.standard_normal(n_persons) + u_true[cluster_id]
    b_true = np.linspace(-1.0, 1.0, n_items)
    eta = theta[:, None] + b_true[None, :]
    y = (rng.random((n_persons, n_items)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    y[np.flatnonzero(cluster_id == n_clusters - 1)[: persons_per_cluster // 2]] = 0.0
    factor_id = np.zeros(n_items, dtype=np.int64)
    config = FitConfig(
        model="ULS2PLM",
        estimator="mmle",
        backend="numpy",
        latent_dim=1,
        q_theta=7,
        q_xi=7,
        q_u=7,
        max_iter=80,
        tolerance=3e-2,
        zero_inflation=True,
    )
    result = fit_reference(y, factor_id, config, cluster_id=cluster_id)
    assert result.convergence_status == "converged"
    assert result.n_iter < config.max_iter
    assert abs(result.loglik_trace[-1] - result.loglik_trace[-2]) < config.tolerance

    population = result.population
    t_nodes, t_weights = marginal_ref._gh(config.q_theta)
    x_grid, x_logw = marginal_ref._xi_nodes(
        config.xi_rule,
        config.latent_dim,
        config.q_xi,
        config.xi_points,
        config.xi_seed,
    )
    contexts = marginal_ref._build_contexts(
        {
            "kind": "multilevel",
            "cluster_id": cluster_id,
            "n_clusters": n_clusters,
        },
        np.zeros((0, 1)),
        np.zeros((0, 1)),
        population["sigma_u"],
        1,
        config.q_u,
    )
    logp1, logp0, c0 = marginal_ref._build_tables(
        result.params.alpha,
        result.params.b,
        result.params.zeta,
        result.params.tau,
        config.normalized_model(),
        factor_id,
        contexts,
        t_nodes,
        x_grid,
        config.eps_distance,
        1,
        None,
    )
    lp_irt = np.empty((n_persons, config.q_u))
    observed = np.ones_like(y, dtype=bool)
    for node in range(config.q_u):
        context = np.full(n_persons, node, dtype=np.int64)
        _, _, lp_irt[:, node] = marginal_ref._person_logliks(
            y,
            observed,
            factor_id,
            logp1,
            logp0,
            c0,
            np.log(t_weights),
            x_logw,
            context,
            1,
        )
    all_zero = ~(y > 0.0).any(axis=1)
    cluster_post, _ = marginal_ref._multilevel_context_posteriors(
        lp_irt,
        all_zero,
        cluster_id,
        n_clusters,
        contexts["u_logw"],
        population["pi_zero"],
    )
    expected_u = cluster_post @ (population["sigma_u"] * contexts["u_nodes"])
    plain_post, _ = marginal_ref._multilevel_context_posteriors(
        lp_irt,
        all_zero,
        cluster_id,
        n_clusters,
        contexts["u_logw"],
        None,
    )
    plain_u = plain_post @ (population["sigma_u"] * contexts["u_nodes"])

    np.testing.assert_allclose(population["u_eap"], expected_u, atol=1e-12, rtol=0.0)
    assert np.max(np.abs(expected_u - plain_u)) > 0.1


def test_position_covariate_via_public_api():
    rng = np.random.default_rng(3)
    P, I = 800, 10
    fid = np.zeros(I, dtype=np.int64)
    gid = np.arange(P) % 2
    w = np.zeros((2, I))
    w[0] = np.linspace(0, 1, I)
    w[1] = np.linspace(1, 0, I)
    theta = rng.standard_normal(P)
    b = np.linspace(-1, 1, I)
    delta_true = -0.9
    eta = theta[:, None] + b[None, :] + delta_true * w[gid]
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    cfg = FitConfig(model="ULSRM", estimator="mmle", max_iter=80, latent_dim=1,
                    q_theta=15, q_xi=7, rust_device="cpu")
    r = fit(y, fid, cfg, group_id=gid, covariate={"w": w, "init_delta": 0.0})
    assert abs(r.population["delta"] - delta_true) < 0.4, r.population["delta"]
    with pytest.raises(ValueError, match="multilevel"):
        fit(y, fid, cfg, cluster_id=gid, covariate={"w": w})


def test_validation_gates():
    rng = np.random.default_rng(5)
    human = (rng.random(500) < 0.5).astype(np.uint32)
    good = human.copy()
    flip = rng.random(500) < 0.03
    good[flip] = 1 - good[flip]
    verdict = validate_judge(good, human, k=2)
    assert verdict.passed, verdict.failed_gates
    bad = human.copy()
    flip = rng.random(500) < 0.4
    bad[flip] = 1 - bad[flip]
    verdict_bad = validate_judge(bad, human, k=2)
    assert not verdict_bad.passed
    assert "qwk" in verdict_bad.failed_gates


def test_irtree_expand_linear_tree():
    # 3 categories, 2 nodes (linear tree): node0 = "beyond cat0",
    # node1 = "cat2 given beyond cat0" (off-path for cat0)
    mapping = np.array([[0.0, 1.0, 1.0], [np.nan, 0.0, 1.0]])
    y = np.array([[0, 2], [1, np.nan]])
    expanded, factor_id = irtree_expand(y, mapping)
    assert expanded.shape == (2, 4)
    # person 0: item0 cat0 -> node0=0, node1=NaN; item1 cat2 -> node0=1, node1=1
    np.testing.assert_array_equal(expanded[0], [0.0, 1.0, np.nan, 1.0])
    # person 1: item0 cat1 -> node0=1, node1=0; item1 missing -> NaN, NaN
    np.testing.assert_array_equal(expanded[1], [1.0, np.nan, 0.0, np.nan])
    np.testing.assert_array_equal(factor_id, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="integer categories"):
        irtree_expand(np.array([[5.0]]), mapping)


@pytest.mark.parametrize(
    ("responses", "mapping", "node_dims", "message"),
    [
        (np.array([0.0]), np.array([[0.0, 1.0]]), None, "persons x items"),
        (np.empty((0, 1)), np.array([[0.0, 1.0]]), None, "at least one person"),
        (np.array([[np.inf]]), np.array([[0.0, 1.0]]), None, "categories or NaN"),
        (np.array([[0.0]]), np.array([[0.0, np.inf]]), None, "0, 1, or NaN"),
        (np.array([[np.nan]]), np.empty((1, 0)), None, "at least one node"),
        (np.array([[0.0]]), np.empty((0, 1)), None, "at least one node"),
        (
            np.array([[0.0]]),
            np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
            None,
            "distinct tree paths",
        ),
        (
            np.array([[0.0]]),
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            np.array([0, 2**63]),
            "node_dims",
        ),
        (
            np.array([[0.0]]),
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            np.array([1, 1]),
            "contiguous dimension indices",
        ),
    ],
)
def test_irtree_expand_rejects_invalid_tree_contracts(
    responses, mapping, node_dims, message
):
    with pytest.raises(ValueError, match=message):
        irtree_expand(responses, mapping, node_dims=node_dims)


@pytest.mark.parametrize(
    "mapping",
    [
        np.array([[np.nan, 0.0, 1.0], [np.nan, 1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_irtree_expand_rejects_unidentified_paths_and_nodes(mapping):
    with pytest.raises(ValueError):
        irtree_expand(np.array([[0.0]]), mapping)


def test_dif_analysis_detects_injected_shift():
    rng = np.random.default_rng(11)
    P, I = 900, 8
    fid = np.zeros(I, dtype=np.int64)
    gid = (np.arange(P) % 2).astype(np.int64)
    a = np.ones(I)
    b = np.linspace(-1, 1, I)
    theta = rng.standard_normal(P)
    eta = a[None, :] * theta[:, None] + b[None, :]
    eta[:, 3] += np.where(gid == 1, 1.2, 0.0)  # uniform DIF on item 3
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    cfg = FitConfig(model="MIRT", estimator="mmle", max_iter=100,
                    tolerance=0.02, q_theta=15, rust_device="cpu")
    res = dif_analysis(y, fid, gid, config=cfg, studied_items=[2, 3])
    assert res.flagged_bh[3], f"item 3 must flag: p={res.p_value[3]}"
    assert res.effect_size[3] > 0.5
    assert not res.flagged_bh[2] or res.p_value[2] > res.p_value[3]


def test_dif_analysis_rejects_spatial_models_with_unaccounted_positions():
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="latent-space item positions"):
        dif_analysis(
            y,
            np.zeros(2, dtype=np.int64),
            np.array([0, 1]),
            config=FitConfig(model="ULSRM", estimator="mmle"),
            studied_items=[0],
        )


@pytest.mark.parametrize(
    ("group_id", "kwargs", "message"),
    [
        (np.array([0, 1]), {"mask": np.ones((2, 2), dtype=int)}, "boolean"),
        (np.array([0, 0]), {}, "at least two groups"),
        (np.array([0, 1]), {"item_codes": ["item_0"]}, "one entry"),
        (np.array([0, 1]), {"studied_items": [-1]}, "out-of-range"),
        (np.array([0, 1]), {"studied_items": []}, "must not be empty"),
        (np.array([0, 1]), {"studied_items": [0, 0]}, "duplicate"),
        (np.array([0, 1]), {"fdr_q": 0.0}, r"in \(0, 1\]"),
        (np.array([0, 1]), {"fdr_q": np.nan}, r"in \(0, 1\]"),
    ],
)
def test_dif_analysis_rejects_ambiguous_public_inputs(group_id, kwargs, message):
    with pytest.raises(ValueError, match=message):
        dif_analysis(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.zeros(2, dtype=np.int64),
            group_id,
            **kwargs,
        )


def test_dif_analysis_compacts_sparse_group_labels(monkeypatch):
    """Equivalent group partitions must have identical DIF bookkeeping."""
    import importlib
    from types import SimpleNamespace

    fit_module = importlib.import_module("fast_mlsirm.fit")
    seen_group_ids = []

    def fake_fit(y, factor_id, config, group_id=None, anchors=None, **_kwargs):
        seen_group_ids.append(np.asarray(group_id).copy())
        n_items = y.shape[1]
        return SimpleNamespace(
            params=SimpleNamespace(
                alpha=np.zeros(n_items),
                b=np.arange(n_items, dtype=float),
                zeta=np.zeros((n_items, 1)),
                tau=0.0,
            ),
            convergence_status="converged",
            n_iter=2,
            loglik_trace=[0.0, 1.0] if anchors is not None else [-1.0, 0.0],
        )

    monkeypatch.setattr(fit_module, "fit", fake_fit)
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = dif_analysis(
        y,
        np.zeros(2, dtype=np.int64),
        np.array([10, 20]),
        studied_items=[0],
    )

    assert result.df[0] == 2.0
    assert result.b_by_group.shape == (2, 2)
    assert result.effect_size[0] == 1.0
    assert all(np.array_equal(gid, [0, 1]) for gid in seen_group_ids)


def test_dif_analysis_rejects_nonconverged_constrained_fit(monkeypatch):
    import importlib
    from types import SimpleNamespace

    fit_module = importlib.import_module("fast_mlsirm.fit")

    def fake_fit(*_args, **_kwargs):
        return SimpleNamespace(
            convergence_status="max_iter_reached",
            n_iter=3,
            loglik_trace=[-3.0, -2.0, -1.5],
        )

    monkeypatch.setattr(fit_module, "fit", fake_fit)
    with pytest.raises(
        RuntimeError,
        match=r"dif_analysis requires converged parameters.*status=max_iter_reached",
    ):
        dif_analysis(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.zeros(2, dtype=np.int64),
            np.array([0, 1]),
            studied_items=[0],
        )


def test_vuong_and_dimensionality_wrappers():
    y, fid, *_ = _sim_2pl(seed=13, P=400, I=10)
    cfg = FitConfig(model="ULSRM", estimator="mmle", max_iter=40, latent_dim=1,
                    q_theta=15, q_xi=7, rust_device="cpu", zero_inflation=False)
    r = fit(y, fid, cfg)
    # Vuong on synthetic casewise logliks
    la = -1.0 + 0.1 * np.random.default_rng(0).random(400)
    lb = la - 0.15 - 0.2 * (np.random.default_rng(1).random(400) - 0.5)
    v = vuong_nonnested(la, lb, 10, 10, bic_correction=False)
    assert v["z"] > 0 and 0 <= v["p_two_sided"] <= 1
    # Residual diagnostics on a converged, well-fitting model: modest Q3 and
    # a small repository-specific mean absolute residual cross-product.
    d = dimensionality_residuals(
        y,
        fid,
        r.params,
        r.model,
        convergence_status=r.convergence_status,
    )
    assert d["q3"].shape[0] == 10 * 9 // 2
    assert d["q3_max_abs"] < 0.5
    assert d["mean_abs_residual_cross_product"] < 0.05
    assert d["gddm"] == d["mean_abs_residual_cross_product"]


def test_dimensionality_residuals_bifactor_uses_inner_product():
    from fast_mlsirm import _core
    from fast_mlsirm.types import MLSIRMParams

    params = MLSIRMParams(
        theta=np.array([[-1.0], [0.2], [0.8], [-0.4], [1.1], [0.0]]),
        alpha=np.log(np.array([1.0, 1.3, 0.8])),
        b=np.array([-0.2, 0.3, -0.1]),
        xi=np.array([[-1.5], [1.2], [0.7], [-0.8], [1.6], [-0.1]]),
        zeta=np.array([[1.1], [-0.9], [0.6]]),
        tau=np.log(2.0),
    )
    factor_id = np.zeros(3, dtype=np.int64)
    eta = (
        np.exp(params.alpha)[None, :] * params.theta[:, factor_id]
        + params.b[None, :]
        + params.xi @ params.zeta.T
    )
    probability = 1.0 / (1.0 + np.exp(-eta))
    responses = (probability >= 0.5).astype(float)
    expected = dict(
        _core.dimensionality_residuals((responses - probability).ravel(), 6, 3)
    )

    actual = dimensionality_residuals(
        responses,
        factor_id,
        params,
        "BIFAC2PLM",
        convergence_status="converged",
    )
    np.testing.assert_allclose(actual["q3"], expected["q3"])
    assert actual["mean_abs_residual_cross_product"] == pytest.approx(
        expected["mean_abs_residual_cross_product"]
    )


def test_dimensionality_residuals_rejects_unfinished_fit_and_invalid_inputs():
    from fast_mlsirm.types import MLSIRMParams

    params = MLSIRMParams(
        theta=np.zeros((3, 1)),
        alpha=np.zeros(2),
        b=np.zeros(2),
        xi=np.zeros((3, 1)),
        zeta=np.zeros((2, 1)),
        tau=0.0,
    )
    y = np.zeros((3, 2))
    factor_id = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="did not converge"):
        dimensionality_residuals(
            y, factor_id, params, "MIRT", convergence_status="max_iter_reached"
        )
    with pytest.raises(ValueError, match="model must be one of"):
        dimensionality_residuals(y, factor_id, params, "not-a-model")
    with pytest.raises(ValueError, match="mask shape"):
        dimensionality_residuals(y, factor_id, params, "MIRT", mask=np.ones(3))
    with pytest.raises(ValueError, match="observed responses must be 0 or 1"):
        dimensionality_residuals(
            np.array([[0.0, 2.0], [0.0, 1.0], [1.0, 0.0]]),
            factor_id,
            params,
            "MIRT",
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_vuong_rejects_nonfinite_casewise_loglikelihoods(bad):
    with pytest.raises(ValueError, match="finite"):
        vuong_nonnested(np.array([0.0, bad]), np.array([0.0, 1.0]), 1, 1)


@pytest.mark.parametrize(
    "k_a,k_b,bic_correction,match",
    [
        (1.5, 1, True, "k_a"),
        (True, 1, True, "k_a"),
        (1, -1, True, "k_b"),
        (1, 1, 1, "bic_correction"),
    ],
)
def test_vuong_rejects_lossy_parameter_controls(k_a, k_b, bic_correction, match):
    with pytest.raises(ValueError, match=match):
        vuong_nonnested(
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            k_a,
            k_b,
            bic_correction=bic_correction,
        )


def test_bifactor_parity_and_recovery():
    from fast_mlsirm.reference import fit_reference

    rng = np.random.default_rng(21)
    P, I, D = 500, 10, 2
    fid = np.array([i % D for i in range(I)])
    lam = 0.6 + 0.8 * rng.random(I)
    b = np.linspace(-1, 1, I)
    g = rng.standard_normal(P)
    th = rng.standard_normal((P, D))
    eta = th[:, fid] + b[None, :] + lam[None, :] * g[:, None]
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    max_iter = 300
    tolerance = 1e-4
    results = {}
    for backend in ("rust", "numpy"):
        cfg = FitConfig(
            model="BIFAC2PLM",
            estimator="mmle",
            max_iter=max_iter,
            backend=backend,
            rust_device="cpu",
            latent_dim=1,
            q_theta=15,
            q_xi=15,
            tolerance=tolerance,
        )
        results[backend] = (fit_reference if backend == "numpy" else fit)(y, fid, cfg)
    r, n = results["rust"], results["numpy"]
    for result in results.values():
        trace = np.asarray(result.loglik_trace)
        assert result.convergence_status == "converged"
        assert result.n_iter < max_iter
        assert np.all(np.isfinite(trace))
        assert np.all(np.diff(trace) >= -1e-10)
        assert abs(trace[-1] - trace[-2]) < tolerance
    np.testing.assert_allclose(r.params.b, n.params.b, atol=1e-9)
    np.testing.assert_allclose(r.params.zeta, n.params.zeta, atol=1e-9)
    np.testing.assert_allclose(r.loglik_trace[-1], n.loglik_trace[-1], atol=1e-9)
    # loadings track the truth
    c = np.corrcoef(r.params.zeta[:, 0], lam)[0, 1]
    assert abs(c) > 0.5, f"lambda recovery: {c}"
    # jmle guard: rejected at FitConfig validation before fit work
    import pytest as _pytest

    with _pytest.raises(ValueError, match="BIFAC2PLM.*mmle"):
        fit(y, fid, FitConfig(model="BIFAC2PLM", estimator="jmle"))


def test_bifactor_covariate_parity_uses_inner_product_predictor():
    from fast_mlsirm.reference import fit_reference

    rng = np.random.default_rng(22)
    P, I, D = 400, 8, 2
    fid = np.arange(I) % D
    gid = np.arange(P) % 2
    w = np.empty((2, I))
    w[0] = np.linspace(0.0, 1.0, I)
    w[1] = w[0, ::-1]
    theta = rng.standard_normal((P, D))
    general = rng.standard_normal(P)
    loadings = np.linspace(0.7, 1.3, I)
    intercepts = np.linspace(-0.8, 0.8, I)
    delta_true = -1.0
    eta = (
        theta[:, fid]
        + intercepts[None, :]
        + loadings[None, :] * general[:, None]
        + delta_true * w[gid]
    )
    y = (rng.random((P, I)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)

    max_iter = 300
    tolerance = 1e-4
    results = {}
    for backend in ("rust", "numpy"):
        cfg = FitConfig(
            model="BIFAC2PLM",
            estimator="mmle",
            max_iter=max_iter,
            tolerance=tolerance,
            backend=backend,
            rust_device="cpu",
            latent_dim=1,
            q_theta=7,
            q_xi=7,
        )
        results[backend] = (fit_reference if backend == "numpy" else fit)(
            y,
            fid,
            cfg,
            group_id=gid,
            covariate={"w": w, "init_delta": 0.0},
        )

    for result in results.values():
        trace = np.asarray(result.loglik_trace)
        assert result.convergence_status == "converged"
        assert result.n_iter < max_iter
        assert np.all(np.isfinite(trace))
        assert np.all(np.diff(trace) >= -1e-10)
        assert abs(trace[-1] - trace[-2]) < tolerance

    rust_delta = results["rust"].population["delta"]
    numpy_delta = results["numpy"].population["delta"]
    assert rust_delta < -0.2
    np.testing.assert_allclose(rust_delta, numpy_delta, atol=1e-8)
    np.testing.assert_allclose(
        results["rust"].loglik_trace[-1],
        results["numpy"].loglik_trace[-1],
        atol=1e-8,
    )


def test_m2_rmsea2_parity_and_fit():
    # M2 limited-information GOF (Maydeu-Olivares & Joe): Rust core vs the
    # NumPy reference, plus a well-specified-vs-local-dependence contrast.
    from fast_mlsirm import fitstats

    y, fid, _a, _b = _sim_2pl(seed=5, P=1800, I=12)
    res = fit(y, fid, FitConfig(model="MIRT", estimator="mmle", max_iter=200,
                                backend="rust", rust_device="cpu"))

    core = fitstats.m2(y, fid, res.params, "MIRT", q_theta=21)
    ref = fitstats._m2_numpy(y, ~np.isnan(y), fid, res.params, "MIRT", 21, 11, 1e-8)

    # exact structural agreement
    assert core.n_moments == ref.n_moments == 78
    assert core.n_parameters == ref.n_parameters == 24
    assert core.df == ref.df == 54.0
    assert core.n_complete == ref.n_complete == 1800
    # numeric parity (hand Cholesky vs LAPACK solve): tight but not bit-exact
    np.testing.assert_allclose(core.m2, ref.m2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(core.rmsea2, ref.rmsea2, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.srmsr, ref.srmsr, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.null_m2, ref.null_m2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(core.cfi, ref.cfi, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.tli, ref.tli, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.rmsea2_ci_lower, ref.rmsea2_ci_lower, atol=1e-6)
    np.testing.assert_allclose(core.rmsea2_ci_upper, ref.rmsea2_ci_upper, atol=1e-6)
    assert core.null_df == ref.null_df == 66.0
    assert core.rmsea == core.rmsea2
    assert core.srmr == core.srmsr
    expected_cfi = np.clip(
        1.0 - (core.m2 - core.df) / (core.null_m2 - core.null_df), 0.0, 1.0
    )
    expected_tli = (
        (core.null_m2 / core.null_df) - (core.m2 / core.df)
    ) / ((core.null_m2 / core.null_df) - 1.0)
    np.testing.assert_allclose(core.cfi, expected_cfi, atol=1e-12)
    np.testing.assert_allclose(core.tli, expected_tli, atol=1e-12)

    # The main diagnostics path exposes the global indices only when explicitly
    # requested, keeping the O(s^3) M2 solve out of ordinary JML diagnostics.
    from fast_mlsirm import fit_diagnostics

    diag = fit_diagnostics(
        y, res.params, fid, model="MIRT", include_m2=True, estimator="mmle"
    )
    for key in ("m2", "rmsea", "srmr", "cfi", "tli", "null_m2"):
        assert key in diag.model_fit and np.isfinite(diag.model_fit[key])
    np.testing.assert_allclose(diag.model_fit["rmsea"], core.rmsea2, atol=1e-12)
    np.testing.assert_allclose(diag.model_fit["srmr"], core.srmsr, atol=1e-12)

    # well specified: small RMSEA2, CI brackets the point estimate
    assert core.rmsea2 < 0.03
    assert core.rmsea2_ci_lower <= core.rmsea2 + 1e-9 <= core.rmsea2_ci_upper + 1e-9

    # inject local dependence (duplicate item) -> M2 and RMSEA2 inflate
    y_ld = y.copy()
    y_ld[:, 1] = y_ld[:, 0]
    res_ld = fit(y_ld, fid, FitConfig(model="MIRT", estimator="mmle", max_iter=200,
                                      backend="rust", rust_device="cpu"))
    ld = fitstats.m2(y_ld, fid, res_ld.params, "MIRT", q_theta=21)
    assert ld.m2 > core.m2
    assert ld.rmsea2 > 0.08
    assert ld.srmsr > core.srmsr
    assert ld.cfi < core.cfi
    assert ld.tli < core.tli

    # JMLE/CMLE estimates may still be inspected against an explicit marginal
    # evaluation population, but ordinary chi-square inference is not claimed.
    descriptive = fitstats.m2(
        y, fid, res.params, "MIRT", q_theta=21, estimator="jmle"
    )
    assert np.isfinite(descriptive.m2)
    assert not descriptive.inference_valid
    assert np.isnan(descriptive.p_value)
    assert np.isnan(descriptive.rmsea2_ci_lower)


def test_m2_noncentral_ci_large_statistic_reference_values():
    """Large noncentralities must not collapse when exp(-lambda/2) underflows."""
    from fast_mlsirm.fitstats import _nc_lambda_for, _ncchi2_cdf

    # Independently evaluated with scipy.stats.ncx2 and scipy.optimize.brentq.
    cases = [
        (2_000.0, 50.0, 0.05, 2_099.9287582915094),
        (10_000.0, 50.0, 0.05, 10_282.274417418035),
        (10_000.0, 50.0, 0.95, 9_625.139462181574),
    ]
    for statistic, df, target, expected in cases:
        got = _nc_lambda_for(statistic, df, target)
        np.testing.assert_allclose(got, expected, rtol=1e-10)
        np.testing.assert_allclose(
            _ncchi2_cdf(statistic, df, got), target, atol=1e-10
        )


def test_m2_singlefree_uses_only_estimated_calibration_columns():
    """FIPC M2 counts free-population columns and excludes anchored items."""
    from fast_mlsirm import fit_diagnostics
    from fast_mlsirm import fitstats
    from fast_mlsirm.types import MLSIRMParams

    rng = np.random.default_rng(2718)
    n_persons, n_items = 1600, 8
    factor_id = np.zeros(n_items, dtype=np.int64)
    alpha = np.log(np.linspace(0.8, 1.4, n_items))
    b = np.linspace(-1.0, 1.0, n_items)
    population_mean = np.array([0.4])
    population_sd = np.array([1.2])
    theta = population_mean[0] + population_sd[0] * rng.standard_normal(n_persons)
    probability = 1.0 / (
        1.0 + np.exp(-(theta[:, None] * np.exp(alpha)[None, :] + b[None, :]))
    )
    responses = (rng.random(probability.shape) < probability).astype(float)
    params = MLSIRMParams(
        theta=theta[:, None],
        alpha=alpha,
        b=b,
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    fixed_items = np.arange(n_items) < 3

    ordinary = fitstats.m2(
        responses,
        factor_id,
        params,
        "MIRT",
        prior_mean=population_mean,
        prior_sd=population_sd,
    )
    singlefree = fitstats.m2(
        responses,
        factor_id,
        params,
        "MIRT",
        prior_mean=population_mean,
        prior_sd=population_sd,
        estimate_population=True,
        fixed_items=fixed_items,
    )

    # Ordinary 2PL estimates 2I item columns. FIPC fixes three item rows and
    # instead estimates the population mean and SD: 2*(8-3) + 2 = 12.
    assert ordinary.n_parameters == 16
    assert singlefree.n_parameters == 12
    assert singlefree.n_moments == 36
    assert singlefree.df == 24.0
    assert np.isfinite(singlefree.m2)
    assert np.isfinite(singlefree.p_value)
    assert "estimated mean/SD" in singlefree.inference_note

    diagnostics = fit_diagnostics(
        responses,
        params,
        factor_id,
        model="MIRT",
        include_m2=True,
        estimator="mmle",
        convergence_status="converged",
        population={
            "kind": "singlefree",
            "mu": population_mean[None, :],
            "sigma": population_sd[None, :],
            "fixed_items": fixed_items,
            "tau_fixed": False,
        },
    )
    assert diagnostics.model_fit["m2_df"] == 24.0
    assert diagnostics.model_fit["m2"] == singlefree.m2

    with pytest.raises(ValueError, match="fixed_items must have shape"):
        fitstats.m2(
            responses,
            factor_id,
            params,
            "MIRT",
            prior_mean=population_mean,
            prior_sd=population_sd,
            estimate_population=True,
            fixed_items=np.zeros(n_items - 1, dtype=bool),
        )
    with pytest.raises(ValueError, match="requires estimator='mmle'"):
        fitstats.m2(
            responses,
            factor_id,
            params,
            "MIRT",
            estimator="jmle",
            fixed_items=fixed_items,
        )


def test_m2_multigroup_and_multilevel_structures():
    """Population structure changes M2 moments/covariance, not just labels."""
    from fast_mlsirm import fit_diagnostics, m2_multigroup, m2_multilevel
    from fast_mlsirm.types import MLSIRMParams

    rng = np.random.default_rng(123)
    n_items = 10
    n_per_group = 700
    factor_id = np.zeros(n_items, dtype=np.int64)
    alpha = np.log(np.linspace(0.8, 1.4, n_items))
    b = np.linspace(-1.0, 1.0, n_items)
    group_id = np.repeat(np.arange(2), n_per_group)
    means = np.array([[0.0], [0.6]])
    sds = np.array([[1.0], [1.2]])
    theta = means[group_id, 0] + sds[group_id, 0] * rng.standard_normal(group_id.size)
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] * np.exp(alpha) + b)))
    responses = (rng.random(prob.shape) < prob).astype(float)
    params = MLSIRMParams(
        theta=np.zeros((responses.shape[0], 1)),
        alpha=alpha,
        b=b,
        xi=np.zeros((responses.shape[0], 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    group_fit = m2_multigroup(
        responses, factor_id, params, "MIRT", group_id, means, sds
    )
    assert group_fit.n_groups == 2
    assert group_fit.n_parameters == 2 * n_items + 2
    assert group_fit.df == 88.0
    assert group_fit.null_df == 90.0
    assert np.isfinite(group_fit.m2)
    assert np.isfinite(group_fit.cfi)
    with pytest.raises(ValueError, match="non-negative integers"):
        m2_multigroup(
            responses,
            factor_id,
            params,
            "MIRT",
            group_id.astype(float) + 0.25,
            means,
            sds,
        )
    with pytest.raises(ValueError, match="mask must match"):
        m2_multigroup(
            responses,
            factor_id,
            params,
            "MIRT",
            group_id,
            means,
            sds,
            mask=np.ones((1, 1), dtype=bool),
        )
    group_diagnostics = fit_diagnostics(
        responses,
        params,
        factor_id,
        model="MIRT",
        group_id=group_id,
        include_m2=True,
        estimator="mmle",
        population={"kind": "multigroup", "mu": means, "sigma": sds},
    )
    np.testing.assert_allclose(group_diagnostics.model_fit["m2"], group_fit.m2)
    assert group_diagnostics.model_fit["m2_n_groups"] == 2.0

    locally_dependent = responses.copy()
    locally_dependent[:, 1] = locally_dependent[:, 0]
    group_ld = m2_multigroup(
        locally_dependent, factor_id, params, "MIRT", group_id, means, sds
    )
    assert group_ld.m2 > group_fit.m2
    assert group_ld.cfi < group_fit.cfi
    assert group_ld.tli < group_fit.tli

    # Random-intercept data: the effective covariance comes from independent
    # cluster totals. There must be more clusters than retained M2 moments.
    n_items = 8
    n_clusters = 70
    cluster_size = 12
    sigma_u = 0.7
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    alpha = np.log(np.linspace(0.9, 1.3, n_items))
    b = np.linspace(-0.8, 0.8, n_items)
    u = np.repeat(sigma_u * rng.standard_normal(n_clusters), cluster_size)
    theta = rng.standard_normal(cluster_id.size) + u
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] * np.exp(alpha) + b)))
    responses = (rng.random(prob.shape) < prob).astype(float)
    params = MLSIRMParams(
        theta=np.zeros((responses.shape[0], 1)),
        alpha=alpha,
        b=b,
        xi=np.zeros((responses.shape[0], 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    cluster_fit = m2_multilevel(
        responses, np.zeros(n_items, dtype=np.int64), params, "MIRT",
        cluster_id, sigma_u,
    )
    assert cluster_fit.n_clusters == n_clusters
    assert cluster_fit.n_parameters == 2 * n_items + 1
    assert cluster_fit.df == 19.0
    assert np.isfinite(cluster_fit.m2)
    assert np.isfinite(cluster_fit.p_value)
    assert "cluster-robust" in cluster_fit.inference_note
    cluster_diagnostics = fit_diagnostics(
        responses,
        params,
        np.zeros(n_items, dtype=np.int64),
        model="MIRT",
        cluster_id=cluster_id,
        include_m2=True,
        estimator="mmle",
        population={"kind": "multilevel", "sigma_u": sigma_u},
    )
    np.testing.assert_allclose(cluster_diagnostics.model_fit["m2"], cluster_fit.m2)
    assert cluster_diagnostics.model_fit["m2_n_clusters"] == float(n_clusters)


def test_m2_multilevel_integrates_one_shared_intercept_across_dimensions():
    """The scalar cluster effect induces cross-factor covariance."""
    from fast_mlsirm.fitstats import _m2_group_components
    from fast_mlsirm.types import MLSIRMParams

    factor_id = np.array([0, 0, 1, 1], dtype=np.int64)
    responses = np.zeros((20, 4), dtype=float)
    params = MLSIRMParams(
        theta=np.zeros((20, 2)),
        alpha=np.zeros(4),
        b=np.zeros(4),
        xi=np.zeros((20, 1)),
        zeta=np.zeros((4, 1)),
        tau=0.0,
    )
    common = dict(
        y0=responses,
        observed0=np.ones_like(responses, dtype=bool),
        d_of_i=factor_id,
        params=params,
        model="MIRT",
        q_theta=15,
        q_xi=7,
        eps_distance=1e-8,
        prior_mean=np.zeros(2),
        prior_sd=np.ones(2),
    )
    independent = _m2_group_components(**common)
    shared = _m2_group_components(**common, shared_sigma_u=0.8, q_u=15)

    # Pair order after four univariate moments is (0,1), (0,2), ... .
    cross_factor_pair = 5
    assert abs(independent["mom"][cross_factor_pair] - 0.25) < 1e-12
    assert shared["mom"][cross_factor_pair] > independent["mom"][cross_factor_pair]
    assert np.any(np.abs(shared["delta_shared"]) > 1e-8)


def test_m2_cmle_rasch_conditions_out_person_ability():
    """CMLE M2 uses raw-score conditional moments, not a Gaussian prior."""
    from fast_mlsirm import m2
    from fast_mlsirm.types import MLSIRMParams

    rng = np.random.default_rng(9)
    n_persons, n_items = 3000, 8
    b = np.linspace(-1.3, 1.3, n_items)
    theta = rng.standard_normal(n_persons)
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] + b)))
    responses = (rng.random(prob.shape) < prob).astype(float)
    assert np.all(np.bincount(responses.sum(axis=1).astype(int), minlength=n_items + 1) > 0)
    params = MLSIRMParams(
        theta=theta[:, None],
        alpha=np.zeros(n_items),
        b=b,
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    result = m2(
        responses,
        np.zeros(n_items, dtype=np.int64),
        params,
        "MIRT",
        estimator="cmle",
    )
    assert result.estimator == "cmle"
    assert result.inference_valid
    assert result.n_parameters == (n_items - 1) + n_items
    assert result.df == 21.0
    assert result.p_value > 0.05
    assert result.rmsea2 < 0.02

    non_rasch = MLSIRMParams(
        theta=params.theta,
        alpha=np.full(n_items, 0.1),
        b=b,
        xi=params.xi,
        zeta=params.zeta,
        tau=0.0,
    )
    with pytest.raises(ValueError, match="Rasch"):
        m2(
            responses,
            np.zeros(n_items, dtype=np.int64),
            non_rasch,
            "MIRT",
            estimator="cmle",
        )


def test_irt_link_recovers_known_transform():
    # separately-calibrated common-item linking (Kolen & Brennan; Haebara;
    # Stocking-Lord): recover a known theta_old = A*theta_new + B.
    from fast_mlsirm import irt_link

    a_old = np.array([1.2, 0.8, 1.5, 1.0, 0.9, 1.3, 1.1, 0.7])
    b_old = np.array([-0.5, 0.3, 1.0, -1.2, 0.0, 0.6, -0.8, 0.4])
    A0, B0 = 1.3, 0.4
    a_new = A0 * a_old
    b_new = b_old + a_old * B0
    for method in ("mean_mean", "mean_sigma", "haebara", "stocking_lord"):
        r = irt_link(a_old, b_old, a_new, b_new, method=method)
        assert abs(r.slope - A0) < 1e-3, f"{method}: slope {r.slope}"
        assert abs(r.intercept - B0) < 1e-3, f"{method}: intercept {r.intercept}"
        assert r.converged
        if method in {"haebara", "stocking_lord"}:
            assert r.termination_reason == "tolerance_met"
            assert r.n_iter < r.max_iter
            assert r.final_objective_span <= r.objective_tolerance
            assert r.final_parameter_span <= r.parameter_tolerance
        else:
            assert r.termination_reason == "closed_form"
            assert r.n_iter == r.max_iter == 0
    import pytest as _pytest

    with _pytest.raises(Exception):
        irt_link(a_old, b_old, a_new, b_new, method="not_a_method")
    with _pytest.raises(ValueError, match="non-zero difficulty spread"):
        irt_link(
            np.ones(3),
            np.array([-0.5, 0.0, 0.5]),
            np.ones(3),
            np.zeros(3),
            method="mean_sigma",
        )
    with _pytest.raises(ValueError, match="integer quadrature size"):
        irt_link(a_old, b_old, a_new, b_new, q_theta=21.5)


def test_category_logprobs_binary_parity_and_gpcm_monotone():
    """The unified GPCM/nominal cell nests binary 2PL (bit-parity check) and is
    a proper log-softmax; GPCM scores make higher `base` favor higher categories.
    Parity reference for the Rust polytomous kernel (design spec)."""
    import numpy as np
    from fast_mlsirm.estimators.marginal import compute_category_logprobs

    rng = np.random.default_rng(0)
    base = rng.normal(size=32)
    b = 0.3

    # binary 2PL: logP_1 == log_sigmoid(base + b), logP_0 == log_sigmoid(-(base+b))
    lp = compute_category_logprobs(base, [0.0, 1.0], [0.0, b])
    assert np.allclose(np.exp(lp).sum(axis=-1), 1.0, atol=1e-12)
    eta = base + b
    assert np.allclose(lp[:, 1], -np.logaddexp(0.0, -eta), atol=1e-12)
    assert np.allclose(lp[:, 0], -np.logaddexp(0.0, eta), atol=1e-12)

    # GPCM (scores 0,1,2): larger base shifts mass to the top category
    lp3 = compute_category_logprobs(np.array([-2.0, 2.0]), [0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert np.allclose(np.exp(lp3).sum(axis=-1), 1.0, atol=1e-12)
    p_top = np.exp(lp3[:, 2])
    assert p_top[1] > p_top[0]

    # baseline must be pinned
    import pytest
    with pytest.raises(ValueError):
        compute_category_logprobs(base, [0.5, 1.0], [0.0, b])


def test_gpcm_node_gradient_matches_finite_difference():
    """The analytic M-step gradient of the GPCM/nominal cell (category residual,
    score-weighted base residual, and nominal-score gradient) matches central
    finite differences — de-risks the Rust M-step before it is written."""
    import numpy as np
    from fast_mlsirm.estimators.marginal import compute_category_logprobs, compute_gpcm_node_gradient

    scores = np.array([0.0, 1.0, 2.0, 3.0])
    intercepts = np.array([0.0, 0.2, -0.1, 0.3])
    counts = np.array([3.0, 5.0, 2.0, 4.0])
    base = 0.4

    def q(b, ic, sc):
        return float(np.dot(counts, compute_category_logprobs(b, sc, ic)))

    g_ic, g_base, g_sc = compute_gpcm_node_gradient(base, scores, intercepts, counts)
    h = 1e-6
    for m in range(1, 4):
        ic_p, ic_m = intercepts.copy(), intercepts.copy()
        ic_p[m] += h
        ic_m[m] -= h
        assert abs((q(base, ic_p, scores) - q(base, ic_m, scores)) / (2 * h) - g_ic[m - 1]) < 1e-5
        sc_p, sc_m = scores.copy(), scores.copy()
        sc_p[m] += h
        sc_m[m] -= h
        assert abs((q(base, intercepts, sc_p) - q(base, intercepts, sc_m)) / (2 * h) - g_sc[m - 1]) < 1e-5
    assert abs((q(base + h, intercepts, scores) - q(base - h, intercepts, scores)) / (2 * h) - g_base) < 1e-5

    # residual closure
    p = np.exp(compute_category_logprobs(base, scores, intercepts))
    assert abs((counts - counts.sum() * p).sum()) < 1e-12


def test_fit_gpcm_numpy_recovers_known_parameters():
    """Unidimensional GPCM MMLE-EM (the polytomous parity reference) recovers
    known slopes and category intercepts from simulated data."""
    import numpy as np
    from fast_mlsirm.estimators.marginal import compute_category_logprobs, fit_gpcm_numpy

    rng = np.random.default_rng(7)
    n_persons, n_items, k_cat = 4000, 6, 3
    a_true = rng.uniform(0.8, 1.8, n_items)
    c_true = np.zeros((n_items, k_cat))
    c_true[:, 1:] = rng.normal(0.0, 1.0, (n_items, k_cat - 1))
    theta = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k_cat, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(compute_category_logprobs(a_true[i] * theta, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k_cat, p=p[pp])

    res = fit_gpcm_numpy(y, k_cat, max_iter=80, q_theta=21, tol=1e-6)
    assert np.isfinite(res["loglik"])
    assert res["converged"]
    assert res["termination_reason"] == "tolerance"
    assert res["n_iter"] < 80
    assert res["loglik_trace"].shape == (res["n_iter"] + 1,)
    assert np.all(np.isfinite(res["loglik_trace"]))
    assert np.all(np.diff(res["loglik_trace"]) >= -1e-10)
    assert res["final_delta"] <= res["stopping_tolerance"]
    assert res["loglik"] == res["loglik_trace"][-1]
    assert np.corrcoef(a_true, res["a"])[0, 1] > 0.9
    assert np.max(np.abs(a_true - res["a"])) < 0.35
    assert np.mean(np.abs(c_true[:, 1:] - res["intercepts"][:, 1:])) < 0.2


def test_fit_gpcm_numpy_reports_likelihood_at_returned_parameters():
    """The reference EM result and trace end at the returned parameter state."""
    import numpy as np

    from fast_mlsirm.estimators.marginal import _gh, compute_category_logprobs, fit_gpcm_numpy

    y = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [1, 2, 1],
            [2, 2, 2],
            [2, 1, 2],
            [1, 0, 1],
            [2, 2, 1],
        ],
        dtype=np.int64,
    )
    res = fit_gpcm_numpy(y, 3, q_theta=7, max_iter=1, tol=1e-6)

    nodes, weights = _gh(7)
    scores = np.arange(3, dtype=np.float64)
    log_node = np.zeros((y.shape[0], nodes.size), dtype=np.float64)
    for item in range(y.shape[1]):
        item_lp = compute_category_logprobs(
            res["a"][item] * nodes, scores, res["intercepts"][item]
        )
        log_node += item_lp[:, y[:, item]].T
    log_node += np.log(weights)[None, :]
    maximum = log_node.max(axis=1, keepdims=True)
    reevaluated = float(
        np.sum(maximum[:, 0] + np.log(np.exp(log_node - maximum).sum(axis=1)))
    )

    assert res["n_iter"] == 1
    assert not res["converged"]
    assert res["termination_reason"] == "max_iter_reached"
    assert res["loglik_trace"].shape == (2,)
    assert np.all(np.isfinite(res["loglik_trace"]))
    assert res["final_delta"] > res["stopping_tolerance"]
    assert np.allclose(res["loglik"], res["loglik_trace"][-1], atol=1e-12)
    assert np.allclose(res["loglik"], reevaluated, atol=1e-12)


def test_fit_gpcm_numpy_rejects_malformed_controls_and_responses():
    import numpy as np
    import pytest

    from fast_mlsirm.estimators.marginal import fit_gpcm_numpy

    valid = np.array([[0, 1], [1, 2]], dtype=np.int64)
    for bad in (np.array([0, 1, 2]), np.empty((0, 2)), np.array([[0.5, 1.0]])):
        with pytest.raises(ValueError):
            fit_gpcm_numpy(bad, 3, q_theta=7, max_iter=1, tol=1e-6)
    for kwargs in (
        {"n_cat": 3.5},
        {"n_cat": 3, "max_iter": 0},
        {"n_cat": 3, "tol": 0.0},
    ):
        with pytest.raises(ValueError):
            fit_gpcm_numpy(valid, **{"q_theta": 7, "max_iter": 1, "tol": 1e-6, **kwargs})


def test_poly_cell_and_fitter_rust_numpy_parity():
    """The Rust polytomous cell matches the NumPy reference bit-for-bit, and the
    Rust unidimensional GPCM fitter agrees with the NumPy mirror on recovery."""
    import numpy as np
    import pytest
    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "fit_poly_unidim"):  # pragma: no cover
        pytest.skip("core built without polytomous functions")
    from fast_mlsirm.estimators.marginal import compute_category_logprobs, fit_gpcm_numpy

    # cell parity: same softmax formula in both languages
    scores = np.array([0.0, 1.0, 2.0])
    intercepts = np.array([0.0, 0.3, -0.2])
    for base in (-1.3, 0.0, 0.75):
        rust = np.array(_core.gpcm_cell_logprobs(float(base), scores, intercepts))
        npy = compute_category_logprobs(np.array([base]), scores, intercepts)[0]
        assert np.allclose(rust, npy, atol=1e-12), f"cell parity at base={base}"

    # fitter agreement: same EM/Newton, same GH grid -> same MLE
    rng = np.random.default_rng(11)
    n_persons, n_items, k = 2500, 5, 3
    a_true = rng.uniform(0.8, 1.6, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.8, (n_items, k - 1))
    theta = rng.normal(0.0, 1.0, n_persons)
    y = np.zeros((n_persons, n_items), dtype=np.int64)
    for i in range(n_items):
        p = np.exp(compute_category_logprobs(a_true[i] * theta, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    rust_fit = _core.fit_poly_unidim(y.ravel(), n_persons, n_items, k, None, "gpcm", 21, 80, 1e-6)
    npy_fit = fit_gpcm_numpy(y, k, q_theta=21, max_iter=80, tol=1e-6)
    assert np.allclose(np.array(rust_fit["slope"]), npy_fit["a"], atol=0.05)
    assert np.isfinite(rust_fit["loglik"])


def test_fit_polytomous_api_recovers_and_validates():
    """The public fit_polytomous wrapper (Rust compute) recovers GRM parameters
    and rejects malformed input."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous
    from fast_mlsirm.estimators.marginal import _gh

    try:
        from fast_mlsirm import _core  # noqa: F401
        if not hasattr(__import__("fast_mlsirm")._core, "fit_poly_unidim"):
            pytest.skip("core built without polytomous functions")
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")

    from fast_mlsirm.polytomous import _core_module
    if _core_module() is None:  # pragma: no cover
        pytest.skip("compiled core not available")

    # GRM recovery via the public API
    rng = np.random.default_rng(3)
    n_persons, n_items, k = 3000, 5, 4
    a_true = rng.uniform(0.9, 1.6, n_items)
    thr_true = np.array([1.3, 0.0, -1.3])
    nodes, _ = _gh(21)
    theta = rng.normal(0.0, 1.0, n_persons)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        for pp in range(n_persons):
            eta = a_true[i] * theta[pp] + thr_true  # cumulative logits P(Y>=k)
            cum = 1.0 / (1.0 + np.exp(-eta))
            p = np.empty(k)
            p[0] = 1 - cum[0]
            p[1:k - 1] = cum[:k - 2] - cum[1:k - 1]
            p[k - 1] = cum[k - 2]
            y[pp, i] = rng.choice(k, p=p / p.sum())
    fit = fit_polytomous(y, k, model="grm", q_theta=21, max_iter=80, tol=1e-6)
    assert fit.model == "grm" and np.isfinite(fit.loglik)
    assert fit.converged
    assert fit.termination_reason == "tolerance"
    assert fit.n_iter < 80
    assert fit.loglik_trace.shape == (fit.n_iter + 1,)
    assert fit.loglik == fit.loglik_trace[-1]
    assert fit.final_delta >= -32 * np.finfo(float).eps * (
        1 + abs(fit.loglik_trace[-2])
    )
    assert fit.final_delta <= fit.stopping_tolerance
    assert np.corrcoef(a_true, fit.slope)[0, 1] > 0.9

    # validation
    with pytest.raises(ValueError):
        fit_polytomous(y, k, model="nominal", q_theta=21, max_iter=80, tol=1e-6)       # unsupported model
    with pytest.raises(ValueError):
        fit_polytomous(y.astype(float) + 0.5, k, model="grm", q_theta=21, max_iter=80, tol=1e-6)    # non-integer categories
    with pytest.raises(ValueError):
        fit_polytomous(y, 2, model="grm", q_theta=21, max_iter=80, tol=1e-6)                          # category out of range
    with pytest.raises(ValueError):
        fit_polytomous(y, k, model="grm", q_theta=21, max_iter=0, tol=1e-6)
    with pytest.raises(ValueError):
        fit_polytomous(y, k, model="grm", q_theta=21, max_iter=80, tol=np.nan)


def test_score_polytomous_recovers_theta():
    """fit_polytomous -> score_polytomous round-trip: EAP trait scores correlate
    with true theta (Rust compute end to end)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, score_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "score_poly_eap"):
        pytest.skip("compiled core without polytomous scoring")

    rng = np.random.default_rng(5)
    n_persons, n_items, k = 3000, 8, 3
    a_true = rng.uniform(0.9, 1.6, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.6, (n_items, k - 1))
    theta_true = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(compute_category_logprobs(a_true[i] * theta_true, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    sc = score_polytomous(y, fit, q_theta=21)
    assert sc["theta_eap"].shape == (n_persons,)
    assert np.all(sc["theta_sd"] > 0)
    assert np.corrcoef(theta_true, sc["theta_eap"])[0, 1] > 0.8


def test_score_polytomous_rejects_malformed_scoring_contract():
    """Scoring must not truncate quadrature controls or emit non-finite scores."""
    import numpy as np
    import pytest

    from fast_mlsirm import score_polytomous
    from fast_mlsirm.polytomous import PolytomousFit, _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "score_poly_eap"):
        pytest.skip("compiled core without polytomous scoring")

    fit = PolytomousFit(
        model="gpcm",
        slope=np.array([1.0]),
        cat_params=np.array([[0.2, -0.3]]),
        loglik=0.0,
        n_iter=0,
    )
    with pytest.raises(ValueError, match="q_theta must be an integer"):
        score_polytomous(np.array([[0.0]]), fit, q_theta=21.5)

    fit.slope[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_polytomous(np.array([[1.0]]), fit, q_theta=21)


def test_grm_cell_rust_numpy_parity():
    """The Rust GRM cumulative-logit cell matches the NumPy reference to 1e-12,
    and the NumPy GRM cell is a proper (normalized) log-distribution."""
    import numpy as np
    import pytest
    from fast_mlsirm.estimators.marginal import compute_grm_category_logprobs

    # NumPy self-consistency: normalization + binary reduction
    for base in (-1.0, 0.3, 1.7):
        lp = compute_grm_category_logprobs(np.array([base]), np.array([1.0, -1.0]))[0]
        assert abs(np.log(np.exp(lp).sum())) < 1e-12
    # binary GRM (K=2): P(Y=1) = sigmoid(base + beta)
    lp2 = compute_grm_category_logprobs(np.array([0.4]), np.array([0.2]))[0]
    assert abs(lp2[1] - (-np.logaddexp(0.0, -(0.4 + 0.2)))) < 1e-12

    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "grm_cell_logprobs"):  # pragma: no cover
        pytest.skip("core built without grm cell")
    thr = np.array([1.3, 0.1, -1.2])
    for base in (-1.4, 0.0, 0.9):
        rust = np.array(_core.grm_cell_logprobs(float(base), thr))
        npy = compute_grm_category_logprobs(np.array([base]), thr)[0]
        assert np.allclose(rust, npy, atol=1e-12), f"grm parity at base={base}"


def test_grm_cell_extreme_predictor_stays_finite():
    from fast_mlsirm.estimators.marginal import compute_grm_category_logprobs

    thresholds = np.array([1.0, 0.0])
    expected_middle = -1000.0 + np.log1p(-np.exp(-1.0))
    npy = compute_grm_category_logprobs(np.array([1000.0]), thresholds)[0]
    assert np.all(np.isfinite(npy)), npy
    np.testing.assert_allclose(npy[1], expected_middle, atol=1e-12)

    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    rust = np.asarray(_core.grm_cell_logprobs(1000.0, thresholds))
    assert np.all(np.isfinite(rust)), rust
    np.testing.assert_allclose(rust, npy, atol=1e-12)


def test_information_polytomous_api():
    """information_polytomous returns positive item/test information curves whose
    test info equals the item-info row sum (Rust compute)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, information_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_information_curves"):
        pytest.skip("compiled core without polytomous information")

    rng = np.random.default_rng(8)
    n_persons, n_items, k = 1500, 5, 3
    a_true = rng.uniform(1.0, 1.5, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.4, (n_items, k - 1))
    theta_p = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(compute_category_logprobs(a_true[i] * theta_p, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    grid = np.linspace(-3, 3, 25)
    info = information_polytomous(fit, grid)
    assert info["item_info"].shape == (25, n_items)
    assert np.all(info["item_info"] >= 0) and np.all(info["test_info"] > 0)
    assert np.allclose(info["test_info"], info["item_info"].sum(axis=1))
    # information is highest in the interior for well-centered items
    assert info["test_info"].argmax() not in (0, 24)


def test_fit_polytomous_handles_missing_data():
    """fit_polytomous marginalizes NaN (missing) responses and still recovers
    slopes; score_polytomous accepts partially-missing rows."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, score_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_poly_unidim"):
        pytest.skip("compiled core not available")

    rng = np.random.default_rng(21)
    n_persons, n_items, k = 5000, 6, 3
    a_true = rng.uniform(0.9, 1.6, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.6, (n_items, k - 1))
    theta = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k, dtype=float)
    y = np.full((n_persons, n_items), np.nan)
    for i in range(n_items):
        p = np.exp(compute_category_logprobs(a_true[i] * theta, scores, c_true[i]))
        for pp in range(n_persons):
            if rng.random() < 0.25:            # ~25% MCAR missing -> stays NaN
                continue
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    assert np.isfinite(fit.loglik)
    assert np.corrcoef(a_true, fit.slope)[0, 1] > 0.9
    sc = score_polytomous(y, fit, q_theta=21)
    assert sc["theta_eap"].shape == (n_persons,) and np.all(np.isfinite(sc["theta_eap"]))


def test_fit_lsirm_polytomous_recovers_positions():
    """The latent-space polytomous LSIRM (Rust compute) recovers item positions
    (distance-matrix RMSE) and slopes (RMSE), and returns person scores."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_lsirm_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_poly_lsirm"):
        pytest.skip("compiled core without polytomous LSIRM")

    rng = np.random.default_rng(4)
    n_persons, n_items, k, ld = 1000, 6, 3, 2
    # two separated item clusters
    zeta_true = np.zeros((n_items, ld))
    for i in range(n_items):
        zeta_true[i, 0] = (-1.2 if i < n_items // 2 else 1.2) + 0.3 * rng.standard_normal()
        zeta_true[i, 1] = 0.3 * rng.standard_normal()
    a_true = 1.0 + 0.1 * np.arange(n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = np.array([0.2, -0.2])
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for p in range(n_persons):
        theta = rng.standard_normal()
        xi = rng.standard_normal(ld)
        for i in range(n_items):
            base = a_true[i] * theta - np.sqrt(1e-8 + np.sum((xi - zeta_true[i]) ** 2))
            pr = np.exp(compute_category_logprobs(np.array([base]), scores, c_true[i])[0])
            y[p, i] = rng.choice(k, p=pr / pr.sum())

    fit = fit_lsirm_polytomous(y, k, latent_dim=ld, model="gpcm", q_theta=7, q_xi=7, max_iter=30, tol=1e-5)
    assert fit.zeta.shape == (n_items, ld)
    assert fit.theta_eap.shape == (n_persons,) and fit.xi_eap.shape == (n_persons, ld)
    assert np.all(fit.theta_sd > 0) and np.isfinite(fit.loglik)

    def dmat(z):
        return np.array([np.linalg.norm(z[i] - z[j]) for i in range(n_items) for j in range(i + 1, n_items)])

    def rmse(u, v):
        return float(np.sqrt(np.mean((u - v) ** 2)))

    assert rmse(a_true, fit.slope) < 0.3, "slope RMSE"
    assert rmse(dmat(zeta_true), dmat(fit.zeta)) < 0.7, "position distance-matrix RMSE"


def test_polytomous_information_criteria():
    """Kang-Cohen-Sung (2009) model-selection indices for a polytomous fit:
    correct free-parameter count and finite, ordered indices."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, polytomous_information_criteria
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_poly_unidim"):
        pytest.skip("compiled core not available")

    rng = np.random.default_rng(6)
    n_persons, n_items, k = 1500, 5, 3
    a = rng.uniform(0.9, 1.5, n_items)
    c = np.zeros((n_items, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (n_items, k - 1))
    theta = rng.standard_normal(n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(compute_category_logprobs(a[i] * theta, scores, c[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    ic = polytomous_information_criteria(fit, n_persons)
    # slope (n_items) + intercepts (n_items*(K-1)) = n_items*K
    assert ic["n_parameters"] == n_items * k
    for key in ("aic", "bic", "caic", "aicc", "sabic"):
        assert np.isfinite(ic[key])
    # BIC/CAIC penalize free parameters more heavily than AIC at N=1500
    assert ic["aic"] < ic["bic"] < ic["caic"]
    with pytest.raises(ValueError):
        polytomous_information_criteria(fit, 1)


def test_polytomous_information_criteria_marks_undefined_aicc():
    """AICc must not report a finite correction when N - K - 1 <= 0."""
    from types import SimpleNamespace

    import numpy as np

    from fast_mlsirm import polytomous_information_criteria

    fit = SimpleNamespace(
        slope=np.ones(2),
        cat_params=np.zeros((2, 2)),
        loglik=-10.0,
    )

    undefined = polytomous_information_criteria(fit, n_persons=5)
    assert undefined["n_parameters"] == 6
    assert np.isnan(undefined["aicc"])
    for key in ("aic", "bic", "caic", "sabic"):
        assert np.isfinite(undefined[key])

    defined = polytomous_information_criteria(fit, n_persons=10)
    expected = defined["aic"] + 2.0 * 6 * 7 / (10 - 6 - 1)
    assert defined["aicc"] == expected


def test_item_fit_polytomous_sx2():
    """Generalized S-X² polytomous item fit (Kang & Chen, 2008, 2011) through the
    public API: well-formed per-item output, calibration at the fitted model
    (statistic tracks df, few false flags), and input validation."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, item_fit_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_item_fit_sx2"
    ):
        pytest.skip("compiled core built without poly_item_fit_sx2")

    rng = np.random.default_rng(11)
    n_persons, n_items, k = 1500, 8, 4
    a = rng.uniform(0.9, 1.5, n_items)
    c = np.zeros((n_items, k))
    c[:, 1:] = rng.normal(0.0, 0.6, (n_items, k - 1))
    theta = rng.standard_normal(n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(compute_category_logprobs(a[i] * theta, scores, c[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    res = item_fit_polytomous(y, fit, q_theta=21)
    for key in ("statistic", "df", "p_value", "n_cells"):
        assert res[key].shape == (n_items,)
    assert np.all(np.isfinite(res["statistic"]))
    # df is the retained cell count minus m = n_cat item parameters
    assert np.array_equal(res["df"].astype(int), res["n_cells"] - k)
    finite_p = res["p_value"][np.isfinite(res["p_value"])]
    assert np.all((finite_p >= 0.0) & (finite_p <= 1.0))
    # a correctly fitted model is rarely flagged (contrast: G2 flags >30%)
    assert np.mean(finite_p < 0.05) < 0.30
    # statistic ~ df at the fitted parameters
    ratio = res["statistic"].sum() / res["df"].sum()
    assert 0.6 < ratio < 1.6, f"S-X2/df ratio off: {ratio}"

    # missing data (NaN) is marginalized: complete-case summed-score table
    y_miss = y.astype(float)
    y_miss[rng.random(y_miss.shape) < 0.05] = np.nan
    res_miss = item_fit_polytomous(y_miss, fit)
    assert np.all(np.isfinite(res_miss["statistic"]))

    # validation
    with pytest.raises(ValueError):
        item_fit_polytomous(y[:, :-1], fit)                 # wrong item count
    with pytest.raises(ValueError):
        item_fit_polytomous(y, fit, min_expected=0.0)       # non-positive floor


def test_m2_polytomous():
    """Polytomous M2 (Maydeu-Olivares & Joe, 2014) through the public API:
    correct moment/df bookkeeping, a good fit for correctly-specified data, and
    detection of a strongly misspecified (skewed-population) fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, m2_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_m2"):
        pytest.skip("compiled core built without poly_m2")

    def sim(theta, a, c, k):
        n, j = theta.size, a.size
        y = np.zeros((n, j), dtype=int)
        scores = np.arange(k, dtype=float)
        for i in range(j):
            p = np.exp(compute_category_logprobs(a[i] * theta, scores, c[i]))
            for pp in range(n):
                y[pp, i] = np.random.default_rng(1000 + i * n + pp).choice(k, p=p[pp])
        return y

    rng = np.random.default_rng(7)
    n, j, k = 2000, 6, 3
    a = rng.uniform(0.9, 1.5, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (j, k - 1))

    # correctly specified (normal ability) -> good fit
    theta = rng.standard_normal(n)
    y = sim(theta, a, c, k)
    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    res = m2_polytomous(y, fit, q_theta=21)
    # Q = j*(k-1) + C(j,2)*(k-1)^2 ; P = j*k ; df = Q - P
    q = j * (k - 1) + (j * (j - 1) // 2) * (k - 1) ** 2
    assert res["n_moments"] == q
    assert res["n_parameters"] == j * k
    assert res["df"] == q - j * k
    assert res["null_df"] == q - j * (k - 1)
    assert np.isfinite(res["m2"]) and res["m2"] >= 0.0
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["rmsea2_ci_lower"] <= res["rmsea2_ci_upper"] + 1e-9
    assert res["rmsea2"] < 0.05  # well-fitting
    assert res["null_m2"] > res["m2"]
    expected_cfi = np.clip(
        1.0 - (res["m2"] - res["df"]) / (res["null_m2"] - res["null_df"]),
        0.0,
        1.0,
    )
    expected_tli = (
        res["null_m2"] / res["null_df"] - res["m2"] / res["df"]
    ) / (res["null_m2"] / res["null_df"] - 1.0)
    np.testing.assert_allclose(res["cfi"], expected_cfi, atol=1e-12)
    np.testing.assert_allclose(res["tli"], expected_tli, atol=1e-12)

    # strongly misspecified: fit the wrong item parameters -> M2 must reject
    bad = fit
    bad.slope = a * 3.0  # inflate discriminations far from the truth
    res_bad = m2_polytomous(y, bad, q_theta=21)
    assert res_bad["m2"] > res["m2"]
    assert res_bad["p_value"] < 0.05
    assert res_bad["cfi"] < res["cfi"]
    assert res_bad["tli"] < res["tli"]

    # validation: fewer than 3 items has non-positive df
    with pytest.raises((ValueError, RuntimeError)):
        fit2 = fit_polytomous(y[:, :2], k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
        m2_polytomous(y[:, :2], fit2, q_theta=21)


def test_m2_polytomous_rejects_nonconverged_calibration():
    """Finite parameters are not sufficient evidence for M2 inference."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, m2_polytomous
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_m2"
    ):
        pytest.skip("compiled core built without poly_m2")

    rng = np.random.default_rng(20260716)
    y = rng.integers(0, 3, size=(240, 6))
    fit = fit_polytomous(
        y,
        3,
        model="gpcm",
        q_theta=11,
        max_iter=1,
        tol=1e-12,
    )

    assert fit.converged is False
    assert fit.termination_reason == "max_iter"
    assert fit.n_iter == 1
    assert fit.final_delta > fit.stopping_tolerance
    assert np.all(np.isfinite(fit.slope))
    assert np.all(np.isfinite(fit.cat_params))
    with pytest.raises(RuntimeError, match="requires a converged fit"):
        m2_polytomous(y, fit, q_theta=11)


@pytest.mark.parametrize(
    ("population", "message"),
    [
        ({"kind": "single", "pi_zero": 0.2}, "zero inflation"),
        ({"kind": "single", "delta": 0.4}, "item covariates"),
        (
            {"kind": "single", "pi_zero": 0.2, "delta": 0.4},
            "zero inflation and item covariates",
        ),
    ],
)
def test_m2_rejects_unsupported_calibration_terms(population, message):
    """M2 must not silently use base-model moments for structured fits."""
    from fast_mlsirm import fit_diagnostics
    from fast_mlsirm.types import MLSIRMParams

    n_persons, n_items = 40, 8
    responses = np.tile([0.0, 1.0], (n_persons, n_items // 2))
    params = MLSIRMParams(
        theta=np.zeros((n_persons, 1)),
        alpha=np.zeros(n_items),
        b=np.zeros(n_items),
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )

    with pytest.raises(ValueError, match=message):
        fit_diagnostics(
            responses,
            params,
            np.zeros(n_items, dtype=np.int64),
            model="MIRT",
            include_m2=True,
            estimator="mmle",
            convergence_status="converged",
            population=population,
        )


def test_local_dependence_polytomous():
    """Item-pair local dependence (Chen & Thissen, 1997) through the public API:
    correct per-pair bookkeeping, calibrated (few flags) for a locally
    independent fit, and detection of an injected testlet pair."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, local_dependence_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_local_dependence"
    ):
        pytest.skip("compiled core built without poly_local_dependence")

    def sim(theta, a, c, k, testlet=None):
        n, j = theta.size, a.size
        y = np.zeros((n, j), dtype=int)
        scores = np.arange(k, dtype=float)
        for i in range(j):
            base = a[i] * theta + (testlet if testlet is not None and i in (0, 1) else 0.0)
            p = np.exp(compute_category_logprobs(base, scores, c[i]))
            for pp in range(n):
                y[pp, i] = np.random.default_rng(300 + i * n + pp).choice(k, p=p[pp])
        return y

    rng = np.random.default_rng(4)
    n, j, k = 1500, 5, 3
    a = rng.uniform(0.9, 1.4, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (j, k - 1))

    # locally independent -> calibrated (the reference is conservative)
    theta = rng.standard_normal(n)
    y = sim(theta, a, c, k)
    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    ld = local_dependence_polytomous(y, fit)
    n_pairs = j * (j - 1) // 2
    for key in ("item_i", "item_j", "x2", "g2", "p_value", "cramers_v", "max_abs_std_resid"):
        assert ld[key].shape == (n_pairs,)
    assert ld["df"] == (k - 1) ** 2
    assert np.all(ld["item_i"] < ld["item_j"])
    assert np.all(np.isfinite(ld["x2"])) and np.all(ld["x2"] >= 0.0)
    finite_p = ld["p_value"][np.isfinite(ld["p_value"])]
    assert np.all((finite_p >= 0.0) & (finite_p <= 1.0))
    assert np.mean(finite_p < 0.05) < 0.35  # few flags under local independence

    # a strong shared testlet on items 0,1 -> that pair is strongly dependent
    dep = sim(theta, a, c, k, testlet=1.5 * rng.standard_normal(n))
    fit_d = fit_polytomous(dep, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    ld_d = local_dependence_polytomous(dep, fit_d)
    pair01 = next(idx for idx in range(n_pairs)
                  if (ld_d["item_i"][idx], ld_d["item_j"][idx]) == (0, 1))
    assert ld_d["x2"][pair01] > np.median(ld_d["x2"])
    assert ld_d["p_value"][pair01] < 0.05

    with pytest.raises(ValueError):
        local_dependence_polytomous(y[:, :-1], fit)


def test_fit_nominal_polytomous():
    """Nominal categories model (Thissen, Cai & Bock, 2010) through the public
    API: correct shapes, GPCM nesting (loglik >= GPCM, linear recovered scores),
    and input validation."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_nominal_polytomous, fit_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_nominal"):
        pytest.skip("compiled core built without fit_nominal")

    rng = np.random.default_rng(9)
    n, j, k = 2500, 5, 3
    a = rng.uniform(0.9, 1.5, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.4, (j, k - 1))
    theta = rng.standard_normal(n)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n, j), dtype=int)
    for i in range(j):
        p = np.exp(compute_category_logprobs(a[i] * theta, scores, c[i]))
        for pp in range(n):
            y[pp, i] = rng.choice(k, p=p[pp])

    nom = fit_nominal_polytomous(y, k, q_theta=21, max_iter=200, tol=1e-6)
    assert nom.scores.shape == (j, k - 1)
    assert nom.intercepts.shape == (j, k - 1)
    assert np.isfinite(nom.loglik)
    assert nom.converged
    assert nom.termination_reason == "tolerance"
    assert nom.loglik_trace.shape == (nom.n_iter + 1,)
    assert nom.loglik == nom.loglik_trace[-1]
    assert nom.final_delta <= nom.stopping_tolerance
    assert np.all(np.diff(nom.loglik_trace) >= -1e-10)

    # nests the GPCM: at least as high a loglik, and linear recovered scores
    gp = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    assert nom.loglik >= gp.loglik - 0.5
    ratio = nom.scores[:, 1] / nom.scores[:, 0]
    assert np.all(np.abs(ratio - 2.0) < 0.4)  # a_k ~ a*k

    with pytest.raises(ValueError):
        fit_nominal_polytomous(y, 1, q_theta=21, max_iter=200, tol=1e-6)
    with pytest.raises(ValueError):
        fit_nominal_polytomous(y.astype(float) + 0.5, k, q_theta=21, max_iter=200, tol=1e-6)  # non-integer categories

    unfinished = fit_nominal_polytomous(y[:50], k, q_theta=21, max_iter=1, tol=1e-12)
    assert not unfinished.converged
    assert unfinished.termination_reason == "max_iter"
    assert unfinished.n_iter == 1
    assert unfinished.loglik_trace.shape == (2,)
    assert unfinished.loglik == unfinished.loglik_trace[-1]
    assert np.isfinite(unfinished.final_delta)
    assert unfinished.final_delta > unfinished.stopping_tolerance

    malformed = (
        np.empty((0, j)),
        np.empty((n, 0)),
        np.column_stack((y[:, 0], np.full(n, np.nan))),
        np.where(np.arange(y.size).reshape(y.shape) == 0, np.inf, y),
    )
    for bad in malformed:
        with pytest.raises(ValueError):
            fit_nominal_polytomous(bad, k, q_theta=21, max_iter=200, tol=1e-6)
    for kwargs in ({"max_iter": 0}, {"tol": np.inf}, {"tol": -1.0}):
        with pytest.raises(ValueError):
            fit_nominal_polytomous(y, k, **{"q_theta": 21, "max_iter": 200, "tol": 1e-6, **kwargs})


def test_person_fit_polytomous():
    """Polytomous person fit l_z / l_z* (Drasgow-Levine-Williams, 1985; Snijders,
    2001) through the public API: calibrated on a clean fit and flagging
    inconsistent responders evaluated against that fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, person_fit_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_person_fit"):
        pytest.skip("compiled core built without poly_person_fit")

    rng = np.random.default_rng(5)
    n, j, k = 1500, 20, 3
    a = 1.0 + 0.03 * np.arange(j)
    c = np.zeros((j, k))
    c[:, 1] = 0.6
    c[:, 2] = -0.6
    scores = np.arange(k, dtype=float)

    def sim_person(th):
        return [
            rng.choice(k, p=np.exp(compute_category_logprobs(np.array([a[i] * th]), scores, c[i])[0]))
            for i in range(j)
        ]

    # clean sample -> fit -> calibrated person fit
    theta = rng.standard_normal(n)
    y = np.array([sim_person(theta[p]) for p in range(n)])
    fit = fit_polytomous(y, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    pf = person_fit_polytomous(y, fit)
    for key in ("lz", "lz_star", "theta_eap", "flagged"):
        assert len(pf[key]) == n
    finite = np.isfinite(pf["lz_star"])
    assert np.mean(pf["flagged"]) < 0.15  # Type I near nominal
    assert abs(np.mean(pf["lz_star"][finite])) < 0.4
    assert 0.8 < np.std(pf["lz_star"][finite]) < 1.25

    # inconsistent responders (implied trait alternates +-1.6 across items) —
    # evaluated at the SAME clean fit — are flagged
    m = 40
    ab = y.copy()
    for p in range(m):
        for i in range(j):
            ti = 1.6 if i % 2 == 0 else -1.6
            pr = np.exp(compute_category_logprobs(np.array([a[i] * ti]), scores, c[i])[0])
            ab[p, i] = rng.choice(k, p=pr)
    pf_ab = person_fit_polytomous(ab, fit)
    assert np.mean(pf_ab["flagged"][:m]) > 0.6

    with pytest.raises(ValueError):
        person_fit_polytomous(y[:, :-1], fit)


def test_cat_simulate_polytomous():
    """Polytomous CAT (Dodd, De Ayala & Koch, 1995) through the public API:
    recovers the trait efficiently and max-information beats random selection."""
    import numpy as np
    import pytest
    from fast_mlsirm import cat_simulate_polytomous, PolytomousFit
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_cat_simulate"):
        pytest.skip("compiled core built without poly_cat_simulate")

    j, k, z = 40, 4, 3
    slope = np.array([1.0 + 0.25 * (i % 3) for i in range(j)])
    cat = np.zeros((j, z))
    for i in range(j):
        b = -2.2 + 4.4 * i / (j - 1)
        cum = 0.0
        for m in range(z):
            cum += b + (m - (z - 1) / 2) * 0.9
            cat[i, m] = -slope[i] * cum
    fit = PolytomousFit(model="gpcm", slope=slope, cat_params=cat, loglik=0.0, n_iter=0)

    tt = np.random.default_rng(0).standard_normal(400)
    var = cat_simulate_polytomous(tt, fit, se_threshold=0.30, min_items=5, max_items=30, seed=1)
    for key in ("theta_eap", "theta_sd", "n_used"):
        assert var[key].shape == (tt.size,)
    rmse = np.sqrt(np.mean((var["theta_eap"] - tt) ** 2))
    assert rmse < 0.40
    assert var["n_used"].mean() < 0.75 * j  # far fewer than the bank
    assert np.all(var["n_used"] <= 30)

    adap = cat_simulate_polytomous(tt, fit, se_threshold=0.0, min_items=12, max_items=12, adaptive=True, seed=2)
    rand = cat_simulate_polytomous(tt, fit, se_threshold=0.0, min_items=12, max_items=12, adaptive=False, seed=3)
    r_a = np.sqrt(np.mean((adap["theta_eap"] - tt) ** 2))
    r_r = np.sqrt(np.mean((rand["theta_eap"] - tt) ** 2))
    assert r_a < r_r  # max-information more efficient than random

    with pytest.raises(ValueError):
        cat_simulate_polytomous(tt, fit, min_items=10, max_items=5)


def test_dif_polytomous():
    """Two-group IRT-LR DIF for polytomous items (Thissen, Steinberg & Wainer,
    1993) via the public API: correct bookkeeping, impact does not trigger DIF,
    and an injected uniform difficulty shift on one item is flagged while the
    anchor items stay clean."""
    import numpy as np
    import pytest
    from fast_mlsirm import dif_polytomous
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_dif"
    ):
        pytest.skip("compiled core built without poly_dif")

    k, j = 3, 8
    n_per = 900
    rng = np.random.default_rng(11)
    a = rng.uniform(0.9, 1.4, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (j, k - 1))
    scores = np.arange(k, dtype=float)

    def gen(dif_on_item0):
        # group 0: theta ~ N(0,1); group 1 (focal): theta ~ N(0.5, 1.2^2) (impact)
        th0 = rng.standard_normal(n_per)
        th1 = 0.5 + 1.2 * rng.standard_normal(n_per)
        theta = np.concatenate([th0, th1])
        gid = np.concatenate([np.zeros(n_per, int), np.ones(n_per, int)])
        y = np.zeros((2 * n_per, j), dtype=float)
        for p in range(2 * n_per):
            focal = gid[p] == 1
            for i in range(j):
                ci = c[i].copy()
                if i == 0 and focal and dif_on_item0:
                    # uniform DIF: shift difficulty => intercept_m += m * a * delta
                    d = 0.7
                    ci = ci + a[i] * d * scores
                base = a[i] * theta[p]
                pr = np.exp(compute_category_logprobs(base, scores, ci))
                y[p, i] = rng.choice(k, p=pr)
        return y, gid

    # impact but NO DIF: anchor items should be (mostly) clean, df = n_cat
    y0, gid0 = gen(dif_on_item0=False)
    res0 = dif_polytomous(y0, gid0, k, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)
    assert res0["item"].shape == (j,)
    for key in ("lr", "df", "p_value", "flagged_bh", "effect_size"):
        assert res0[key].shape == (j,)
    assert np.all(res0["df"] == k)  # (G-1)*K = K
    assert np.all(res0["lr"] >= 0.0)
    p0 = res0["p_value"]
    assert np.all((p0 >= 0.0) & (p0 <= 1.0))
    assert res0["flagged_bh"].sum() <= 1  # impact alone must not manufacture DIF

    # inject uniform DIF on item 0: it should be flagged with the largest effect
    y1, gid1 = gen(dif_on_item0=True)
    res1 = dif_polytomous(y1, gid1, k, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)
    assert res1["flagged_bh"][0]
    assert res1["p_value"][0] < 0.01
    assert res1["effect_size"][0] == res1["effect_size"].max()
    assert res1["flagged_bh"][1:].sum() <= 1  # anchors stay clean

    # studied_items subset restricts the sweep
    sub = dif_polytomous(
        y1, gid1, k, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05,
        studied_items=np.array([0, 3]),
    )
    assert list(sub["item"]) == [0, 3]

    # non-contiguous labels {0,2} must densify to 2 groups (df == n_cat), giving
    # the SAME result as contiguous {0,1} -- not an inflated, conservative df.
    gid_gap = np.where(gid1 == 1, 2, 0)
    res_gap = dif_polytomous(y1, gid_gap, k, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)
    assert np.all(res_gap["df"] == k)  # not (3-1)*k = 2k
    assert res_gap["flagged_bh"][0]  # strong DIF still detected despite label gap
    np.testing.assert_allclose(res_gap["lr"], res1["lr"], rtol=1e-6)

    with pytest.raises(ValueError):
        dif_polytomous(
            y1, gid1[:-5], k, model="gpcm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05,
        )  # group_id length mismatch


def test_mantel_haenszel_dif():
    """Observed-score Mantel-Haenszel DIF (Holland & Thayer, 1988) via the public API: a uniform
    (b-shift) DIF planted on one dichotomous item, no group impact. MH flags the planted item as
    practically large (ETS class B/C, BH-significant) with the delta sign matching the shift (harder for
    the focal group -> negative MH_D-DIF and STD-P-DIF), classifies the clean items as A, and returns the
    documented per-item arrays. Validation guards reject non-dichotomous responses and a single group."""
    import numpy as np
    import pytest
    from fast_mlsirm import mantel_haenszel_dif
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mantel_haenszel_dif"):
        pytest.skip("compiled core built without mantel_haenszel_dif")

    n, n_items = 3000, 12
    dif_item = 6
    rng = np.random.default_rng(1988)
    a = np.full(n_items, 1.2)
    b = -0.8 + 0.14 * np.arange(n_items)
    group = np.tile([0, 1], n // 2).astype(np.int64)
    theta = rng.standard_normal(n)  # equal ability distribution -> no impact
    bmat = np.tile(b, (n, 1))
    bmat[group == 1, dif_item] += 0.7  # item harder for the focal group (uniform DIF)
    p = 1.0 / (1.0 + np.exp(-(a * (theta[:, None] - bmat))))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = mantel_haenszel_dif(y, group)
    for key in ("item", "alpha_mh", "chi2_mh", "p_value", "mh_d_dif", "se_d_dif", "std_p_dif",
                "ets_class", "flagged_bh"):
        assert res[key].shape == (n_items,), key
    assert list(res["item"]) == list(range(n_items))
    # planted item: BH-flagged, large negative delta, negative std-p-dif, class B or C
    assert res["flagged_bh"][dif_item]
    assert res["mh_d_dif"][dif_item] < -0.8
    assert res["std_p_dif"][dif_item] < 0.0
    assert res["ets_class"][dif_item] in ("B", "C")
    # clean items: negligible (class A) by the practical-significance classification
    clean = [i for i in range(n_items) if i != dif_item]
    assert all(res["ets_class"][i] == "A" for i in clean)
    assert np.all(np.abs(res["mh_d_dif"][clean]) < 1.0)

    # validation: non-dichotomous responses and a single-group sample are rejected
    ybad = y.copy()
    ybad[0, 0] = 2
    with pytest.raises(ValueError):
        mantel_haenszel_dif(ybad, group)
    with pytest.raises(ValueError):
        mantel_haenszel_dif(y, np.zeros(n, dtype=np.int64))


def test_logistic_dif_zumbo():
    """Zumbo (1999) logistic-regression DIF via the public API. Its reason to exist is NON-UNIFORM DIF:
    a crossing item whose ICCs intersect at the common group ability mean is flagged through the
    score x group interaction, while Mantel-Haenszel — a stratified odds-ratio test — calls the very
    same item negligible. A plain b-shift item shows the reverse pattern (uniform component fires, the
    interaction does not), and the exact LR decomposition holds for every item."""
    import numpy as np
    import pytest
    from fast_mlsirm import logistic_dif, mantel_haenszel_dif
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "logistic_dif"):
        pytest.skip("compiled core built without logistic_dif")

    rng = np.random.default_rng(1999)
    n, n_items = 4000, 10
    cross_item, unif_item = 4, 7
    b = -0.9 + 0.2 * np.arange(n_items)
    group = (np.arange(n) % 2).astype(np.int64)
    theta = rng.standard_normal(n)  # identical ability distribution in both groups

    a_mat = np.ones((n, n_items))
    b_mat = np.tile(b, (n, 1))
    focal = group == 1
    # crossing DIF centered at the common ability mean: same difficulty, different slope
    a_mat[:, cross_item] = np.where(focal, 0.4, 1.7)
    b_mat[:, cross_item] = 0.0
    # pure uniform DIF
    b_mat[focal, unif_item] += 0.8
    p = 1.0 / (1.0 + np.exp(-(a_mat * (theta[:, None] - b_mat))))
    y = (rng.random((n, n_items)) < p).astype(float)

    lr = logistic_dif(y, group)
    mh = mantel_haenszel_dif(y, group)
    for key in ("item", "chi2_uniform", "p_uniform", "chi2_nonuniform", "p_nonuniform",
                "chi2_total", "p_total", "delta_r2", "delta_r2_uniform", "jg_class",
                "flagged_bh", "converged"):
        assert lr[key].shape == (n_items,), key

    # crossing item: the interaction fires, the group main effect does not, MH says negligible
    assert lr["converged"][cross_item]
    assert lr["p_nonuniform"][cross_item] < 0.01
    assert lr["p_uniform"][cross_item] > 0.05
    assert lr["flagged_bh"][cross_item]
    assert mh["ets_class"][cross_item] == "A", "MH unexpectedly flagged the crossing item"

    # uniform item: the reverse pattern, and MH does see it
    assert lr["p_uniform"][unif_item] < 0.01
    assert lr["p_nonuniform"][unif_item] > 0.05
    assert mh["ets_class"][unif_item] != "A"

    # exact nesting decomposition everywhere; clean items negligible
    np.testing.assert_allclose(
        lr["chi2_uniform"] + lr["chi2_nonuniform"], lr["chi2_total"], atol=1e-6
    )
    # #1880: jg_class is "not applicable" (U) for every item now, clean or not -- the bands were
    # calibrated on a different, underdetermined statistic (see logistic_dif's docstring).
    assert all(lr["jg_class"][i] == "U" for i in range(n_items))

    # validation
    with pytest.raises(ValueError):
        bad = y.copy()
        bad[0, 0] = 2
        logistic_dif(bad, group)
    with pytest.raises(ValueError):
        logistic_dif(y, np.zeros(n, dtype=np.int64) + 3)


def test_score_wle_poly():
    """Warm (1989) WLE for polytomous items via the public API.
    Pins the two properties that distinguish it from the EAP already shipped: it is NOT shrunk toward
    a prior mean, and it stays finite for the all-lowest/all-highest patterns where ML diverges.
    Also pins the K=2 reduction to the dichotomous score_wle, which is the plumbing anchor."""
    import numpy as np
    import pytest
    from fast_mlsirm import score_wle, score_wle_poly
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "score_wle_poly"):
        pytest.skip("compiled core built without score_wle_poly")

    slope = np.array([1.7, 0.9, 1.3, 1.1, 0.8])
    cat = np.array([[1.6, 0.1, -1.2], [2.4, -0.5, -1.9], [0.8, -0.2, -2.6],
                    [1.2, 0.3, -1.5], [2.0, -0.7, -2.1]])
    y = np.array([[1, 3, 2, 0, 2], [0, 0, 0, 0, 0], [3, 3, 3, 3, 3]], float)

    for model in ("grm", "gpcm"):
        r = score_wle_poly(y, slope, cat, 4, model=model)
        for key in ("theta", "se", "boundary"):
            assert key in r
        assert r["theta"].shape == (3,)
        assert np.all(np.isfinite(r["theta"])), f"{model}: WLE must be finite for every pattern"
        assert not r["boundary"].any(), f"{model}: no pattern should hit the bound here"
        # the extreme patterns are where ML diverges; WLE must be finite AND ordered
        assert r["theta"][1] < r["theta"][0] < r["theta"][2]
        assert np.all(r["se"] > 0)

    # K=2 must reproduce the dichotomous estimator exactly (both families ARE the 2PL there)
    a = np.array([1.4, 0.8, 2.1, 1.1, 0.6])
    b = np.array([-0.9, 0.3, 1.4, -1.7, 0.8])
    ybin = np.array([[1, 1, 0, 1, 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 0]], float)
    dich = score_wle(a, b, ybin, theta_bound=6.0, tol=1e-10)
    cat2 = (-a * b).reshape(-1, 1)
    for model in ("gpcm", "grm"):
        poly = score_wle_poly(ybin, a, cat2, 2, model=model, theta_bound=6.0, tol=1e-10)
        np.testing.assert_allclose(poly["theta"], dich["theta"], atol=1e-6)
        np.testing.assert_allclose(poly["se"], dich["se"], atol=1e-6)

    # missing data: a person with nothing observed is undefined, not average
    ymiss = y.copy()
    ymiss[1, :] = np.nan
    r = score_wle_poly(ymiss, slope, cat, 4, model="grm")
    assert np.isnan(r["theta"][1]) and r["boundary"][1]
    assert np.isfinite(r["theta"][0])

    # validation
    with pytest.raises(ValueError):
        score_wle_poly(y, slope, cat, 1, model="grm")
    with pytest.raises(ValueError):
        score_wle_poly(np.full_like(y, 9.0), slope, cat, 4, model="grm")
    with pytest.raises(ValueError):
        score_wle_poly(y, slope, cat[:, :2], 4, model="grm")


def test_sibtest_uniform():
    """Uniform SIBTEST (Shealy & Stout, 1993) through the public API.
    Pins the parts that are actually true: the sign is OPPOSITE to Mantel-Haenszel's, the per-group
    reliabilities are exposed, planted DIF is detected, and a degenerate item yields NaN rather than a
    plausible number. Deliberately does NOT assert that SIBTEST beats MH under impact -- measurement
    says the reverse (it over-rejects), and that is documented in the docstring."""
    import numpy as np
    import pytest
    from fast_mlsirm import mantel_haenszel_dif, sibtest
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "sibtest"):
        pytest.skip("compiled core built without sibtest")

    rng = np.random.default_rng(23)
    n, J = 3000, 8
    group = (np.arange(n) % 2).astype(np.int64)
    theta = rng.standard_normal(n)
    b = np.tile(-1.1 + 0.23 * np.arange(J), (n, 1))
    b[group == 1, 4] += 1.1  # item 4 harder for the focal group
    p = 1.0 / (1.0 + np.exp(-(1.3 * (theta[:, None] - b))))
    y = (rng.random((n, J)) < p).astype(float)

    sib = sibtest(y, group, fdr_q=0.05, j_min=5)
    mh = mantel_haenszel_dif(y, group)
    for key in ("beta_uni", "se_beta", "b_uni", "p_value", "alpha_ref", "alpha_focal",
                "n_strata_used", "flagged_bh"):
        assert key in sib, key
    assert sib["beta_uni"].shape == (J,)

    # planted item detected, and the sign conventions are OPPOSITE across the two procedures
    assert sib["p_value"][4] < 0.01
    assert sib["beta_uni"][4] > 0.0, "positive beta_uni = harder for the focal group"
    assert mh["mh_d_dif"][4] < 0.0, "mh_d_dif is focal-oriented and goes the other way"
    assert sib["beta_uni"][4] * mh["std_p_dif"][4] < 0.0

    # swapping the group labels flips beta_uni and preserves its magnitude
    sib_sw = sibtest(y, 1 - group, fdr_q=0.05, j_min=5)
    assert sib_sw["beta_uni"][4] < 0.0
    assert abs(sib_sw["beta_uni"][4] + sib["beta_uni"][4]) < 1e-9

    # reliabilities are real, per-group, and in range
    assert np.all((sib["alpha_ref"] > 0.0) & (sib["alpha_ref"] < 1.0))
    assert np.all((sib["alpha_focal"] > 0.0) & (sib["alpha_focal"] < 1.0))
    assert not np.array_equal(sib["alpha_ref"], sib["alpha_focal"]), "alphas must be per-group"
    assert np.all(sib["n_strata_used"][np.isfinite(sib["beta_uni"])] > 0)

    # the binding's field mapping is pinned by internal identities, so a swapped pair of dict keys
    # (se_beta/b_uni, or alpha_ref/alpha_focal) cannot slip through the boundary unnoticed
    ok = np.isfinite(sib["b_uni"])
    assert np.array_equal(sib["item"], np.arange(J))
    assert np.allclose(sib["b_uni"][ok], sib["beta_uni"][ok] / sib["se_beta"][ok], rtol=1e-12)
    from fast_mlsirm import chi2_sf
    expected_p = np.array([chi2_sf(v * v, 1.0) for v in sib["b_uni"][ok]])
    assert np.allclose(sib["p_value"][ok], expected_p, rtol=1e-10)

    # j_min is plumbed through: a stricter floor can only retain fewer strata, and at a floor high
    # enough to bite on this fixture it must retain strictly fewer (60 is too low to constrain a
    # 3000-person bank, so asserting on it would prove nothing)
    strict = sibtest(y, group, fdr_q=0.05, j_min=250)
    assert np.all(strict["n_strata_used"] <= sib["n_strata_used"])
    assert strict["n_strata_used"].sum() < sib["n_strata_used"].sum()

    # a constant item is undefined, not "no DIF"
    y2 = y.copy()
    y2[:, 7] = 1.0
    deg = sibtest(y2, group, fdr_q=0.05, j_min=5)
    assert np.isnan(deg["beta_uni"][7]) and np.isnan(deg["p_value"][7])
    assert not deg["flagged_bh"][7]

    # validation
    with pytest.raises(ValueError):
        sibtest(y[:, :2], group, fdr_q=0.05, j_min=5)
    with pytest.raises(ValueError):
        sibtest(y, group, fdr_q=0.05, j_min=1)


def test_dif_purification():
    """Iterative item purification (Candell & Drasgow, 1988; Clauser et al., 1993) via the public API.
    Seeded regression fixture: several items are shifted UNIDIRECTIONALLY against the focal group, which
    depresses that group's number-correct total, so CLEAN items pick up spurious DIF. Rebuilding the
    criterion from the unflagged anchor reduces those false flags while the planted items stay flagged.
    The precondition is asserted first so the test cannot pass on a fixture with no contamination."""
    import numpy as np
    import pytest
    from fast_mlsirm import (
    velicer_map,
    velicer_map_from_data,logistic_dif_purified, mantel_haenszel_dif,
                             mantel_haenszel_dif_purified)
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mantel_haenszel_dif_purified"):
        pytest.skip("compiled core built without mantel_haenszel_dif_purified")

    def bank(dif_items, shift, seed, n=3000, n_items=12):
        rng = np.random.default_rng(seed)
        b = -0.9 + 0.16 * np.arange(n_items)
        group = (np.arange(n) % 2).astype(np.int64)
        theta = rng.standard_normal(n)
        bmat = np.tile(b, (n, 1))
        for i in dif_items:
            bmat[group == 1, i] += shift  # all against the SAME group
        p = 1.0 / (1.0 + np.exp(-(1.2 * (theta[:, None] - bmat))))
        return (rng.random((n, n_items)) < p).astype(float), group

    dif_items = [2, 5, 8]
    y, group = bank(dif_items, 1.2, 41)
    plain = mantel_haenszel_dif(y, group)
    pur = mantel_haenszel_dif_purified(y, group)
    for key in (
        "anchor",
        "n_anchor",
        "rounds",
        "purify_converged",
        "purify_termination_reason",
    ):
        assert key in pur
    assert pur["anchor"].shape == (12,)

    clean = [i for i in range(12) if i not in dif_items]
    before = [j for j in clean if plain["ets_class"][j] != "A"]
    assert before, "fixture precondition: the unpurified sweep false-flagged no clean item"
    after = [j for j in clean if pur["ets_class"][j] != "A"]
    assert len(after) < len(before), f"purification did not reduce false flags: {before} -> {after}"
    # true positives retained, and the planted items are what left the anchor
    for d in dif_items:
        assert pur["ets_class"][d] in ("B", "C")
        assert not pur["anchor"][d]
    # the criterion genuinely changed
    assert any(abs(pur["chi2_mh"][j] - plain["chi2_mh"][j]) > 1e-6 for j in clean)

    # a clean bank needs no purification and reproduces the plain sweep exactly
    y0, g0 = bank([], 0.0, 7)
    p0 = mantel_haenszel_dif(y0, g0)
    q0 = mantel_haenszel_dif_purified(y0, g0)
    assert q0["rounds"] == 0 and q0["purify_converged"] and q0["n_anchor"] == 12
    assert q0["purify_termination_reason"] == "stable_flag_set"
    np.testing.assert_array_equal(q0["chi2_mh"], p0["chi2_mh"])

    # FIXED (#1941): the logistic variant's purification criterion switched from the unreachable
    # jg_class (retired to "U" for every item by #1880) to flagged_bh, so the anchor now actually
    # shrinks around the planted items instead of staying stuck at the full test. A smaller shift
    # than the Mantel-Haenszel fixture above: the 2-df omnibus chi2_total logistic test is powerful
    # enough at shift=1.2, n=3000 to flag every item (universal criterion contamination, not just
    # the planted ones), which would make "planted items leave, clean items stay" unfalsifiable.
    y_logit, group_logit = bank(dif_items, 0.6, 41)
    lp = logistic_dif_purified(y_logit, group_logit)
    assert all(not lp["anchor"][d] for d in dif_items), "planted items must leave the anchor"
    assert lp["n_anchor"] == 12 - len(dif_items) and lp["rounds"] >= 1
    # the loop's scalar flag must NOT collide with logistic_dif's PER-ITEM convergence array: the two
    # answer different questions, and a same-named scalar silently destroyed the array at the boundary
    assert np.asarray(lp["converged"], dtype=bool).shape == (12,)
    assert isinstance(lp["purify_converged"], bool)

    # the interpretation limits must survive to __doc__ -- a docstring written as `"""..""" + NOTE`
    # is a BinOp, not a constant, so the compiler discards it and help() shows nothing
    for fn in (mantel_haenszel_dif_purified, logistic_dif_purified):
        assert fn.__doc__ and "FDR" in fn.__doc__, f"{fn.__name__} lost its caveat"


def test_score_wle_warm():
    """Warm's WLE (1989) via the public API: FINITE estimates for the perfect/zero patterns where the
    MLE diverges (correct > incorrect), monotone in the raw score, SE = 1/sqrt(I), 3PL support, and
    input validation."""
    import numpy as np
    import pytest
    from fast_mlsirm import score_wle
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "score_wle"):
        pytest.skip("compiled core built without score_wle")

    j = 8
    a = 1.0 + 0.1 * np.arange(j)
    b = np.linspace(-2.0, 2.0, j)
    # perfect and zero patterns -> finite, interior, correct > incorrect (MLE would be +/-inf)
    rc = score_wle(a, b, np.ones((1, j)))
    rw = score_wle(a, b, np.zeros((1, j)))
    assert np.isfinite(rc["theta"][0]) and not rc["boundary"][0]
    assert np.isfinite(rw["theta"][0]) and not rw["boundary"][0]
    assert rc["theta"][0] > rw["theta"][0]
    # monotone in the number-correct score (Rasch, a=1)
    a1 = np.ones(j)
    thetas = [
        score_wle(a1, b, np.array([[1.0 if i < k else 0.0 for i in range(j)]]))["theta"][0]
        for k in range(j + 1)
    ]
    assert all(thetas[k] <= thetas[k + 1] + 1e-9 for k in range(j))
    # SE = 1/sqrt(I) at the estimate (2PL information)
    y = np.array([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
    res = score_wle(a, b, y)
    assert res["theta"].shape == (1,) and res["se"].shape == (1,) and res["boundary"].shape == (1,)
    theta = res["theta"][0]
    p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    info = np.sum(a**2 * p * (1.0 - p))
    assert abs(res["se"][0] - 1.0 / np.sqrt(info)) < 1e-6
    # 3PL support (lower asymptote c > 0)
    r3 = score_wle(a, b, y, c=np.full(j, 0.2))
    assert np.isfinite(r3["theta"][0])
    # validation: a/b length mismatch and a non-0/1 response
    with pytest.raises(ValueError):
        score_wle(a, b[:-1], y)
    with pytest.raises(ValueError):
        score_wle(a, b, np.array([[2.0] + [0.0] * (j - 1)]))


def test_score_wle_resolves_narrow_global_mode():
    """The Rust WLE search resolves a narrow high-discrimination 4PL global mode that falls between
    the historical fixed-grid nodes."""
    import numpy as np

    from fast_mlsirm import score_wle

    a = np.array(
        [
            3.329447657883643,
            0.27232757528116147,
            84.38646237902715,
            4.507142332708399,
            0.216076032654272,
            1.152868526694496,
            0.5026701543207452,
            3.594020470848568,
        ]
    )
    b = np.array(
        [
            -2.2559085720992726,
            4.784793518100594,
            -2.7313173853279284,
            3.16639784715872,
            2.45483432935667,
            3.577399138394002,
            -0.541499889021253,
            -3.1606254220709538,
        ]
    )
    c = np.array(
        [
            0.4293154638946107,
            0.03968316086976924,
            0.2117187277379179,
            0.4041453105751009,
            0.14842532496042327,
            0.2781240730868334,
            0.07100800469041686,
            0.16882942315223948,
        ]
    )
    d = np.array(
        [
            0.9271440266982822,
            0.8326920519773708,
            0.7052699247299387,
            0.7321429393598535,
            0.7250331916969143,
            0.8800003001396377,
            0.7964931220169523,
            0.8078636510671307,
        ]
    )
    y = np.array([[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]])

    result = score_wle(a, b, y, c=c, d=d, tol=1e-10)
    assert not result["boundary"][0]
    assert abs(result["theta"][0] + 2.74) < 0.02


def test_rasch_cml_and_andersen_lr():
    """Rasch CML (Andersen, 1970/1972) and Andersen's (1973) LR test via the public API: the item
    difficulties are recovered PERSON-DISTRIBUTION-FREE (the same beta under N(0,1) vs skewed ability,
    the CML signature), and the LR test does not over-reject true Rasch data but flags a planted
    group-specific difficulty shift."""
    import numpy as np
    import pytest
    from fast_mlsirm import andersen_lr_test, fit_rasch_cml
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rasch_cml"):
        pytest.skip("compiled core built without fit_rasch_cml")

    rng = np.random.default_rng(1972)
    k, n = 8, 4000
    beta = np.array([-1.6, -0.9, -0.3, 0.2, 0.7, 1.2, 1.7, 0.0])
    beta -= beta.mean()

    def sim(theta):
        p = 1.0 / (1.0 + np.exp(-(theta[:, None] - beta)))
        return (rng.random((n, k)) < p).astype(float)

    # (a) N(0,1) ability and (b) a strongly right-skewed ability
    ya = sim(rng.standard_normal(n))
    yb = sim(1.5 * rng.exponential(size=n) - 1.5)
    fa = fit_rasch_cml(ya, max_iter=100, tol=1e-8)
    fb = fit_rasch_cml(yb, max_iter=100, tol=1e-8)
    assert fa["converged"] and fb["converged"]
    assert fa["beta"].shape == (k,) and fa["se"].shape == (k,)
    assert abs(fa["beta"].sum()) < 1e-8  # sum-zero identification
    assert np.sqrt(np.mean((fa["beta"] - beta) ** 2)) < 0.15
    # the defining property: the two ability distributions give the SAME estimates
    assert np.sqrt(np.mean((fa["beta"] - fb["beta"]) ** 2)) < 0.15
    assert np.all(np.isfinite(fa["se"])) and np.all(fa["se"] > 0)

    # Andersen LR: arbitrary split of true Rasch data -> not rejected; planted DIF -> rejected
    theta = rng.standard_normal(n)
    group = (np.arange(n) % 2)
    y_rasch = sim(theta)
    t1 = andersen_lr_test(y_rasch, group, max_iter=100, tol=1e-8)
    assert t1["df"] == (2 - 1) * (k - 1)
    assert t1["p_value"] > 0.01
    y_dif = y_rasch.copy()
    # regenerate group-1 responses on item 0 under a shifted difficulty
    b0 = beta.copy()
    b0[0] += 1.5
    mask = group == 1
    p_dif = 1.0 / (1.0 + np.exp(-(theta[mask][:, None] - b0)))
    y_dif[mask] = (rng.random((mask.sum(), k)) < p_dif).astype(float)
    t2 = andersen_lr_test(y_dif, group, max_iter=100, tol=1e-8)
    assert t2["lr"] > t1["lr"] + 15.0 and t2["p_value"] < 0.01

    # validation: non-0/1 responses rejected
    with pytest.raises(ValueError):
        bad = ya.copy()
        bad[0, 0] = 2
        fit_rasch_cml(bad, max_iter=100, tol=1e-8)


def test_andersen_group_labels_must_be_finite(monkeypatch):
    """Non-finite labels must fail before NumPy's float-to-int cast can densify them."""
    import numpy as np
    import pytest

    from fast_mlsirm import andersen_lr_test

    class UnexpectedCore:
        def andersen_lr_test(self, *_args):  # pragma: no cover - must not be reached
            pytest.fail("invalid group labels reached the Rust core")

    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: UnexpectedCore())
    responses = np.array([[1.0, 0.0], [0.0, 1.0]])
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite non-negative integers"):
            andersen_lr_test(responses, np.array([0.0, invalid]), max_iter=100, tol=1e-8)


def test_dif_polytomous_grm_no_silent_false_negative():
    """A GRM studied item whose focal group never uses a middle category can
    disorder thresholds -> NaN loglik. The finiteness guard must surface that as
    NaN (unflagged) rather than let the `.max(0.0)` clamp report a strongly-DIF
    item as clean (lr=0, p=1)."""
    import numpy as np
    import pytest
    from fast_mlsirm import dif_polytomous
    from fast_mlsirm.estimators.marginal import compute_grm_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_dif"):
        pytest.skip("compiled core built without poly_dif")

    k, j, n_per = 4, 6, 500
    rng = np.random.default_rng(7)
    a = rng.uniform(1.0, 1.4, j)
    # decreasing GRM thresholds (ordered)
    b = np.tile(np.array([1.2, 0.0, -1.2]), (j, 1))

    def draw(theta, ai, bi):
        p = np.exp(compute_grm_category_logprobs(ai * theta, bi))
        return rng.choice(k, p=p)

    y = np.zeros((2 * n_per, j))
    gid = np.concatenate([np.zeros(n_per, int), np.ones(n_per, int)])
    theta = np.concatenate([rng.standard_normal(n_per), 0.6 + rng.standard_normal(n_per)])
    for p in range(2 * n_per):
        focal = gid[p] == 1
        for i in range(j):
            if i == 0 and focal:
                # strong DIF + squeeze a middle category out for the focal group
                # (category 1 ~ 0 counts) so the GRM per-group M-step disorders
                # thresholds and the fit goes non-finite -- the guard's trigger
                y[p, i] = draw(theta[p], a[i], np.array([4.2, 4.1, -4.2]))
            else:
                y[p, i] = draw(theta[p], a[i], b[i])

    res = dif_polytomous(y, gid, k, model="grm", q_theta=21, max_iter=200, tol=1e-5, fdr_q=0.05)
    # item 0 must NOT be a silent clean report: either surfaced NaN, or (if the
    # fit stayed finite) correctly flagged. A finite p>0.5 unflagged would be the
    # masked false-negative the guard exists to prevent.
    p0, flg0 = res["p_value"][0], res["flagged_bh"][0]
    assert np.isnan(p0) or flg0, f"item 0 silently reported clean: p={p0}, flagged={flg0}"
    # a NaN p-value must be reported unflagged (never counted as significant)
    assert not (np.isnan(p0) and flg0)


def test_u3_person_fit_polytomous():
    """Nonparametric polytomous U3poly (Emons, 2008) through the public API:
    correct bookkeeping, calibrated flag rate under a simulated cutoff, and
    detection of careless responders."""
    import numpy as np
    import pytest
    from fast_mlsirm import (
    velicer_map,
    velicer_map_from_data,
        fit_polytomous,
        u3_cutoff_polytomous,
        u3_person_fit_polytomous,
    )
    from fast_mlsirm.estimators.marginal import compute_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "u3_person_fit"):
        pytest.skip("compiled core built without u3_person_fit")

    k, j, n = 5, 20, 800
    rng = np.random.default_rng(3)
    a = rng.uniform(0.9, 1.5, j)
    # spread item difficulty across the bank so items differ in popularity -- the
    # regime where a popularity-based statistic like U3 has power (Emons, 2008)
    bdiff = np.linspace(-1.6, 1.6, j)
    c = np.zeros((j, k))
    for i in range(j):
        cum = 0.0
        for m in range(1, k):
            cum += bdiff[i] + (m - 1 - (k - 2) / 2) * 0.8
            c[i, m] = -a[i] * cum
    scores = np.arange(k, dtype=float)

    def gen(n_care):
        theta = rng.standard_normal(n)
        y = np.zeros((n, j))
        for p in range(n):
            for i in range(j):
                if p < n_care:  # careless: uniform-random category
                    y[p, i] = rng.integers(k)
                else:
                    pr = np.exp(compute_category_logprobs(a[i] * theta[p], scores, c[i]))
                    y[p, i] = rng.choice(k, p=pr)
        return y

    # clean data -> fit a bank -> simulated cutoff -> near-nominal flag rate
    y0 = gen(0)
    res0 = u3_person_fit_polytomous(y0, k)
    for key in ("u3poly", "total_score", "flagged"):
        assert res0[key].shape == (n,)
    finite = res0["u3poly"][np.isfinite(res0["u3poly"])]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    assert not res0["flagged"].any()  # no cutoff -> nothing flagged

    fit = fit_polytomous(y0, k, model="gpcm", q_theta=21, max_iter=80, tol=1e-6)
    cutoff = u3_cutoff_polytomous(fit, n_persons=n, alpha=0.05, n_rep=60, seed=7)
    assert 0.0 < cutoff < 1.0
    flagged0 = u3_person_fit_polytomous(y0, k, cutoff=cutoff)["flagged"]
    assert flagged0.mean() < 0.15  # calibrated-ish on clean data

    # inject careless responders -> they are flagged far more than clean persons
    n_care = 120
    y1 = gen(n_care)
    res1 = u3_person_fit_polytomous(y1, k, cutoff=cutoff)
    care_rate = res1["flagged"][:n_care].mean()
    clean_rate = res1["flagged"][n_care:].mean()
    assert care_rate > 0.6, f"careless detection too weak: {care_rate}"
    assert care_rate > 3 * clean_rate

    # K=2 stays in range and total_score bookkeeping is correct
    yb = (rng.random((200, 10)) < 0.5).astype(float)
    rb = u3_person_fit_polytomous(yb, 2)
    assert np.array_equal(rb["total_score"], yb.sum(axis=1).astype(np.int64))

    # an all-missing respondent has no conditioning group -> undefined (NaN),
    # never a silent perfect fit that can't be flagged
    ymiss = y0.copy()
    ymiss[0, :] = np.nan
    rm = u3_person_fit_polytomous(ymiss, k, cutoff=cutoff)
    assert np.isnan(rm["u3poly"][0])
    assert not rm["flagged"][0]

    with pytest.raises(ValueError):
        u3_person_fit_polytomous(y0, k, cutoff=float("nan"))


def test_equate_observed_scores_and_neat():
    """Observed-score equating (Kolen & Brennan, 2014) through the public API:
    equipercentile self-equate is the identity, mean/linear recover a known
    transform, and NEAT chained/FE collapse to EG equipercentile under equal
    anchor distributions."""
    import numpy as np
    import pytest
    from fast_mlsirm import equate_neat, equate_observed_scores
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_observed_scores"):
        pytest.skip("compiled core built without equating")

    rng = np.random.default_rng(5)
    k = 40
    x = np.clip(np.round(20 + 8 * rng.standard_normal(4000)), 0, k)

    # self-equate is the identity on populated scores
    r = equate_observed_scores(x, x, method="equipercentile", k_x=k, k_y=k)
    g = np.bincount(x.astype(int), minlength=k + 1)
    dev = np.abs(r.y_equivalents - r.x_scores)[g > 0]
    assert dev.max() < 1e-9
    assert r.design == "EG"

    # mean equating recovers Y = X + 5
    y = x + 5
    rm = equate_observed_scores(x, y, method="mean", k_x=k, k_y=k + 5)
    assert abs(rm.intercept - 5.0) < 1e-9 and abs(rm.slope - 1.0) < 1e-12

    # NEAT collapse under equal anchor distributions (identical anchor vectors)
    n = 6000
    kv, kx, ky = 15, 30, 40
    v = np.clip(np.round(7 + 3 * rng.standard_normal(n)), 0, kv)
    xt = np.clip(np.round(1.4 * v + 4 + 4 * rng.standard_normal(n)), 0, kx)
    yt = np.clip(np.round(2.0 * v + 6 + 5 * rng.standard_normal(n)), 0, ky)
    eg = equate_observed_scores(xt, yt, method="equipercentile", k_x=kx, k_y=ky)
    ch = equate_neat(xt, v, yt, v, method="chained", k_x=kx, k_y=ky, k_v=kv)
    fe = equate_neat(xt, v, yt, v, method="frequency_estimation", k_x=kx, k_y=ky, k_v=kv, w1=0.5)
    assert np.max(np.abs(ch.y_equivalents - eg.y_equivalents)) < 1e-9
    assert np.max(np.abs(fe.y_equivalents - eg.y_equivalents)) < 1e-9
    assert ch.design == "NEAT"

    with pytest.raises(ValueError):
        equate_observed_scores(x, y, method="bogus", k_x=k, k_y=k + 5)


def test_kernel_equating_and_presmoothing():
    """Kernel equating + log-linear presmoothing (von Davier et al., 2004;
    Holland & Thayer, 2000) through the public API: presmoothing preserves
    moments, uniform-kernel ext matches equipercentile, and a large bandwidth
    drives kernel equating to linear."""
    import numpy as np
    import pytest
    from fast_mlsirm import (
    velicer_map,
    velicer_map_from_data,
        equate_observed_scores,
        equate_observed_scores_kernel,
        loglinear_smooth,
    )
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_observed_scores_ext"):
        pytest.skip("compiled core built without kernel equating")

    rng = np.random.default_rng(7)
    k = 40
    x = np.clip(np.round(20 + 7 * rng.standard_normal(5000)), 0, k)
    counts = np.bincount(x.astype(int), minlength=k + 1).astype(float)

    # presmoothing preserves the first `degree` moments (on the u=x/k scale)
    fit = loglinear_smooth(counts, degree=4)
    assert fit["converged"] and abs(fit["probs"].sum() - 1.0) < 1e-12
    g = counts / counts.sum()
    for j, fm in enumerate(fit["moments"], start=1):
        sm = float(((np.arange(k + 1) / k) ** j * g).sum())
        assert abs(fm - sm) < 1e-8, f"moment {j}: {fm} vs {sm}"

    # uniform-kernel ext == equipercentile
    y = np.clip(np.round(22 + 8 * rng.standard_normal(5000)), 0, k)
    base = equate_observed_scores(x, y, method="equipercentile", k_x=k, k_y=k)
    uni = equate_observed_scores_kernel(x, y, continuization="uniform", k_x=k, k_y=k)
    assert np.max(np.abs(base.y_equivalents - uni.y_equivalents)) < 1e-12
    assert np.isnan(uni.h_x)

    # large-bandwidth Gaussian kernel -> linear equating
    lin = equate_observed_scores(x, y, method="linear", k_x=k, k_y=k)
    ker = equate_observed_scores_kernel(
        x, y, continuization="gaussian", k_x=k, k_y=k, bandwidth_x=1e6, bandwidth_y=1e6
    )
    assert np.max(np.abs(lin.y_equivalents - ker.y_equivalents)) < 1e-3
    assert ker.h_x == 1e6

    # penalty-selected bandwidth is finite and positive
    auto = equate_observed_scores_kernel(x, y, continuization="gaussian", k_x=k, k_y=k)
    assert np.isfinite(auto.h_x) and auto.h_x > 0

    # default degree clamps to k, so short forms (k < 6) do not error
    short = loglinear_smooth(np.array([10.0, 20.0, 30.0, 15.0, 8.0, 4.0]))  # k = 5
    assert short["converged"] and short["probs"].shape == (6,)

    # A successful function return is not a successful optimization.  This sparse
    # fixture stops before satisfying the log-linear score tolerance, so the
    # high-level equating path must not consume its unfinished density.
    sparse_counts = np.array([0, 1564, 426, 0, 1008, 0, 0])
    unfinished = loglinear_smooth(sparse_counts, degree=5)
    assert not unfinished["converged"] and unfinished["iters"] < 50
    assert unfinished["termination_reason"] == "line_search_stalled"
    assert unfinished["final_gradient_max"] > unfinished["gradient_tolerance"]
    sparse_scores = np.repeat(np.arange(7), sparse_counts).astype(float)
    with pytest.raises(ValueError, match="presmoothing did not converge"):
        equate_observed_scores_kernel(
            sparse_scores,
            sparse_scores,
            continuization="uniform",
            k_x=6,
            k_y=6,
            smooth_x=5,
            smooth_y=5,
        )

    with pytest.raises(ValueError):
        equate_observed_scores_kernel(x, y, continuization="bogus", k_x=k, k_y=k)
    with pytest.raises(ValueError):
        equate_observed_scores_kernel(x, y, continuization="gaussian", k_x=k, k_y=k, smooth_x=-1)


def test_equate_neat_linear_tucker_levine():
    """Tucker & Levine linear NEAT equating (Kolen & Brennan, 2014) through the
    public API: with equal anchor moments every variant collapses to EG linear,
    and the internal/external Levine gamma give distinct conversions."""
    import numpy as np
    import pytest
    from fast_mlsirm import equate_neat_linear, equate_observed_scores
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_neat_linear"):
        pytest.skip("compiled core built without equate_neat_linear")

    rng = np.random.default_rng(5)
    n, kx, ky = 4000, 30, 40
    anchor = np.clip(np.round(7 + 3 * rng.standard_normal(n)), 0, 15)
    xt = np.clip(np.round(1.5 * anchor + 4 + 3 * rng.standard_normal(n)), 0, kx)
    yt = np.clip(np.round(1.8 * anchor + 6 + 4 * rng.standard_normal(n)), 0, ky)

    # collapse: shared anchor -> equal anchor moments -> EG linear
    eg = equate_observed_scores(xt, yt, method="linear", k_x=kx, k_y=ky)
    for method in ("tucker", "levine"):
        for ak in ("internal", "external"):
            r = equate_neat_linear(xt, anchor, yt, anchor, method=method, anchor_kind=ak, k_x=kx, k_y=ky)
            assert abs(r.slope - eg.slope) < 1e-9 and abs(r.intercept - eg.intercept) < 1e-9
            assert r.design == "NEAT"

    # internal vs external Levine differ on a genuinely non-equivalent anchor
    ancx = np.clip(np.round(6 + 2.5 * rng.standard_normal(n)), 0, 15)
    ancy = np.clip(np.round(9 + 2.5 * rng.standard_normal(n)), 0, 15)  # shifted
    xt2 = np.clip(np.round(1.2 * ancx + 5 + 3 * rng.standard_normal(n)), 0, kx)
    yt2 = np.clip(np.round(1.2 * ancy + 8 + 3 * rng.standard_normal(n)), 0, ky)
    li = equate_neat_linear(xt2, ancx, yt2, ancy, method="levine", anchor_kind="internal", k_x=kx, k_y=ky)
    le = equate_neat_linear(xt2, ancx, yt2, ancy, method="levine", anchor_kind="external", k_x=kx, k_y=ky)
    tk = equate_neat_linear(xt2, ancx, yt2, ancy, method="tucker", k_x=kx, k_y=ky)
    assert abs(li.slope - le.slope) > 1e-6 or abs(li.intercept - le.intercept) > 1e-6
    assert abs(tk.slope - li.slope) > 1e-6 or abs(tk.intercept - li.intercept) > 1e-6

    with pytest.raises(ValueError):
        equate_neat_linear(xt, anchor, yt, anchor, method="bogus", k_x=kx, k_y=ky)


def test_equating_standard_errors():
    """Standard errors of equating (Kolen & Brennan, 2014, ch. 7) through the
    public API: analytic and bootstrap linear SEE agree, Mean SEE is constant,
    the CI brackets the point estimate, and equipercentile SEE is bootstrap-only."""
    import numpy as np
    import pytest
    from fast_mlsirm import equating_standard_errors
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "bootstrap_see"):
        pytest.skip("compiled core built without SEE")

    rng = np.random.default_rng(7)
    k, n = 30, 3000
    x = np.clip(np.round(15 + 5 * rng.standard_normal(n)), 0, k)
    y = np.clip(np.round(16 + 5 * rng.standard_normal(n)), 0, k)

    a = equating_standard_errors(x, y, method="linear", route="analytic", k_x=k, k_y=k)
    b = equating_standard_errors(x, y, method="linear", route="bootstrap", k_x=k, k_y=k, n_boot=2000, seed=1)
    lo, hi = int(np.ceil(k * 0.1)), k - int(np.ceil(k * 0.1))
    rel = np.abs(b["se"][lo:hi] - a["se"][lo:hi]) / a["se"][lo:hi]
    assert rel.max() < 0.15, f"analytic vs bootstrap linear SEE: {rel.max()}"
    # CI brackets the point estimate
    assert np.all(b["ci_lo"][lo:hi] <= b["y_equivalents"][lo:hi] + 1e-9)
    assert np.all(b["y_equivalents"][lo:hi] <= b["ci_hi"][lo:hi] + 1e-9)
    assert b["n_boot"] == 2000 and abs(b["ci_level"] - 0.95) < 1e-12

    # Mean SEE constant in x
    m = equating_standard_errors(x, y, method="mean", route="analytic", k_x=k, k_y=k)
    assert np.allclose(m["se"], m["se"][0])

    # equipercentile has no analytic SEE; bootstrap works
    ep = equating_standard_errors(x, y, method="equipercentile", route="bootstrap", k_x=k, k_y=k, n_boot=300, seed=3)
    assert np.all(ep["se"][lo:hi] > 0) and ep["n_boot"] == 300
    with pytest.raises(ValueError):
        equating_standard_errors(x, y, method="equipercentile", route="analytic", k_x=k, k_y=k)
    with pytest.raises(ValueError):
        equating_standard_errors(x, y, method="linear", route="bogus", k_x=k, k_y=k)
    with pytest.raises(ValueError, match="bootstrap buffer size"):
        equating_standard_errors(
            x,
            y,
            method="mean",
            route="bootstrap",
            k_x=k,
            k_y=k,
            n_boot=np.iinfo(np.uintp).max,
        )


def test_fit_response_times():
    """Lognormal response-time model (van der Linden, 2007) through the public
    API: recovers the item time parameters, the speed SD, and the speed EAP, and
    handles missing (NaN) response times."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_response_times
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rt_lognormal"):
        pytest.skip("compiled core built without fit_rt_lognormal")

    rng = np.random.default_rng(4)
    n, m = 800, 20
    beta = np.linspace(3.5, 4.5, m)
    alpha = np.linspace(1.0, 3.0, m)
    sigma = 0.3
    tau = sigma * rng.standard_normal(n)
    y = beta[None, :] - tau[:, None] + rng.standard_normal((n, m)) / alpha[None, :]
    times = np.exp(y)
    # inject ~20% missing as NaN
    times[rng.random((n, m)) < 0.2] = np.nan

    fit = fit_response_times(times)
    assert fit.converged
    assert fit.termination_reason == "converged"
    assert fit.n_iter < 500
    assert fit.final_loglik_change < 1e-6
    assert np.isfinite(fit.loglik_trace).all()
    assert np.all(
        np.diff(fit.loglik_trace)
        >= -1e-6 * np.maximum(np.abs(fit.loglik_trace[:-1]), 1)
    )
    assert fit.loglik == fit.loglik_trace[-1]
    assert fit.alpha.shape == (m,) and fit.tau_eap.shape == (n,)
    assert np.corrcoef(fit.beta, beta)[0, 1] > 0.95
    assert np.corrcoef(fit.alpha, alpha)[0, 1] > 0.85
    assert np.corrcoef(fit.tau_eap, tau)[0, 1] > 0.8
    assert abs(fit.sigma_tau - sigma) < 0.1
    assert fit.mu_tau == 0.0
    # loglik trace is non-decreasing (monotone EM)
    # (exposed via n_iter/converged; recompute a small fit to confirm determinism)
    fit2 = fit_response_times(times)
    assert np.allclose(fit.beta, fit2.beta)

    with pytest.warns(RuntimeWarning, match="max_iter_reached"):
        fit_nc = fit_response_times(times, max_iter=1)
    assert not fit_nc.converged
    assert fit_nc.termination_reason == "max_iter_reached"
    assert fit_nc.n_iter == 1
    assert fit_nc.final_loglik_change >= 1e-6

    with pytest.raises(RuntimeError, match="max_iter_reached"):
        fit_response_times(times, max_iter=1, require_convergence=True)

    with pytest.raises(ValueError, match="max_iter"):
        fit_response_times(times, max_iter=0)

    with pytest.raises(ValueError):
        fit_response_times(times.ravel())  # not 2-D


def test_fit_speed_accuracy():
    """Joint speed-accuracy model (van der Linden, 2007, Level 2) through the
    public API: recovers a positive ability-speed correlation with item banks
    fixed, and returns ~0 under true independence."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_response_times, fit_speed_accuracy
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_speed_accuracy_covariance"):
        pytest.skip("compiled core built without fit_speed_accuracy_covariance")

    rng = np.random.default_rng(7)
    n, m = 1000, 20
    a = 0.9 + 0.6 * (np.arange(m) % 3) / 2.0
    b = np.linspace(-1.5, 1.5, m)
    alpha = np.linspace(1.0, 3.0, m)
    beta = np.linspace(3.5, 4.5, m)

    def sim(rho, sig=0.3):
        za = rng.standard_normal(n)
        zb = rng.standard_normal(n)
        theta = za
        tau = rho * sig * za + sig * np.sqrt(1 - rho * rho) * zb
        pr = 1.0 / (1.0 + np.exp(-(a[None, :] * theta[:, None] + b[None, :])))
        resp = (rng.random((n, m)) < pr).astype(float)
        y = beta[None, :] - tau[:, None] + rng.standard_normal((n, m)) / alpha[None, :]
        return resp, np.exp(y)

    resp, times = sim(0.5)
    res = fit_speed_accuracy(resp, times, a, b, alpha, beta)
    assert res["converged"]
    assert res["termination_reason"] == "converged"
    assert res["n_iter"] < 500
    assert res["final_loglik_change"] < 1e-6
    assert np.isfinite(res["loglik_trace"]).all()
    assert np.all(np.diff(res["loglik_trace"]) >= -1e-6 * np.maximum(np.abs(res["loglik_trace"][:-1]), 1))
    assert res["loglik"] == res["loglik_trace"][-1]
    assert abs(res["rho"] - 0.5) < 0.1, res["rho"]
    assert abs(res["sigma_tau"] - 0.3) < 0.05
    assert res["theta_eap"].shape == (n,) and res["tau_eap"].shape == (n,)

    # true independence -> rho ~ 0
    r0, t0 = sim(0.0)
    res0 = fit_speed_accuracy(r0, t0, a, b, alpha, beta)
    assert res0["converged"]
    assert res0["termination_reason"] == "converged"
    assert res0["final_loglik_change"] < 1e-6
    assert abs(res0["rho"]) < 0.08, res0["rho"]

    # works with a fitted RT model's alpha/beta
    rt = fit_response_times(times)
    res_rt = fit_speed_accuracy(resp, times, a, b, rt.alpha, rt.beta)
    assert res_rt["converged"]
    assert res_rt["termination_reason"] == "converged"
    assert res_rt["final_loglik_change"] < 1e-6
    assert abs(res_rt["rho"] - 0.5) < 0.15

    with pytest.warns(RuntimeWarning, match="max_iter_reached"):
        res_nc = fit_speed_accuracy(
            resp,
            times,
            a,
            b,
            alpha,
            beta,
            q=7,
            max_iter=1,
        )
    assert not res_nc["converged"]
    assert res_nc["termination_reason"] == "max_iter_reached"
    assert res_nc["n_iter"] == 1
    assert res_nc["final_loglik_change"] >= 1e-6

    with pytest.raises(RuntimeError, match="max_iter_reached"):
        fit_speed_accuracy(
            resp,
            times,
            a,
            b,
            alpha,
            beta,
            q=7,
            max_iter=1,
            require_convergence=True,
        )

    with pytest.raises(ValueError):
        fit_speed_accuracy(resp.ravel(), times, a, b, alpha, beta)  # not 2-D

    with pytest.raises(ValueError, match="observed"):
        fit_speed_accuracy(
            np.full((1, 1), np.nan),
            np.full((1, 1), np.nan),
            np.ones(1),
            np.zeros(1),
            np.ones(1),
            np.ones(1),
        )

    with pytest.raises(ValueError, match="discrimination"):
        fit_speed_accuracy(
            np.array([[0.0], [1.0]]),
            np.array([[2.0], [1.8]]),
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            np.ones(1),
        )

    with pytest.raises(ValueError, match="non-finite"):
        fit_speed_accuracy(
            np.array([[0.0], [1.0]]),
            np.array([[2.0], [1.8]]),
            np.ones(1),
            np.zeros(1),
            np.array([1e308]),
            np.ones(1),
        )


def test_rt_person_fit():
    """RT person fit (van der Linden & Guo, 2008): W ~ chi2(n-1) with l_t ~ N(0,1)
    on model-consistent data, and rapid-guessing responders flagged."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_response_times, rt_person_fit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "rt_person_fit"):
        pytest.skip("compiled core built without rt_person_fit")

    rng = np.random.default_rng(3)
    n, m = 2000, 20
    beta = np.linspace(3.5, 4.5, m)
    alpha = np.linspace(1.0, 3.0, m)
    tau = 0.3 * rng.standard_normal(n)
    y = beta[None, :] - tau[:, None] + rng.standard_normal((n, m)) / alpha[None, :]
    # first 10% rapid-guess on the last 7 items
    n_ab = n // 10
    for p in range(n_ab):
        y[p, -7:] = (beta[-7:] - tau[p]) - 2.5 + 0.3 * rng.standard_normal(7)
    times = np.exp(y)

    # exact calibration with the (uncontaminated) item parameters: W ~ chi2(n-1),
    # l_t ~ N(0,1) on the clean responders, ~.05 Type I, high power
    pf = rt_person_fit(times, alpha, beta)
    assert pf["w"].shape == (n,) and pf["z_resid"].shape == (n, m)
    clean = pf["l_t"][n_ab:]
    assert abs(clean.mean()) < 0.15 and 0.8 < clean.std() < 1.25
    assert pf["flagged"][n_ab:].mean() < 0.12
    assert pf["flagged"][:n_ab].mean() > 0.7
    # the tampered items are flagged too-fast (strongly negative residual)
    assert pf["item_flag"][:n_ab, -7:].mean() > 0.7
    assert np.all(pf["z_resid"][:n_ab, -7:] < 0)  # too-fast = negative

    # with a fitted RT model (production path) the aberrant are still detected;
    # fitting on the contaminated sample makes the clean responders conservative
    fit = fit_response_times(times)
    pf2 = rt_person_fit(times, fit.alpha, fit.beta)
    assert pf2["flagged"][:n_ab].mean() > 0.6

    with pytest.raises(ValueError):
        rt_person_fit(times.ravel(), alpha, beta)  # not 2-D
    with pytest.raises(ValueError, match="finite positive squares"):
        rt_person_fit(np.array([[1.0, 2.0]]), np.array([1e308, 1.0]), np.zeros(2))
    with pytest.raises(ValueError, match="finite positive squares"):
        rt_person_fit(np.array([[1.0, 2.0]]), np.full(2, 1e-308), np.zeros(2))
    with pytest.raises(ValueError, match="non-finite"):
        rt_person_fit(np.array([[1.0, 2.0]]), np.ones(2), np.full(2, 1e308))


def _sim_cdm(rng, q, s, g, profiles, model="dina"):
    """Simulate DINA/DINO responses for the given bit-encoded true profiles."""
    n, (n_items, k) = len(profiles), q.shape
    y = np.empty((n, n_items))
    for j in range(n):
        for i in range(n_items):
            mask = int(np.dot(q[i], 1 << np.arange(k)))
            c = int(profiles[j])
            eta = (c & mask) == mask if model == "dina" else (c & mask) != 0
            p = 1.0 - s[i] if eta else g[i]
            y[j, i] = 1.0 if rng.random() < p else 0.0
    return y


def test_fit_cdm_dina_recovers_and_classifies():
    """DINA cognitive diagnosis (de la Torre, 2009): recover slip/guess and classify
    attribute mastery under a known Q-matrix; DINO reduces to DINA on single-attribute
    items."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_cdm, CdmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_cdm"):
        pytest.skip("compiled core built without fit_cdm")

    rng = np.random.default_rng(11)
    k, n_items, n = 3, 15, 1500
    # 6 single-attribute items (2 per attribute) + pairs + one triple.
    rows = []
    for a in range(k):
        rows += [[1 if t == a else 0 for t in range(k)]] * 2
    rows += [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]]
    q = np.array(rows[:n_items], dtype=np.int64)
    s = np.full(n_items, 0.15)
    g = np.full(n_items, 0.15)
    profiles = rng.integers(0, 1 << k, size=n)
    y = _sim_cdm(rng, q, s, g, profiles)

    res = fit_cdm(y, q, model="dina")
    assert isinstance(res, CdmFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert np.all(1.0 - res.slip > res.guess)  # identification
    assert np.sqrt(np.mean((res.slip - s) ** 2)) < 0.05
    assert np.sqrt(np.mean((res.guess - g) ** 2)) < 0.05
    # attribute classification agreement (marginal mastery vs truth)
    true_bits = ((profiles[:, None] >> np.arange(k)) & 1)
    attr_ok = (res.attribute_mastery() == true_bits).mean()
    assert attr_ok > 0.85, attr_ok
    # pattern-wise agreement (exact 3-bit profile)
    assert (res.map_profile == profiles).mean() > 0.75
    assert res.n_parameters == 2 * n_items + ((1 << k) - 1)
    assert res.profile_bits().shape == (n, k)

    # single-attribute Q => DINA and DINO share identical eta and thus identical fits.
    q1 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int64)
    prof1 = rng.integers(0, 4, size=800)
    y1 = _sim_cdm(rng, q1, np.full(4, 0.2), np.full(4, 0.2), prof1)
    a = fit_cdm(y1, q1, model="dina")
    b = fit_cdm(y1, q1, model="dino")
    assert np.allclose(a.slip, b.slip, atol=1e-9)
    assert np.allclose(a.guess, b.guess, atol=1e-9)

    # missing-at-random cells are dropped, not imputed.
    ym = y.copy()
    ym[rng.random(ym.shape) < 0.15] = np.nan
    resm = fit_cdm(ym, q, model="dina")
    assert resm.converged and np.all(1.0 - resm.slip > resm.guess)

    with pytest.raises(ValueError):
        fit_cdm(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_cdm(y, q, model="rasch")  # unknown gate
    with pytest.raises(ValueError):
        fit_cdm(y, np.zeros((n_items, k), dtype=np.int64))  # all-zero Q rows/cols


def test_fit_gdina_recovers_saturated_and_reduces_to_dina():
    """Saturated G-DINA (de la Torre, 2011): recover free reduced-class success
    probabilities, and confirm DINA-generated data yields the DINA identity-link
    pattern (only intercept + highest-order interaction nonzero)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_gdina, GdinaFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gdina"):
        pytest.skip("compiled core built without fit_gdina")

    def reduce_class(c, qmask, k):
        l, m = 0, 0
        for bit in range(k):
            if (qmask >> bit) & 1:
                l |= ((c >> bit) & 1) << m
                m += 1
        return l

    rng = np.random.default_rng(2011)
    k, n_items, n = 2, 12, 2500
    q = np.zeros((n_items, k), dtype=np.int64)
    for i in range(n_items):
        if i < 4:
            q[i, 0] = 1
        elif i < 8:
            q[i, 1] = 1
        else:
            q[i] = [1, 1]
    qmask = [int(np.dot(q[i], 1 << np.arange(k))) for i in range(n_items)]
    item_off = np.concatenate([[0], np.cumsum([1 << int(q[i].sum()) for i in range(n_items)])])

    # DINA truth in CSR layout: top reduced class = 1 - s, all others = g.
    s, g = 0.15, 0.2
    truth = np.empty(item_off[-1])
    for i in range(n_items):
        truth[item_off[i] : item_off[i + 1]] = g
        truth[item_off[i + 1] - 1] = 1.0 - s
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.empty((n, n_items))
    for j in range(n):
        for i in range(n_items):
            p = truth[item_off[i] + reduce_class(int(profiles[j]), qmask[i], k)]
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = fit_gdina(y, q)
    assert isinstance(res, GdinaFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert np.sqrt(np.mean((res.item_prob - truth) ** 2)) < 0.03
    assert res.n_parameters == int(item_off[-1]) + ((1 << k) - 1)
    # DINA identity-link pattern per item: delta_0 ~ g, delta_full ~ (1-s)-g, mids ~ 0.
    for i in range(n_items):
        d = res.item_delta_row(i)
        assert abs(d[0] - g) < 0.05
        assert abs(d[-1] - ((1.0 - s) - g)) < 0.05
        if len(d) > 2:
            assert np.all(np.abs(d[1:-1]) < 0.05)
    # For this DINA-generated monotone truth, the all-mastered class is highest;
    # unconstrained saturated G-DINA does not guarantee this for arbitrary data.
    for i in range(n_items):
        row = res.item_prob_row(i)
        assert row[-1] >= row.max() - 1e-9

    # missing-at-random cells are dropped.
    ym = y.copy()
    ym[rng.random(ym.shape) < 0.15] = np.nan
    assert fit_gdina(ym, q).converged

    with pytest.raises(ValueError):
        fit_gdina(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_gdina(y, np.zeros((n_items, k), dtype=np.int64))  # all-zero Q rows/cols


def test_fit_gdina_does_not_claim_or_project_order_restrictions():
    """The saturated estimator leaves G-DINA order restrictions unconstrained."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_gdina
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gdina"):
        pytest.skip("compiled core built without fit_gdina")

    rng = np.random.default_rng(20260720)
    k, n_items, n = 2, 12, 1500
    q = np.zeros((n_items, k), dtype=np.int64)
    q[:4, 0] = 1
    q[4:8, 1] = 1
    q[8:, :] = 1
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.empty((n, n_items), dtype=float)
    joint_prob = np.array([0.1, 0.8, 0.7, 0.2])
    for person, profile in enumerate(profiles):
        for item in range(n_items):
            if item < 4:
                prob = (0.2, 0.8)[profile & 1]
            elif item < 8:
                prob = (0.2, 0.8)[(profile >> 1) & 1]
            else:
                prob = joint_prob[profile]
            y[person, item] = rng.random() < prob

    result = fit_gdina(y, q, max_iter=500, tol=1e-6)
    delta = np.diff(result.loglik_trace)
    assert result.converged and result.n_iter < 500
    assert np.isfinite(result.loglik_trace).all()
    assert np.all(delta >= -1e-6)
    assert abs(delta[-1]) < 1e-6
    for item in range(8, n_items):
        row = result.item_prob_row(item)
        assert row[-1] < row[1:-1].max() - 0.3


def test_validate_q_matrix_corrects_misspecification():
    """PVAF Q-matrix validation (de la Torre & Chiu, 2016): the true Q validates to
    itself, and a Q with an over-specified and an under-specified item is corrected
    back to the truth while flagging exactly those items."""
    import numpy as np
    import pytest
    from fast_mlsirm import validate_q_matrix, QMatrixValidation
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "validate_q_matrix"):
        pytest.skip("compiled core built without validate_q_matrix")

    rng = np.random.default_rng(715)
    k, n_items, n = 3, 15, 3000
    rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    truth = np.array(rows, dtype=np.int64)
    s = np.full(n_items, 0.1)
    g = np.full(n_items, 0.1)
    profiles = rng.integers(0, 1 << k, size=n)
    y = _sim_cdm(rng, truth, s, g, profiles)

    # Anchor: the true Q validates to itself, nothing flagged.
    res = validate_q_matrix(y, truth, epsilon=0.95)
    assert isinstance(res, QMatrixValidation)
    assert np.array_equal(res.suggested_q, truth)
    assert not res.flagged.any()
    assert np.all(res.provisional_pvaf > 0.9)
    assert res.calibration_termination_reason == "tolerance_met"
    assert 0 < res.calibration_n_iter < res.calibration_max_iter
    assert 0 <= res.calibration_final_loglik_change < res.calibration_tol

    # Over-specify item 0 ({0} -> {0,1}) and under-specify item 6 ({0,1} -> {0}).
    prov = truth.copy()
    prov[0, 1] = 1
    prov[6, 1] = 0
    res2 = validate_q_matrix(y, prov, epsilon=0.95)
    assert np.array_equal(res2.suggested_q[0], truth[0])  # trimmed back
    assert np.array_equal(res2.suggested_q[6], truth[6])  # enlarged back
    assert res2.flagged[0] and res2.flagged[6]
    # the under-specified item's provisional q falls short of the cutoff
    assert res2.provisional_pvaf[6] < 0.95
    assert res2.calibration_termination_reason == "tolerance_met"
    assert 0 < res2.calibration_n_iter < res2.calibration_max_iter
    assert 0 <= res2.calibration_final_loglik_change < res2.calibration_tol

    with pytest.raises(ValueError):
        validate_q_matrix(y.ravel(), truth)  # responses not 2-D
    with pytest.raises(ValueError):
        validate_q_matrix(y, truth, epsilon=1.5)  # epsilon out of range
    with pytest.raises(
        ValueError,
        match=r"G-DINA calibration did not converge after 1 of 1 M-steps",
    ):
        validate_q_matrix(y, truth, max_iter=1, tol=1e-12)


def test_gdina_wald_selection_classifies_items():
    """Item-level Wald model selection (de la Torre, 2011; de la Torre & Lee, 2013):
    a conjunctive (DINA), disjunctive (DINO), identity-additive (A-CDM), logit-additive
    (LLM), and log-additive (R-RUM) item are each classified as their reduced model,
    and an item with both main effects and an interaction keeps the saturated G-DINA.
    The LLM and R-RUM truths are additive ONLY on their own link (identity- and
    cross-link-nonadditive), so classifying them correctly exercises the link-transformed
    delta and its delta-method covariance, not just the identity link."""
    import numpy as np
    import pytest
    from fast_mlsirm import gdina_wald_selection, WaldModelSelection
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "gdina_wald_selection"):
        pytest.skip("compiled core built without gdina_wald_selection")

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    rng = np.random.default_rng(2011)
    k, n = 2, 8000
    # 5 single-attribute items per attribute (identification) + 6 pair items:
    # DINA, DINO, A-CDM, LLM, R-RUM, saturated.
    rows = [[1, 0]] * 5 + [[0, 1]] * 5 + [[1, 1]] * 6
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    # per reduced-class truth [none, a0, a1, both]
    truth_pair = {10: [0.15, 0.15, 0.15, 0.85],  # DINA (conjunctive)
                  11: [0.15, 0.85, 0.85, 0.85],  # DINO (disjunctive)
                  12: [0.10, 0.45, 0.45, 0.80],  # A-CDM (identity-additive)
                  # LLM: logit(P) = -3 + 2 a0 + 2 a1 (logit-additive, identity- &
                  # log-nonadditive).
                  13: [sig(-3.0), sig(-1.0), sig(-1.0), sig(1.0)],
                  # R-RUM: P = 0.92 * 0.3^(1-a0) * 0.4^(1-a1) (log-additive, identity- &
                  # logit-nonadditive).
                  14: [0.92 * 0.3 * 0.4, 0.92 * 0.4, 0.92 * 0.3, 0.92],
                  15: [0.10, 0.35, 0.35, 0.90]}  # saturated
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.empty((n, n_items))
    for j in range(n):
        c = int(profiles[j])
        for i in range(n_items):
            if i < 10:
                a = i // 5  # attribute of this single item
                p = 0.85 if (c >> a) & 1 else 0.15
            else:
                l = (c & 1) + 2 * ((c >> 1) & 1)  # reduced class for a {0,1} item
                p = truth_pair[i][l]
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = gdina_wald_selection(y, q, alpha=0.05)
    assert isinstance(res, WaldModelSelection)
    assert res.models == ["dina", "dino", "acdm", "llm", "rrum"]
    assert res.selected[10] == 0   # DINA
    assert res.selected[11] == 1   # DINO
    assert res.selected[12] == 2   # A-CDM
    assert res.selected[13] == 3   # LLM (logit link)
    assert res.selected[14] == 4   # R-RUM (log link)
    assert res.selected[15] == -1  # saturated G-DINA
    # single-attribute items carry no test (df 0), keep saturated
    assert np.all(res.selected[:10] == -1)
    assert np.all(res.wald_df[:10] == 0)
    # the tested pair items have the right degrees of freedom (K=2): DINA & DINO
    # df = 2^K-2 = 2; A-CDM, LLM, R-RUM df = 2^K-1-K = 1.
    assert res.wald_df[10, 0] == 2 and res.wald_df[10, 1] == 2
    assert res.wald_df[10, 2] == 1 and res.wald_df[10, 3] == 1 and res.wald_df[10, 4] == 1

    with pytest.raises(ValueError):
        gdina_wald_selection(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        gdina_wald_selection(y, q, alpha=0.0)  # alpha out of range
    with pytest.raises(
        ValueError,
        match=r"G-DINA calibration did not converge after 1 of 1 M-steps",
    ):
        gdina_wald_selection(y, q, max_iter=1, tol=1e-12)


def test_fit_ho_cdm_recovers_higher_order_structure():
    """Higher-order DINA (de la Torre & Douglas, 2004): a continuous trait structures
    attribute mastery; recover the attribute slopes/intercepts, slip/guess, and
    classification, and confirm the slope-zero reduction to independent attributes."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_ho_cdm, HoCdmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_ho_cdm"):
        pytest.skip("compiled core built without fit_ho_cdm")

    rng = np.random.default_rng(2004)
    k, n = 3, 4000
    # 4 single-attribute items per attribute + 3 pair items (all attributes identified)
    rows = []
    for a in range(k):
        for _ in range(4):
            rows.append([1 if t == a else 0 for t in range(k)])
    rows += [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    a_true = np.array([1.2, 1.5, 0.9])
    d_true = np.array([0.3, -0.5, 0.6])
    s, g = np.full(n_items, 0.12), np.full(n_items, 0.12)

    theta = rng.standard_normal(n)
    alpha = (rng.random((n, k)) < 1.0 / (1.0 + np.exp(-(theta[:, None] * a_true + d_true)))).astype(int)
    codes = (alpha * (1 << np.arange(k))).sum(1)
    y = np.empty((n, n_items))
    for j in range(n):
        c = int(codes[j])
        for i in range(n_items):
            mask = int(np.dot(q[i], 1 << np.arange(k)))
            eta = (c & mask) == mask
            p = 1.0 - s[i] if eta else g[i]
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = fit_ho_cdm(y, q, model="dina")
    assert isinstance(res, HoCdmFit) and res.converged
    assert res.termination_reason == "tolerance_met"
    assert 0 < res.n_iter < res.max_iter == 500
    assert 0 <= res.final_loglik_change < res.stopping_tolerance == 1e-6
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert res.n_parameters == 2 * n_items + 2 * k
    assert abs(res.profile_prob.sum() - 1.0) < 1e-9
    assert np.all(res.attr_slope > 0)  # anchored non-negative
    assert np.sqrt(np.mean((res.slip - s) ** 2)) < 0.05
    assert np.sqrt(np.mean((res.attr_slope - a_true) ** 2)) < 0.4  # identified at K=3
    assert np.sqrt(np.mean((res.attr_intercept - d_true) ** 2)) < 0.3
    # attribute classification agreement
    est = res.attribute_mastery()
    assert (est == alpha).mean() > 0.85

    limited = fit_ho_cdm(y, q, model="dina", max_iter=1, tol=1e-12)
    assert not limited.converged
    assert limited.termination_reason == "max_iter_reached"
    assert limited.n_iter == limited.max_iter == 1
    assert np.isfinite(limited.final_loglik_change)
    assert limited.final_loglik_change >= limited.stopping_tolerance == 1e-12

    with pytest.raises(ValueError):
        fit_ho_cdm(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_ho_cdm(y, q, model="rasch")  # unknown gate


def test_fit_ho_gdina_recovers_saturated_and_structure():
    """Higher-order G-DINA (de la Torre & Douglas, 2004; de la Torre, 2011): a free
    saturated item fit of DINA-patterned data recovers the DINA identity-link delta
    and the higher-order attribute parameters."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_ho_gdina, HoGdinaFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_ho_gdina"):
        pytest.skip("compiled core built without fit_ho_gdina")

    rng = np.random.default_rng(2011)
    k, n = 3, 3000
    rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 3  # 9 singles
    rows += [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]  # 5 pairs
    rows += [[1, 1, 1]]  # 1 triple
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    s, g = 0.15, 0.2
    a_true = np.array([1.2, 1.5, 0.9])
    d_true = np.array([0.3, -0.5, 0.6])

    theta = rng.standard_normal(n)
    alpha = (rng.random((n, k)) < 1.0 / (1.0 + np.exp(-(theta[:, None] * a_true + d_true)))).astype(int)
    codes = (alpha * (1 << np.arange(k))).sum(1)
    y = np.empty((n, n_items))
    for j in range(n):
        c = int(codes[j])
        for i in range(n_items):
            mask = int(np.dot(q[i], 1 << np.arange(k)))
            eta = (c & mask) == mask  # DINA gate
            p = (1.0 - s) if eta else g
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = fit_ho_gdina(y, q)
    assert isinstance(res, HoGdinaFit) and res.converged
    assert res.termination_reason == "tolerance_met"
    assert res.final_relative_loglik_change < res.stopping_tolerance
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert np.all(res.attr_slope > 0)  # anchored non-negative
    assert abs(res.profile_prob.sum() - 1.0) < 1e-9
    # the triple item's identity-link delta shows the DINA pattern (intercept + top)
    triple = n_items - 1
    dl = res.item_delta[res.item_off[triple] : res.item_off[triple + 1]]
    assert abs(dl[0] - g) < 0.06 and abs(dl[-1] - ((1.0 - s) - g)) < 0.06
    assert np.all(np.abs(dl[1:-1]) < 0.06)
    # higher-order parameter recovery (identified at K=3)
    assert np.sqrt(np.mean((res.attr_slope - a_true) ** 2)) < 0.45
    # attribute classification
    est = (res.attr_prob >= 0.5).astype(int)
    assert (est == alpha).mean() > 0.9

    with pytest.raises(ValueError):
        fit_ho_gdina(y.ravel(), q)  # not 2-D
    with pytest.raises(ValueError, match="at least 3 attributes"):
        fit_ho_gdina(y[:, :2], np.eye(2, dtype=np.int64))
    for bad in (np.inf, -np.inf):
        malformed = y.copy()
        malformed[0, 0] = bad
        with pytest.raises(ValueError, match="only 0, 1, or NaN"):
            fit_ho_gdina(malformed, q)

    limited = y[:100].copy()
    limited[0, 0] = np.nan
    unfinished = fit_ho_gdina(limited, q, max_iter=1, tol=1e-12)
    assert not unfinished.converged
    assert unfinished.n_iter == 1
    assert unfinished.termination_reason == "max_iter_reached"
    assert np.isfinite(unfinished.final_loglik_change)
    assert np.isfinite(unfinished.final_relative_loglik_change)


def test_fit_seq_gdina_recovers_polytomous_and_reduces_to_gdina():
    """Shared-Q sequential G-DINA (Ma & de la Torre, 2016): recover the ordered-category
    step and category probabilities of a polytomous item, and confirm that binary data
    (one step per item) reduces exactly to :func:`fit_gdina`. The M=2 truth is additive
    on neither trivial link — its two step tables are distinct and asymmetric, so the
    ordered structure (not a degenerate collapse) is what is recovered."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_seq_gdina, SeqGdinaFit, fit_gdina
    from fast_mlsirm.fitstats import _core_module

    for public_doc in (SeqGdinaFit.__doc__, fit_seq_gdina.__doc__):
        assert "fit_seq_gdina_qr" in (public_doc or "")
        assert "deferred non-goal" not in (public_doc or "")

    core = _core_module()
    if core is None or not hasattr(core, "fit_seq_gdina"):
        pytest.skip("compiled core built without fit_seq_gdina")

    rng = np.random.default_rng(2016)
    k, n = 2, 6000
    # 4 single-attribute items per attribute (M=1 identification) + 4 pair M=2 items.
    rows = [[1, 0]] * 4 + [[0, 1]] * 4 + [[1, 1]] * 4
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    # per reduced-class step tables (class-major [00,10,01,11]); asymmetric, increasing.
    pair_s1 = {0: 0.25, 1: 0.55, 2: 0.50, 3: 0.85}
    pair_s2 = {0: 0.15, 1: 0.30, 2: 0.25, 3: 0.70}
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.zeros((n, n_items))
    for j in range(n):
        c = int(profiles[j])
        for i in range(n_items):
            if i < 8:
                a = i // 4
                p = 0.85 if (c >> a) & 1 else 0.15
                y[j, i] = 1.0 if rng.random() < p else 0.0
            else:
                l = (c & 1) + 2 * ((c >> 1) & 1)
                cat = 0
                if rng.random() < pair_s1[l]:
                    cat = 1
                    if rng.random() < pair_s2[l]:
                        cat = 2
                y[j, i] = float(cat)

    res = fit_seq_gdina(y, q)
    assert isinstance(res, SeqGdinaFit) and res.converged
    assert res.termination_reason == "tolerance_met"
    assert abs(res.final_loglik_change) < res.stopping_tolerance
    assert np.isfinite(res.final_relative_loglik_change)
    assert res.max_cat.tolist() == [1] * 8 + [2] * 4  # M_i derived from data
    assert abs(res.profile_prob.sum() - 1.0) < 1e-9
    # category probabilities sum to 1 per (item, reduced class)
    for i in range(n_items):
        cp = res.item_cat_prob(i)
        assert cp.shape == (1 << int(res.k_required[i]), int(res.max_cat[i]) + 1)
        assert np.allclose(cp.sum(axis=1), 1.0, atol=1e-9)
    # recover the pair items' category probabilities (stable, model-predicted quantity)
    for i in range(8, n_items):
        cp = res.item_cat_prob(i)  # 4 classes x 3 categories
        for l in range(4):
            s1, s2 = pair_s1[l], pair_s2[l]
            truth = np.array([1 - s1, s1 * (1 - s2), s1 * s2])
            assert np.max(np.abs(cp[l] - truth)) < 0.04, f"item{i} class{l}: {cp[l]} vs {truth}"
    # attribute classification
    est = (res.attr_prob >= 0.5).astype(int)
    alpha = ((profiles[:, None] >> np.arange(k)) & 1)
    assert (est == alpha).mean() > 0.9

    # Binary data (M_i = 1 for all items) reduces to fit_gdina bit-for-bit.
    ybin = (y[:, :8] > 0).astype(float)
    qbin = q[:8]
    sq1 = fit_seq_gdina(ybin, qbin)
    g1 = fit_gdina(ybin, qbin)
    assert sq1.max_cat.tolist() == [1] * 8
    assert np.allclose(sq1.step_prob, g1.item_prob, atol=1e-12)
    assert len(sq1.loglik_trace) == len(g1.loglik_trace)
    assert np.allclose(sq1.loglik_trace, g1.loglik_trace, atol=1e-12)

    # Validation: an item stuck at category 0 (measures nothing) is rejected; a
    # non-integer category is rejected; missing (NaN) is allowed.
    with pytest.raises(ValueError):
        fit_seq_gdina(y.ravel(), q)  # not 2-D
    yzero = y.copy()
    yzero[:, 8] = 0.0
    with pytest.raises(ValueError, match="never leaves category 0"):
        fit_seq_gdina(yzero, q)
    ybad = y.copy()
    ybad[0, 8] = 1.5
    with pytest.raises(ValueError, match="non-negative integer category"):
        fit_seq_gdina(ybad, q)
    ymiss = y.copy()
    ymiss[0, 0] = np.nan
    assert fit_seq_gdina(ymiss, q).converged

    unfinished = fit_seq_gdina(y[:100], q, max_iter=1, tol=1e-12)
    assert not unfinished.converged
    assert unfinished.n_iter == 1
    assert unfinished.termination_reason == "max_iter_reached"
    assert np.isfinite(unfinished.final_loglik_change)
    assert np.isfinite(unfinished.final_relative_loglik_change)
    assert unfinished.stopping_tolerance == 1e-12


def test_fit_seq_gdina_qr_per_step_q_reduces_and_recovers_structure():
    """Per-step-Q sequential G-DINA (Ma & de la Torre, 2016, restricted-Q full model):
    each ordered step of an item may require its OWN attributes. Three guards:

    (1) SHARED-Q REDUCTION -- expanding every step of an item to the item's Q reproduces
        :func:`fit_seq_gdina` BIT-EXACTLY (layout-aware step_prob compare: item-major
        ``s_off[i]+l*M_i+(k-1)`` vs step-row-major ``spo[step_off[i]+(k-1)]+l``; cat_prob
        and loglik_trace are class-major and compared directly). A dimension-map, layout,
        or union-collapse bug fails this exact-zero guard.
    (2) STRUCTURE -- four items have step1 q={A} (block width 2^1=2), step2 q={A,B}
        (width 2^2=4), with four single-step anchors per attribute for identification. The
        per-step widths and n_parameters must reflect the distinct step Qs, NOT a single
        union block. A large B-contrast in step 2 (s2(A1,B0)=0.20 vs s2(A1,B1)=0.80,
        gap 0.60) is recovered (gap >= 0.4) while the union stays lossless. Value recovery
        alone can't catch an over-collapse to the union; the width assertions can.
    (3) VALIDATION -- all-zero step row (a step measuring nothing), an attribute used by no
        step (all-zero union column), and n_steps != observed max are all rejected."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_seq_gdina, fit_seq_gdina_qr, SeqGdinaQrFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_seq_gdina_qr"):
        pytest.skip("compiled core built without fit_seq_gdina_qr")

    rng = np.random.default_rng(2016_3)
    k, n = 2, 6000
    # 4 single-attribute M=1 items per attribute (identification) + 4 shared-Q M=2 pairs.
    rows = [[1, 0]] * 4 + [[0, 1]] * 4 + [[1, 1]] * 4
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    pair_s1 = {0: 0.25, 1: 0.55, 2: 0.50, 3: 0.85}
    pair_s2 = {0: 0.15, 1: 0.30, 2: 0.25, 3: 0.70}
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.zeros((n, n_items))
    for j in range(n):
        c = int(profiles[j])
        for i in range(n_items):
            if i < 8:
                a = i // 4
                p = 0.85 if (c >> a) & 1 else 0.15
                y[j, i] = 1.0 if rng.random() < p else 0.0
            else:
                l = (c & 1) + 2 * ((c >> 1) & 1)
                cat = 0
                if rng.random() < pair_s1[l]:
                    cat = 1
                    if rng.random() < pair_s2[l]:
                        cat = 2
                y[j, i] = float(cat)

    # (1) Shared-Q reduction: expand item Q into per-step rows sharing the item Q.
    n_steps = np.array([1] * 8 + [2] * 4, dtype=np.int64)
    step_rows = []
    for i in range(n_items):
        for _ in range(int(n_steps[i])):
            step_rows.append(q[i])
    step_q = np.vstack(step_rows).astype(np.int64)

    sh = fit_seq_gdina(y, q, tol=1e-8)
    qr = fit_seq_gdina_qr(y, step_q, n_steps, tol=1e-8)
    assert isinstance(qr, SeqGdinaQrFit) and qr.converged
    assert qr.termination_reason == "tolerance_met"
    assert qr.max_cat.tolist() == [1] * 8 + [2] * 4
    # cat_prob and loglik_trace are class-major: direct bit-exact compare.
    assert sh.cat_prob.shape == qr.cat_prob.shape
    assert np.array_equal(sh.cat_prob, qr.cat_prob)
    assert len(sh.loglik_trace) == len(qr.loglik_trace)
    assert np.array_equal(sh.loglik_trace, qr.loglik_trace)
    # step_prob layouts are transposed: cell-by-cell exact-zero difference.
    for i in range(n_items):
        Mi = int(n_steps[i])
        width = 1 << int(q[i].sum())
        for l in range(width):
            for kk in range(1, Mi + 1):
                sh_val = sh.step_prob[int(sh.s_off[i]) + l * Mi + (kk - 1)]
                qr_val = qr.item_step_prob(i, kk)[l]
                assert sh_val == qr_val, f"item{i} l{l} k{kk}: {sh_val} vs {qr_val}"

    # (2) Structure: distinct per-step Qs the shared-Q model cannot represent. Four
    #     polytomous items have step1={A}, step2={A,B}; four M=1 anchors per attribute
    #     identify both dimensions and avoid treating a likelihood ridge as recovery.
    n_pair2, n_anchor2 = 4, 4
    step_q2 = np.array(
        [[1, 0], [1, 1]] * n_pair2
        + [[1, 0]] * n_anchor2
        + [[0, 1]] * n_anchor2,
        dtype=np.int64,
    )
    n_steps2 = np.array([2] * n_pair2 + [1] * (2 * n_anchor2), dtype=np.int64)
    s2_by_class = {0: 0.15, 1: 0.20, 2: 0.30, 3: 0.80}  # big B-contrast at A=1
    n2 = 4000
    al2 = rng.integers(0, 2, size=(n2, k))
    Y2 = np.zeros((n2, n_pair2 + 2 * n_anchor2))
    for j in range(n2):
        a0, a1 = int(al2[j, 0]), int(al2[j, 1])
        for i in range(n_pair2):
            if rng.random() < (0.25 + 0.5 * a0):
                Y2[j, i] = 1
                rcAB = a0 + 2 * a1
                if rng.random() < s2_by_class[rcAB]:
                    Y2[j, i] = 2
        for i in range(n_pair2, n_pair2 + n_anchor2):
            Y2[j, i] = 1.0 if rng.random() < (0.2 + 0.6 * a0) else 0.0
        for i in range(n_pair2 + n_anchor2, Y2.shape[1]):
            Y2[j, i] = 1.0 if rng.random() < (0.2 + 0.6 * a1) else 0.0
    qr2 = fit_seq_gdina_qr(Y2, step_q2, n_steps2, max_iter=1000, tol=1e-6)
    assert qr2.converged and qr2.termination_reason == "tolerance_met"
    assert qr2.n_iter < 1000
    assert abs(qr2.final_loglik_change) < qr2.stopping_tolerance
    assert np.isfinite(qr2.loglik_trace).all()
    assert np.min(np.diff(qr2.loglik_trace)) >= -1e-8
    # per-step block widths reflect the distinct Qs (2 and 4), not a single union block.
    assert len(qr2.item_step_prob(0, 1)) == 2
    assert len(qr2.item_step_prob(0, 2)) == 4
    assert qr2.step_kq.tolist() == [1, 2] * n_pair2 + [1] * (2 * n_anchor2)
    # n_parameters = total step cells + (2^K - 1) free profile weights.
    assert qr2.n_parameters == n_pair2 * (2 + 4) + 2 * n_anchor2 * 2 + ((1 << k) - 1)
    # large B-contrast recovered in step 2 (class A1B1 minus A1B0).
    s2 = qr2.item_step_prob(0, 2)
    assert s2[3] - s2[1] >= 0.4, f"B-gap too small: {s2}"
    # category space: P(X=2 | A1B1) >> P(X=2 | A1B0), P(X>=1) roughly equal.
    cp = qr2.cat_prob[int(qr2.cat_off[0]):int(qr2.cat_off[0]) + 4 * 3].reshape(4, 3)
    assert cp[3, 2] - cp[1, 2] >= 0.3
    assert abs((1 - cp[3, 0]) - (1 - cp[1, 0])) < 0.15  # P(X>=1) close across B
    # Four M=1 anchors per attribute identify both dimensions and support stable recovery.
    est = (qr2.attr_prob >= 0.5).astype(int)
    assert (est == al2).mean() > 0.75

    # (3) Validation.
    with pytest.raises(ValueError):
        fit_seq_gdina_qr(Y2.ravel(), step_q2, n_steps2)  # not 2-D
    zero_row = step_q2.copy()
    zero_row[1] = [0, 0]  # a step measuring nothing
    with pytest.raises(ValueError):
        fit_seq_gdina_qr(Y2, zero_row, n_steps2)
    dead_col = step_q2.copy()
    dead_col[:, 1] = 0  # attr B unused
    with pytest.raises(ValueError):
        fit_seq_gdina_qr(Y2, dead_col, n_steps2)
    with pytest.raises(ValueError, match="sum"):
        too_many_steps = n_steps2.copy()
        too_many_steps[0] = 3
        fit_seq_gdina_qr(Y2, step_q2, too_many_steps)  # wrapper: rows != sum(n_steps)
    # max observed category != declared n_steps -- reaches the Rust guard, NOT the wrapper's
    # row-count guard: keep sum(n_steps)=16 (matching step_q2's 16 rows) but let item0 declare
    # 2 steps while its data never reaches category 2.
    y_low = Y2.copy()
    y_low[y_low[:, 0] == 2, 0] = 1
    with pytest.raises(ValueError, match="max observed category"):
        fit_seq_gdina_qr(y_low, step_q2, n_steps2)
    ymiss = Y2.copy()
    ymiss[0, 0] = np.nan
    fm = fit_seq_gdina_qr(ymiss, step_q2, n_steps2, max_iter=1000)  # MAR: dropped, no crash
    assert np.isfinite(fm.loglik_trace[-1]) and abs(fm.profile_prob.sum() - 1.0) < 1e-9


def test_fit_crm_recovers_continuous_responses():
    """Continuous Response Model (Samejima, 1973): recover the item slope/intercept/
    residual-sd and the Samejima discrimination/difficulty from continuous bounded
    responses, plus the trait (continuous responses are information-rich)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_crm, CrmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_crm"):
        pytest.skip("compiled core built without fit_crm")

    rng = np.random.default_rng(1973)
    n_items, n = 15, 1500
    a_true = 0.8 + 0.05 * np.arange(n_items)
    d_true = -0.6 + 0.08 * np.arange(n_items)
    sigma_true = 0.6 + 0.02 * (np.arange(n_items) % 5)
    theta = rng.standard_normal(n)
    x = a_true * theta[:, None] + d_true + sigma_true * rng.standard_normal((n, n_items))
    z = 1.0 / (1.0 + np.exp(-x))  # in (0,1)

    res = fit_crm(z)
    assert isinstance(res, CrmFit) and res.converged
    assert res.termination_reason == "tolerance"
    assert res.final_delta <= res.stopping_tolerance
    assert res.n_iter + 1 == len(res.loglik_trace)
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert res.n_parameters == 3 * n_items
    assert np.all(res.slope > 0)  # reflection convention
    assert np.sqrt(np.mean((res.slope - a_true) ** 2)) < 0.15
    assert np.sqrt(np.mean((res.intercept - d_true) ** 2)) < 0.1
    assert np.sqrt(np.mean((res.resid_sd - sigma_true) ** 2)) < 0.1
    # Samejima re-parameterization
    assert np.sqrt(np.mean((res.discrimination - a_true / sigma_true) ** 2)) < 0.3
    assert np.sqrt(np.mean((res.difficulty - (-d_true / a_true)) ** 2)) < 0.2
    # trait recovery
    assert np.corrcoef(res.theta, theta)[0, 1] > 0.9

    limited = fit_crm(z, max_iter=1, tol=1e-12)
    assert not limited.converged
    assert limited.termination_reason == "max_iter"
    assert limited.n_iter == 1
    assert limited.final_delta > limited.stopping_tolerance

    # missing-at-random handling
    zm = z.copy()
    zm[rng.random(zm.shape) < 0.15] = np.nan
    assert fit_crm(zm).converged

    with pytest.raises(ValueError):
        fit_crm(z.ravel())  # responses not 2-D
    with pytest.raises(ValueError):
        fit_crm(np.full((4, 3), 1.5))  # outside (0,1)
    with pytest.raises(ValueError, match="at least one person"):
        fit_crm(np.empty((0, 3)))
    with pytest.raises(ValueError, match="at least one person"):
        fit_crm(np.empty((3, 0)))
    with pytest.raises(ValueError, match="max_iter"):
        fit_crm(z, max_iter=0)
    with pytest.raises(ValueError, match="1..=100000"):
        fit_crm(z, max_iter=100_001)
    with pytest.raises(ValueError, match="tol"):
        fit_crm(z, tol=np.nan)
    with pytest.raises(ValueError, match="no observed responses"):
        missing_item = z.copy()
        missing_item[:, 0] = np.nan
        fit_crm(missing_item)


def test_fit_rsm_recovers_shared_thresholds():
    """Rating Scale Model (Andrich, 1978): recover item locations and the shared
    category thresholds (centered) plus the trait; K=2 reduces to Rasch."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_rsm, RsmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rsm"):
        pytest.skip("compiled core built without fit_rsm")

    rng = np.random.default_rng(1978)
    n_items, n_cat, n = 12, 5, 2500
    delta_true = -1.2 + 0.2 * np.arange(n_items)
    tau_true = np.array([0.9, 0.2, -0.3, -0.8])  # sums to 0
    theta = rng.standard_normal(n)

    def draw(th, d):
        # cumulative psi_k = k*th - k*d - sum_{m<=k} tau
        tk = np.concatenate([[0.0], np.cumsum(tau_true)])
        psi = np.arange(n_cat) * th - np.arange(n_cat) * d - tk
        p = np.exp(psi - psi.max())
        p /= p.sum()
        return rng.choice(n_cat, p=p)

    y = np.array([[draw(theta[j], delta_true[i]) for i in range(n_items)] for j in range(n)],
                 dtype=float)

    res = fit_rsm(y, q_theta=41, max_iter=500, tol=1e-6)
    assert isinstance(res, RsmFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert res.n_parameters == n_items + n_cat - 2
    assert abs(res.thresholds.sum()) < 1e-6  # centered
    assert np.sqrt(np.mean((res.item_location - delta_true) ** 2)) < 0.15
    assert np.sqrt(np.mean((res.thresholds - tau_true) ** 2)) < 0.12
    assert np.corrcoef(res.theta, theta)[0, 1] > 0.85

    # missing-at-random
    ym = y.copy()
    ym[rng.random(ym.shape) < 0.15] = np.nan
    assert fit_rsm(ym, q_theta=41, max_iter=500, tol=1e-6).converged

    with pytest.raises(ValueError):
        fit_rsm(y.ravel(), q_theta=41, max_iter=500, tol=1e-6)  # not 2-D


def test_fit_rsm_rejects_unidentified_or_malformed_inputs():
    """RSM must not report convergence for data that cannot identify a fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_rsm
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rsm"):
        pytest.skip("compiled core built without fit_rsm")

    valid = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="at least one person"):
        fit_rsm(np.empty((0, 2)), n_cat=2, q_theta=41, max_iter=500, tol=1e-6)
    with pytest.raises(ValueError, match="at least one person"):
        fit_rsm(np.empty((2, 0)), n_cat=2, q_theta=41, max_iter=500, tol=1e-6)
    with pytest.raises(ValueError, match="integer categories"):
        fit_rsm(np.array([[0.2, 1.0], [1.0, 0.0]]), n_cat=2, q_theta=41, max_iter=500, tol=1e-6)
    with pytest.raises(ValueError, match="finite integer categories"):
        fit_rsm(np.array([[0.0, np.inf], [1.0, 0.0]]), n_cat=2, q_theta=41, max_iter=500, tol=1e-6)
    with pytest.raises(ValueError, match="item 1 has no observed responses"):
        fit_rsm(np.array([[0.0, np.nan], [1.0, np.nan]]), n_cat=2, q_theta=41, max_iter=500, tol=1e-6)
    with pytest.raises(ValueError, match="max_iter"):
        fit_rsm(valid, n_cat=2, max_iter=0, q_theta=41, tol=1e-6)
    with pytest.raises(ValueError, match="tol"):
        fit_rsm(valid, n_cat=2, tol=np.inf, q_theta=41, max_iter=500)

    unfinished = fit_rsm(valid, n_cat=2, max_iter=1, q_theta=41, tol=1e-6)
    assert not unfinished.converged
    assert unfinished.n_iter == 1
    assert len(unfinished.loglik_trace) == 2
    assert np.all(np.isfinite(unfinished.loglik_trace))


def test_fit_2pl_recovers_confirmatory_loadings():
    """Compensatory MIRT (Reckase, 2009): recover a confirmatory 2-dimensional loading
    pattern (dim0-only, dim1-only, and BOTH-loading items) including a genuinely NEGATIVE
    loading, plus the per-dimension trait EAP; and reject a rotationally-degenerate
    (all-ones) pattern."""
    import numpy as np
    import pytest
    from fast_mlsirm import TwoPlFit, fit_2pl, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_2pl"):
        pytest.skip("compiled core built without fit_2pl")

    rng = np.random.default_rng(2009)
    n, n_dims = 4000, 2
    pattern = np.array([[1, 0]] * 4 + [[0, 1]] * 4 + [[1, 1]] * 3, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    loading[:4, 0] = [1.2, 0.8, 1.5, -0.9]  # dim0-only, incl. a negative loading
    loading[4:8, 1] = [1.0, 1.3, 0.7, 1.1]  # dim1-only
    loading[8:] = [[0.9, 1.1], [1.2, -0.7], [0.8, 0.9]]  # both, incl. a negative cross-loading
    intercept = np.linspace(-0.8, 0.8, n_items)
    theta = rng.standard_normal((n, n_dims))
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = fit_2pl(y, model=models.confirmatory(pattern), q=21, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert isinstance(res, TwoPlFit) and res.converged
    assert res.loading.shape == (n_items, n_dims) and res.n_dims == 2
    # off-pattern entries are exactly zero
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert np.sqrt(np.mean((res.loading - loading) ** 2)) < 0.13
    # negative loadings recovered with the correct sign (needs the symmetric clamp)
    assert res.loading[3, 0] < -0.5
    assert res.loading[9, 1] < -0.3
    # per-dimension trait recovery (positive correlation; a sign/swap bug -> near 0 or negative)
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.7, f"dim {d} theta corr {c}"
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # EM monotone
    assert res.termination_reason == "converged"
    assert res.n_iter < 500
    assert np.isfinite(res.final_loglik_change)
    assert res.final_loglik_change < 1e-6

    # a rotationally-degenerate all-ones pattern is rejected (no pure anchor per dimension)
    with pytest.raises(ValueError):
        fit_2pl(y, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    fractional = pattern.astype(float)
    fractional[0, 1] = 0.5
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        fit_2pl(y, model=models.confirmatory(fractional), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError, match="q must be a finite integer"):
        fit_2pl(y, model=models.confirmatory(pattern), q=15.5, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError, match="max_iter must be a finite integer"):
        fit_2pl(y, model=models.confirmatory(pattern), max_iter=1.5, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    # missing (MAR) handled
    ymiss = y.copy()
    ymiss[0, 0] = np.nan
    assert fit_2pl(ymiss, model=models.confirmatory(pattern), q=15, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15).converged

    # A one-step run that has not met the documented tolerance is explicitly unfinished.
    unfinished = fit_2pl(y, model=models.confirmatory(pattern), q=7, max_iter=1, tol=1e-12, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert not unfinished.converged
    assert unfinished.termination_reason == "max_iter_reached"
    assert unfinished.n_iter == 1
    assert len(unfinished.loglik_trace) == 2
    assert unfinished.final_loglik_change >= 1e-12

    # estimate_corr=False reports Sigma = I; estimate_corr=True recovers a known correlation.
    ortho = fit_2pl(y, model=models.confirmatory(pattern), q=15, estimate_corr=False, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert np.allclose(ortho.corr, np.eye(n_dims))
    ncorr = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    thc = (ncorr @ rng.standard_normal((n_dims, n))).T
    pc = 1.0 / (1.0 + np.exp(-(thc @ loading.T + intercept)))
    yc = (rng.random((n, n_items)) < pc).astype(float)
    rc = fit_2pl(yc, model=models.confirmatory(pattern), q=15, estimate_corr=True, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert rc.corr.shape == (n_dims, n_dims)
    assert np.allclose(np.diag(rc.corr), 1.0) and np.allclose(rc.corr, rc.corr.T)
    realized = np.corrcoef(thc.T)[0, 1]
    assert abs(rc.corr[0, 1] - realized) < 0.06, f"corr {rc.corr[0, 1]} vs realized {realized}"
    assert np.all(np.linalg.eigvalsh(rc.corr) > 0)  # positive-definite


def test_fit_2pl_qmc_high_dim():
    """QMC compensatory MIRT (Jank, 2005): the D>3 quasi-Monte-Carlo path the Gauss-Hermite
    product grid cannot reach. Recovers a D=4 confirmatory loading pattern (2 pure anchors per
    dimension + cross-loaders including a genuine NEGATIVE one) on Halton nodes; confirms the GH
    path still caps D<=3 while QMC/MC reach D<=6; and checks the wrapper plumbing is two-sided
    (a D<=3 QMC fit agrees with GH within QMC error but is NOT a silent bit-identical GH fallback)."""
    import numpy as np
    import pytest
    from fast_mlsirm import TwoPlFit, fit_2pl, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_2pl"):
        pytest.skip("compiled core built without fit_2pl")

    rng = np.random.default_rng(2005)
    n, n_dims = 2500, 4
    # 2 pure anchors per dim + 3 cross-loaders (one with a negative dim-1 loading).
    rows = []
    for d in range(n_dims):
        rows += [[1 if k == d else 0 for k in range(n_dims)]] * 2
    rows += [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    pattern = np.array(rows, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        loading[2 * d, d] = 1.2 + 0.1 * d
        loading[2 * d + 1, d] = 0.9
    cross = 2 * n_dims
    loading[cross] = [1.0, -0.8, 0.0, 0.0]  # the negative cross-loader
    loading[cross + 1] = [0.0, 1.1, 0.7, 0.0]
    loading[cross + 2] = [0.0, 0.0, 0.8, 1.0]
    intercept = np.linspace(-0.5, 0.7, n_items)
    theta = rng.standard_normal((n, n_dims))
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    # GH cannot reach D=4; QMC (Halton) can.
    with pytest.raises(ValueError):
        fit_2pl(y, model=models.confirmatory(pattern), node_rule="gh", max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    res = fit_2pl(y, model=models.confirmatory(pattern), node_rule="qmc", xi_points=4000, xi_seed=12345, max_iter=500, tol=1e-6)
    assert isinstance(res, TwoPlFit) and res.n_dims == 4
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert np.sqrt(np.mean((res.loading - loading) ** 2)) < 0.18
    assert res.loading[cross, 1] < -0.3  # negative cross-loader recovered with sign
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.55, f"dim {d} theta corr {c}"
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # EM monotone

    # node_rule validation and D<=6 bounds.
    with pytest.raises(ValueError, match="node_rule"):
        fit_2pl(y, model=models.confirmatory(pattern), node_rule="nope", max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    pat7 = np.eye(7, dtype=np.int64)
    y7 = (rng.random((200, 7)) < 0.5).astype(float)
    with pytest.raises(ValueError):
        fit_2pl(y7, model=models.confirmatory(pat7), node_rule="qmc", xi_points=200, max_iter=500, tol=1e-6, xi_seed=0x9E3779B97F4A7C15)  # D=7 > 6

    # Two-sided wrapper plumbing at D=2: GH and QMC agree within QMC error yet differ bit-wise
    # (a silent GH fallback on the QMC arm would make them identical).
    pat2 = np.array([[1, 0]] * 3 + [[0, 1]] * 3 + [[1, 1]], dtype=np.int64)
    ld2 = np.zeros((7, 2))
    for i in range(3):
        ld2[i, 0] = 1.0 + 0.1 * i
        ld2[3 + i, 1] = 1.0
    ld2[6] = [0.9, 0.8]
    ic2 = np.linspace(-0.4, 0.5, 7)
    th2 = rng.standard_normal((2000, 2))
    p2 = 1.0 / (1.0 + np.exp(-(th2 @ ld2.T + ic2)))
    y2 = (rng.random((2000, 7)) < p2).astype(float)
    gh2 = fit_2pl(y2, model=models.confirmatory(pat2), q=21, node_rule="gh", max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    qmc2 = fit_2pl(y2, model=models.confirmatory(pat2), node_rule="qmc", xi_points=6000, xi_seed=0, max_iter=500, tol=1e-6)
    max_abs = max(
        np.max(np.abs(gh2.loading - qmc2.loading)),
        np.max(np.abs(gh2.intercept - qmc2.intercept)),
    )
    assert max_abs < 0.10, f"QMC vs GH beyond QMC error: {max_abs}"
    assert max_abs > 1e-10, "QMC fit bit-identical to GH (silent fallback?)"


def test_fit_mhrm_recovers_high_dimensional_2pl():
    """MH-RM (Cai, 2010): high-dimensional confirmatory 2PL by Metropolis-Hastings Robbins-Monro
    stochastic approximation. Recovers a D=6 confirmatory loading pattern — the q**D Gauss-Hermite
    grid (21**6 ~ 8.6e7) and even the QMC E-step are infeasible at this dimensionality, which is the
    module's reason to exist — including a genuine NEGATIVE cross-loader, with reflection-
    canonicalized signs, per-dimension trait recovery, finite Louis observed-information SEs, and a
    tuned acceptance rate; and rejects rotationally-degenerate patterns and non-binary responses."""
    import numpy as np
    import pytest
    from fast_mlsirm import MhrmFit, fit_mhrm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        pytest.skip("compiled core built without fit_mhrm")

    rng = np.random.default_rng(2010)
    n, n_dims = 3000, 6
    rows = []
    for d in range(n_dims):
        rows += [[1 if k == d else 0 for k in range(n_dims)]] * 3  # 3 pure anchors per dim
    cross = [0] * n_dims
    cross[0] = 1
    cross[3] = 1
    rows.append(cross)  # one cross-loader on dims 0 and 3
    pattern = np.array(rows, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        for a in range(3):
            loading[3 * d + a, d] = 0.9 + 0.1 * a
    xi = n_items - 1
    loading[xi, 0] = 1.0
    loading[xi, 3] = -0.7  # negative cross-loader
    intercept = np.linspace(-0.5, 0.6, n_items)
    theta = rng.standard_normal((n, n_dims))
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = fit_mhrm(y, model=models.confirmatory(pattern), max_cycles=1400, burn_in=280, mh_steps=8, seed=7, target_accept=0.3, tol=1e-3)
    assert isinstance(res, MhrmFit) and res.n_dims == 6
    assert res.loading.shape == (n_items, n_dims)
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert res.n_parameters == int(pattern.sum()) + n_items
    onpat = pattern == 1
    assert np.sqrt(np.mean((res.loading[onpat] - loading[onpat]) ** 2)) < 0.22
    assert res.loading[xi, 3] < -0.3  # negative cross-loader recovered with sign
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.5, f"dim {d} theta corr {c}"
    # Louis SEs: right shape, finite on-pattern
    assert res.se_loading.shape == (n_items, n_dims)
    assert np.all(np.isfinite(res.se_loading[onpat]))
    assert res.se_intercept.shape == (n_items,)
    # acceptance auto-tuned into a sane band
    assert 0.1 < res.acceptance_rate < 0.7, res.acceptance_rate

    # rotationally-degenerate pattern (every item loads all dims -> no pure anchor) rejected
    with pytest.raises(ValueError):
        fit_mhrm(y, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)), max_cycles=2000, burn_in=200, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)
    # non-binary response rejected
    with pytest.raises(ValueError):
        ybad = y.copy()
        ybad[0, 0] = 2
        fit_mhrm(ybad, model=models.confirmatory(pattern), max_cycles=2000, burn_in=200, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)


def test_fit_mhrm_reports_the_windowed_stopping_metric():
    """The reported metric must be the same window mean used to declare convergence."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_mhrm
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        pytest.skip("compiled core built without fit_mhrm")

    rng = np.random.default_rng(20260720)
    n, n_items = 100, 6
    theta = rng.normal(size=n)
    loading = np.array([0.7, 0.9, 1.1, 1.3, 0.8, 1.0])
    intercept = np.array([-0.8, -0.4, 0.0, 0.3, 0.6, -0.2])
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] * loading + intercept)))
    responses = (rng.random((n, n_items)) < prob).astype(float)
    tol = 0.005

    result = fit_mhrm(
        responses,
        model=1,
        max_cycles=150,
        burn_in=5,
        mh_steps=1,
        tol=tol,
        seed=52,
        estimate_se=False, target_accept=0.3
    )

    assert result.converged
    assert result.termination_reason == "converged"
    assert result.n_cycles == 110
    assert result.final_param_change < tol


def test_fit_mhrm_estimate_corr_recovers_factor_correlation():
    """MH-RM with estimate_corr (Cai, 2010b): recover a free latent factor CORRELATION at D=2 from
    theta ~ MVN(0, Phi), and confirm estimate_corr=False yields exactly the identity."""
    import numpy as np
    import pytest
    from fast_mlsirm import MhrmFit, fit_mhrm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        pytest.skip("compiled core built without fit_mhrm")

    rng = np.random.default_rng(2010)
    n, n_dims, rho = 3000, 2, 0.5
    per = 4
    n_items = per * n_dims
    pattern = np.zeros((n_items, n_dims), dtype=np.int64)
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        for a in range(per):
            pattern[d * per + a, d] = 1
            loading[d * per + a, d] = 1.0 + 0.1 * a
    intercept = np.linspace(-0.4, 0.5, n_items)
    phi = np.array([[1.0, rho], [rho, 1.0]])
    theta = rng.multivariate_normal(np.zeros(n_dims), phi, size=n)
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = fit_mhrm(y, model=models.confirmatory(pattern), max_cycles=1500, burn_in=320,
                   mh_steps=8, estimate_corr=True, seed=3, target_accept=0.3, tol=1e-3)
    assert isinstance(res, MhrmFit)
    assert res.corr.shape == (n_dims, n_dims)
    assert np.allclose(np.diag(res.corr), 1.0)
    assert abs(res.corr[0, 1] - rho) < 0.12, res.corr[0, 1]
    assert res.n_parameters == n_items + n_items + n_dims * (n_dims - 1) // 2

    # estimate_corr=False -> exactly the identity, and fewer parameters
    res0 = fit_mhrm(y, model=models.confirmatory(pattern), max_cycles=400, burn_in=100,
                    estimate_corr=False, seed=3, mh_steps=5, target_accept=0.3, tol=1e-3)
    assert np.array_equal(res0.corr, np.eye(n_dims))
    assert res0.n_parameters == n_items + n_items


def test_fit_mhrm_gpcm_recovers_high_dimensional_polytomous():
    """MH-RM GPCM (Muraki, 1992; Cai, 2010): high-dimensional confirmatory GENERALIZED PARTIAL CREDIT
    model by Metropolis-Hastings Robbins-Monro. Recovers a D=3 confirmatory pattern of loadings and
    UNORDERED step intercepts — the q**D Gauss-Hermite grid and the QMC E-step are infeasible for a
    polytomous item factor model at this dimensionality — including a genuinely NEGATIVE cross-loader,
    with reflection-canonicalized signs; exposes the family/n_cat/step result shape (intercept empty);
    and rejects out-of-range responses and a never-observed category (an unidentified step)."""
    import numpy as np
    import pytest
    from fast_mlsirm import MhrmFit, fit_mhrm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        pytest.skip("compiled core built without fit_mhrm")

    rng = np.random.default_rng(1992)
    n, n_dims, n_cat = 3000, 3, 3
    rows = []
    for d in range(n_dims):
        rows += [[1 if k == d else 0 for k in range(n_dims)]] * 3  # 3 pure anchors per dim
    cross = [0] * n_dims
    cross[0] = 1
    cross[2] = 1
    rows.append(cross)  # one cross-loader on dims 0 and 2
    pattern = np.array(rows, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        for a in range(3):
            loading[3 * d + a, d] = 0.9 + 0.1 * a
    xi = n_items - 1
    loading[xi, 0] = 1.0
    loading[xi, 2] = -0.7  # negative cross-loader
    # non-monotone (unordered) steps per item
    step = np.column_stack([
        0.7 - 0.12 * (np.arange(n_items) % 3),
        -0.4 + 0.1 * (np.arange(n_items) % 4),
    ])
    theta = rng.standard_normal((n, n_dims))
    base = theta @ loading.T  # (n, n_items), no intercept
    # psi_k = k*base + step_k (step_0 = 0); sample categories from the softmax
    ks = np.arange(n_cat)
    full_step = np.hstack([np.zeros((n_items, 1)), step])  # (n_items, n_cat)
    psi = base[:, :, None] * ks[None, None, :] + full_step[None, :, :]  # (n, n_items, n_cat)
    psi -= psi.max(axis=2, keepdims=True)
    prob = np.exp(psi)
    prob /= prob.sum(axis=2, keepdims=True)
    u = rng.random((n, n_items))
    y = (u[:, :, None] > np.cumsum(prob, axis=2)).sum(axis=2).astype(float)  # inverse-CDF draw

    res = fit_mhrm(y, model=models.confirmatory(pattern), family="gpcm", n_cat=n_cat,
                   max_cycles=1200, burn_in=250, mh_steps=6, seed=11, target_accept=0.3, tol=1e-3)
    assert isinstance(res, MhrmFit) and res.n_dims == n_dims
    assert res.family == "gpcm" and res.n_cat == n_cat
    assert res.loading.shape == (n_items, n_dims)
    assert res.step.shape == (n_items, n_cat - 1)
    assert res.intercept.size == 0  # 2PL intercept empty for GPCM
    assert res.se_step.shape == (n_items, n_cat - 1)
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert res.n_parameters == int(pattern.sum()) + n_items * (n_cat - 1)
    onpat = pattern == 1
    assert np.sqrt(np.mean((res.loading[onpat] - loading[onpat]) ** 2)) < 0.25
    assert res.loading[xi, 2] < -0.3  # negative cross-loader recovered with sign
    assert np.sqrt(np.mean((res.step - step) ** 2)) < 0.25
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.5, f"dim {d} theta corr {c}"

    # out-of-range response (== n_cat) rejected
    with pytest.raises(ValueError):
        ybad = y.copy()
        ybad[0, 0] = n_cat
        fit_mhrm(ybad, model=models.confirmatory(pattern), family="gpcm", n_cat=n_cat, max_cycles=2000, burn_in=200, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)
    # a declared category never observed for an item (unidentified step) rejected
    with pytest.raises(ValueError):
        ycov = y.copy()
        ycov[ycov[:, 0] == 1, 0] = 0  # item 0 never shows category 1
        fit_mhrm(ycov, model=models.confirmatory(pattern), family="gpcm", n_cat=n_cat, max_cycles=2000, burn_in=200, mh_steps=5, target_accept=0.3, tol=1e-3, seed=0x9E3779B97F4A7C15)


def test_fit_nominal_recovers_confirmatory_multidimensional_categories():
    """Confirmatory MULTIDIMENSIONAL nominal response model (Bock, 1972; Thissen-Cai-Bock, 2010):
    recover a D=2 confirmatory pattern of CATEGORY-SPECIFIC multidimensional slopes (unordered
    categories) including a genuinely NEGATIVE cross-loader slope with an OPPOSITE-sign sibling
    category on the same dimension (the signature a per-item RMSE would average away), assessed up
    to per-dimension reflection; confirm the baseline/off-pattern slopes are exactly zero; and
    reject rotationally-degenerate patterns, out-of-range and unobserved categories, and GH D>3."""
    import numpy as np
    import pytest
    from fast_mlsirm import NominalResponseFit, fit_nominal, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_nominal_model"):
        pytest.skip("compiled core built without fit_nominal_model")

    rng = np.random.default_rng(1972)
    n_dims, n_cat, n = 2, 3, 6000
    # items 0,1 pure dim0; items 2,3 pure dim1; item 4 cross-loader {0,1}.
    pattern = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int64)
    n_items = pattern.shape[0]
    anchor = [0, 2]  # pure anchor item per dim
    slope = np.zeros((n_items, n_cat, n_dims))
    slope[0, 1, 0], slope[0, 2, 0] = 1.4, 0.8
    slope[1, 1, 0], slope[1, 2, 0] = 1.0, 1.3
    slope[2, 1, 1], slope[2, 2, 1] = 1.2, 0.9
    slope[3, 1, 1], slope[3, 2, 1] = 1.1, 1.4
    slope[4, 1, 0], slope[4, 2, 0] = -1.1, 1.0   # negative + positive sibling on dim0
    slope[4, 1, 1], slope[4, 2, 1] = 0.9, 0.7
    intercept = np.zeros((n_items, n_cat))
    for i in range(n_items):
        for k in range(1, n_cat):
            intercept[i, k] = -0.2 + 0.15 * k - 0.05 * i
    theta = rng.standard_normal((n, n_dims))
    eta = np.zeros((n, n_items, n_cat))
    for k in range(1, n_cat):
        eta[:, :, k] = theta @ slope[:, k, :].T + intercept[:, k]
    ex = np.exp(eta - eta.max(axis=2, keepdims=True))
    probs = ex / ex.sum(axis=2, keepdims=True)
    u = rng.random((n, n_items))
    y = (probs.cumsum(axis=2) < u[:, :, None]).sum(axis=2)

    res = fit_nominal(y, n_cat, model=models.confirmatory(pattern), q=21, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert isinstance(res, NominalResponseFit) and res.converged
    assert res.slope.shape == (n_items, n_cat, n_dims) and res.n_dims == 2 and res.n_cat == 3
    # baseline category and off-pattern entries are EXACTLY zero
    assert np.all(res.slope[:, 0, :] == 0.0)
    for i in range(n_items):
        for d in range(n_dims):
            if pattern[i, d] == 0:
                assert np.all(res.slope[i, :, d] == 0.0)
    assert np.all(res.intercept[:, 0] == 0.0)
    # free-parameter count = sum_i (n_cat-1)*(|S_i|+1)
    assert res.n_parameters == 2 * 2 + 2 * 2 + 2 * 2 + 2 * 2 + 2 * 3

    # per-dimension reflection alignment to truth (same rule applied to est), then compare
    est = res.slope.copy()
    for d in range(n_dims):
        if est[anchor[d], 1, d] * slope[anchor[d], 1, d] < 0:
            est[:, :, d] = -est[:, :, d]
    assert np.sqrt(np.mean((est - slope) ** 2)) < 0.16
    # the negative cross-loader slope and its opposite-sign sibling recovered with the right signs
    assert est[4, 1, 0] < -0.4
    assert est[4, 2, 0] > 0.4
    # per-dim trait EAP correlation (sign-aligned)
    for d in range(n_dims):
        th = res.theta[:, d].copy()
        if res.slope[anchor[d], 1, d] * slope[anchor[d], 1, d] < 0:
            th = -th
        assert np.corrcoef(th, theta[:, d])[0, 1] > 0.6
    assert np.all(np.diff(res.loglik_trace) >= -1e-9)  # EM monotone

    # validation
    with pytest.raises(ValueError):  # GH cannot reach D=4
        pat4 = np.eye(4, dtype=np.int64)
        fit_nominal(np.zeros((50, 4), dtype=np.int64), n_cat, model=models.confirmatory(pat4), node_rule="gh", max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # no pure anchor for either dim
        fit_nominal(y, n_cat, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # category out of range
        ybad = y.copy()
        ybad[0, 0] = n_cat
        fit_nominal(ybad, n_cat, model=models.confirmatory(pattern), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # an unobserved category for an item
        ygap = y.copy()
        ygap[ygap[:, 0] == 2, 0] = 1
        fit_nominal(ygap, n_cat, model=models.confirmatory(pattern), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_grm_recovers_confirmatory_multidimensional_ordered_categories():
    """Confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson, 1995):
    recover a D=2 confirmatory pattern of item discrimination vectors (ORDERED categories) including a
    genuinely NEGATIVE cross-loader on a positively-anchored dimension; confirm the recovered
    thresholds are strictly ordered and the baseline reflection is canonicalized (pure anchors
    positive); and reject rotationally-degenerate patterns, out-of-range and unobserved categories,
    and GH D>3."""
    import numpy as np
    import pytest
    from fast_mlsirm import GrmFit, fit_grm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_grm"):
        pytest.skip("compiled core built without fit_grm")

    rng = np.random.default_rng(1969)
    n_dims, n_cat, n = 2, 3, 6000
    pattern = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int64)
    n_items = pattern.shape[0]
    slope = np.zeros((n_items, n_dims))
    slope[0, 0], slope[1, 0] = 1.4, 1.0
    slope[2, 1], slope[3, 1] = 1.2, 1.1
    slope[4, 0], slope[4, 1] = -1.0, 0.9   # negative cross-loader on dim0 (anchor item 0 positive)
    threshold = np.zeros((n_items, n_cat - 1))
    for i in range(n_items):
        threshold[i] = [1.1 + 0.05 * i, 0.05 * i - 0.9]  # strictly decreasing
    theta = rng.standard_normal((n, n_dims))
    # simulate via cumulative logits P(Y>=k)=sigmoid(base+beta_{k-1})
    y = np.zeros((n, n_items), dtype=np.int64)
    for i in range(n_items):
        base = theta @ slope[i]
        ge = 1.0 / (1.0 + np.exp(-(base[:, None] + threshold[i][None, :])))  # (n, n_cat-1)
        pk = np.zeros((n, n_cat))
        pk[:, 0] = 1.0 - ge[:, 0]
        for k in range(1, n_cat - 1):
            pk[:, k] = ge[:, k - 1] - ge[:, k]
        pk[:, n_cat - 1] = ge[:, n_cat - 2]
        pk = np.clip(pk, 1e-12, None)
        pk /= pk.sum(axis=1, keepdims=True)
        u = rng.random(n)
        y[:, i] = (pk.cumsum(axis=1) < u[:, None]).sum(axis=1)

    res = fit_grm(y, n_cat, model=models.confirmatory(pattern), q=21, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert isinstance(res, GrmFit) and res.converged
    assert res.slope.shape == (n_items, n_dims) and res.threshold.shape == (n_items, n_cat - 1)
    assert res.n_dims == 2 and res.n_cat == 3
    # off-pattern slopes exactly zero
    for i in range(n_items):
        for d in range(n_dims):
            if pattern[i, d] == 0:
                assert res.slope[i, d] == 0.0
    # free-parameter count = sum_i (|S_i| + (n_cat-1))
    assert res.n_parameters == 4 * (1 + 2) + (2 + 2)
    # recovered thresholds strictly decreasing on every item
    assert np.all(res.threshold[:, 0] > res.threshold[:, 1])
    # canonical: pure anchors positive; negative cross-loader recovered negative
    assert res.slope[0, 0] > 0.5 and res.slope[2, 1] > 0.5
    assert res.slope[4, 0] < -0.4, f"neg cross-loader {res.slope[4, 0]}"
    assert np.sqrt(np.mean((res.slope - slope) ** 2)) < 0.16
    for d in range(n_dims):
        assert np.corrcoef(res.theta[:, d], theta[:, d])[0, 1] > 0.6
    assert np.all(np.diff(res.loglik_trace) >= -1e-9)  # EM monotone

    # validation
    with pytest.raises(ValueError):  # GH D=4
        fit_grm((np.arange(200).reshape(50, 4) % n_cat).astype(np.int64), n_cat,
                model=models.confirmatory(np.eye(4, dtype=np.int64)), node_rule="gh", max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # no pure anchor
        fit_grm(y, n_cat, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # category out of range
        ybad = y.copy(); ybad[0, 0] = n_cat
        fit_grm(ybad, n_cat, model=models.confirmatory(pattern), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # unobserved category
        ygap = y.copy(); ygap[ygap[:, 0] == 1, 0] = 0
        fit_grm(ygap, n_cat, model=models.confirmatory(pattern), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_gpcm_recovers_confirmatory_multidimensional_adjacent_category():
    """Confirmatory MULTIDIMENSIONAL generalized partial credit model (Muraki, 1992): recover a D=2
    confirmatory pattern of item discrimination vectors (INTEGER-scored adjacent-category logits)
    including a genuinely NEGATIVE cross-loader on a positively-anchored dimension; recover the
    UNORDERED category step intercepts numerically (GPCM steps carry no ordering constraint, so a
    monotone canary would be vacuous — RMSE is the only guard); confirm the baseline reflection is
    canonicalized (pure anchors positive) while steps are left unflipped; and reject
    rotationally-degenerate patterns, out-of-range and unobserved categories, and GH D>3."""
    import numpy as np
    import pytest
    from fast_mlsirm import GpcmFit, fit_gpcm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gpcm"):
        pytest.skip("compiled core built without fit_gpcm")

    rng = np.random.default_rng(1992)
    n_dims, n_cat, n = 2, 4, 6000
    pattern = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int64)
    n_items = pattern.shape[0]
    slope = np.zeros((n_items, n_dims))
    slope[0, 0], slope[1, 0] = 1.4, 1.0
    slope[2, 1], slope[3, 1] = 1.2, 1.1
    slope[4, 0], slope[4, 1] = -1.0, 0.9   # negative cross-loader on dim0 (anchor item 0 positive)
    # UNORDERED step intercepts gamma_k (psi_k = k*base + gamma_k, gamma_0 = 0); deliberately
    # non-monotone across k to exercise the free-step estimator.
    step = np.array([
        [0.7, -0.4, 0.9],
        [-0.3, 0.6, 0.1],
        [0.5, 0.2, -0.6],
        [-0.2, 0.8, -0.3],
        [0.4, -0.5, 0.7],
    ])
    theta = rng.standard_normal((n, n_dims))
    # simulate via adjacent-category softmax P(Y=k) = softmax_k(k*base + gamma_k)
    y = np.zeros((n, n_items), dtype=np.int64)
    for i in range(n_items):
        base = theta @ slope[i]
        psi = np.zeros((n, n_cat))
        for k in range(1, n_cat):
            psi[:, k] = k * base + step[i, k - 1]
        psi -= psi.max(axis=1, keepdims=True)
        pk = np.exp(psi)
        pk /= pk.sum(axis=1, keepdims=True)
        u = rng.random(n)
        y[:, i] = (pk.cumsum(axis=1) < u[:, None]).sum(axis=1)

    res = fit_gpcm(y, n_cat, model=models.confirmatory(pattern), q=21, max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    assert isinstance(res, GpcmFit) and res.converged
    assert res.slope.shape == (n_items, n_dims) and res.step.shape == (n_items, n_cat - 1)
    assert res.n_dims == 2 and res.n_cat == 4
    # off-pattern slopes exactly zero
    for i in range(n_items):
        for d in range(n_dims):
            if pattern[i, d] == 0:
                assert res.slope[i, d] == 0.0
    # free-parameter count = sum_i (|S_i| + (n_cat-1))
    assert res.n_parameters == 4 * (1 + 3) + (2 + 3)
    # canonical: pure anchors positive; negative cross-loader recovered negative
    assert res.slope[0, 0] > 0.5 and res.slope[2, 1] > 0.5
    assert res.slope[4, 0] < -0.4, f"neg cross-loader {res.slope[4, 0]}"
    assert np.sqrt(np.mean((res.slope - slope) ** 2)) < 0.16
    # UNORDERED steps recovered numerically (no ordering canary possible for GPCM)
    assert np.sqrt(np.mean((res.step - step) ** 2)) < 0.16, f"step RMSE {res.step}"
    for d in range(n_dims):
        assert np.corrcoef(res.theta[:, d], theta[:, d])[0, 1] > 0.6
    assert np.all(np.diff(res.loglik_trace) >= -1e-9)  # EM monotone

    # validation
    with pytest.raises(ValueError):  # GH D=4
        fit_gpcm((np.arange(200).reshape(50, 4) % n_cat).astype(np.int64), n_cat,
                 model=models.confirmatory(np.eye(4, dtype=np.int64)), node_rule="gh", max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # no pure anchor
        fit_gpcm(y, n_cat, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # category out of range
        ybad = y.copy(); ybad[0, 0] = n_cat
        fit_gpcm(ybad, n_cat, model=models.confirmatory(pattern), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)
    with pytest.raises(ValueError):  # unobserved category
        ygap = y.copy(); ygap[ygap[:, 0] == 1, 0] = 0
        fit_gpcm(ygap, n_cat, model=models.confirmatory(pattern), max_iter=500, tol=1e-6, xi_points=4000, xi_seed=0x9E3779B97F4A7C15)


def test_fit_mixture_recovers_two_class_rasch():
    """Mixed Rasch / mixture IRT (Rost, 1990): recover two latent classes with a
    difficulty reversal (a single-class model cannot fit both orderings)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_mixture, MixtureFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mixture"):
        pytest.skip("compiled core built without fit_mixture")

    rng = np.random.default_rng(1990)
    n, j, pi_true = 1500, 15, 0.6
    b0 = np.linspace(-2.0, 2.0, j)
    # class 0: b0; class 1: -b0 (reversal). theta ~ N(0,1).
    cls = (rng.random(n) >= pi_true).astype(int)  # 0 w.p. pi_true
    theta = rng.standard_normal(n)
    y = np.empty((n, j))
    for p in range(n):
        b = b0 if cls[p] == 0 else -b0
        y[p] = (rng.random(j) < 1 / (1 + np.exp(-(theta[p] + b)))).astype(float)

    res = fit_mixture(y, n_classes=2, model="rasch", n_starts=8, seed=123, max_iter=500, tol=1e-6)
    assert isinstance(res, MixtureFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert res.a.shape == (2, j) and np.allclose(res.a, 1.0)  # Rasch: a == 1
    assert res.n_parameters == 2 * j + 1  # 2 classes * J difficulties + (C-1)

    # permutation-match the two fitted classes to (b0, -b0) by difficulty SSE
    b_true = np.stack([b0, -b0])
    sse = lambda perm: float(np.sum((res.b[list(perm)] - b_true) ** 2))
    perm = (0, 1) if sse((0, 1)) <= sse((1, 0)) else (1, 0)
    brmse = np.sqrt(np.mean((res.b[list(perm)] - b_true) ** 2))
    assert brmse < 0.25, f"matched b RMSE {brmse}"
    assert abs(res.pi[perm[0]] - pi_true) < 0.06, f"pi {res.pi[perm[0]]}"

    # Adjusted Rand Index (label-invariant) between MAP class and truth
    def ari(a, b):
        from itertools import product
        ka, kb = a.max() + 1, b.max() + 1
        tab = np.zeros((ka, kb))
        for x, yv in zip(a, b):
            tab[x, yv] += 1
        c2 = lambda m: m * (m - 1) / 2
        idx = sum(c2(tab[i, k]) for i, k in product(range(ka), range(kb)))
        sa = sum(c2(tab[i].sum()) for i in range(ka))
        sb = sum(c2(tab[:, k].sum()) for k in range(kb))
        exp = sa * sb / c2(len(a))
        return (idx - exp) / (0.5 * (sa + sb) - exp)

    assert ari(res.map_class, cls) > 0.35, "class recovery (ARI) too low"

    with pytest.raises(ValueError):
        fit_mixture(y.ravel(), n_classes=2, model='rasch', n_starts=1, max_iter=500, tol=1e-6, seed=0x2545F491)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_mixture(y, n_classes=2, model="graded", n_starts=1, max_iter=500, tol=1e-6, seed=0x2545F491)  # unknown within-class model


def test_fit_lltm_recovers_basic_parameters():
    """LLTM (Fischer, 1973): recover the basic cognitive-operation parameters from the
    design matrix, and the LR test does not reject when the restriction holds."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_lltm, LltmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_lltm"):
        pytest.skip("compiled core built without fit_lltm")

    rng = np.random.default_rng(1973)
    n, j, k = 2500, 20, 5
    # full-rank design with varying row sums (column k has period k+2)
    q = np.array([[(i + kk) % (kk + 2) for kk in range(k)] for i in range(j)], dtype=float)
    eta_true = np.array([0.6, -0.4, 0.9, -0.5, 0.3])
    c_true = -0.2
    b_true = c_true + q @ eta_true
    theta = rng.standard_normal(n)
    y = (rng.random((n, j)) < 1 / (1 + np.exp(-(theta[:, None] + b_true[None, :])))).astype(float)

    res = fit_lltm(y, q, max_iter=500, tol=1e-6)
    assert isinstance(res, LltmFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert res.eta.shape == (k,) and res.b.shape == (j,)
    assert np.corrcoef(res.eta, eta_true)[0, 1] > 0.95
    assert np.sqrt(np.mean((res.b - b_true) ** 2)) < 0.15
    assert res.n_parameters == k + 1  # K basic + intercept
    # LR test: LLTM restriction HOLDS, so it should not reject
    assert res.lr_df == j - k - 1
    assert res.lr_p > 0.01, f"LR falsely rejected true LLTM: p={res.lr_p}"

    with pytest.raises(ValueError):
        fit_lltm(y.ravel(), q, max_iter=500, tol=1e-6)  # responses not 2-D
    with pytest.raises(ValueError):
        # rows sum to a constant + intercept => rank-deficient design, rejected
        fit_lltm(y, np.ones((j, 1)), max_iter=500, tol=1e-6)


def test_fit_testlet_recovers_local_dependence():
    """Testlet model (Bradlow, Wainer, & Wang, 1999): recover the per-testlet variance
    (local dependence), and confirm sigma^2=0 reduces to the ordinary Rasch/2PL fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_testlet, TestletFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_testlet"):
        pytest.skip("compiled core built without fit_testlet")

    rng = np.random.default_rng(1999)
    n, per, d = 800, 8, 2
    j = per * d
    tid = np.repeat(np.arange(d), per)  # contiguous testlets
    sig2 = np.array([0.6, 0.3])
    beta = np.tile(np.linspace(-1.5, 1.5, per), d)  # Rasch: b = -beta (a=1)
    theta = rng.standard_normal(n)
    gamma = rng.standard_normal((n, d)) * np.sqrt(sig2)[None, :]
    y = np.empty((n, j))
    for p in range(n):
        eta = theta[p] + beta - gamma[p, tid]
        y[p] = (rng.random(j) < 1 / (1 + np.exp(-eta))).astype(float)

    res = fit_testlet(y, tid, model="rasch")
    assert isinstance(res, TestletFit) and res.converged
    assert res.termination_reason == "converged"
    assert res.final_loglik_change < 1e-6
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert np.all(res.a == 1.0)  # Rasch
    assert res.sigma2.shape == (d,)
    # the strong-LD testlet is recovered as clearly larger than the weak one
    assert res.sigma2[0] > 0.35 and res.sigma2[0] > res.sigma2[1]
    assert np.sqrt(np.mean((res.sigma2 - sig2) ** 2)) < 0.2

    # sigma^2 pinned to 0 => ordinary Rasch (no local dependence modeled)
    res0 = fit_testlet(y, tid, model="rasch", estimate_sigma=False, init_sigma2=0.0)
    assert np.all(res0.sigma2 == 0.0)
    assert res0.n_parameters == j  # fixed variances are not free parameters

    with pytest.raises(ValueError):
        fit_testlet(y.ravel(), tid)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_testlet(y, tid, model="graded")  # unknown model

    with pytest.raises(RuntimeError, match="max_iter_reached"):
        fit_testlet(y[:40], tid, model="rasch", max_iter=1, require_convergence=True)


def test_fit_facets_recovers_rater_severity():
    """Many-facet Rasch model (Linacre, 1989): recover asymmetric rater
    severities, item difficulties, and shared thresholds from a sparse judging
    plan; single-rater case must agree with fit_rsm (RSM reduction).

    Asserts read crate outputs (res.rater_severity / item_difficulty /
    thresholds / theta / loglik_trace / n_parameters / connected). A severity
    sign-flip or a d/c dimension-map swap in the Rust core fails the recovery
    and reduction checks."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_facets, fit_rsm, FacetsFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_facets"):
        pytest.skip("compiled core built without fit_facets")

    rng = np.random.default_rng(1989)
    n, n_items, n_raters, n_cat = 1500, 8, 4, 4
    d_true = -0.9 + 0.25 * np.arange(n_items)
    c_true = np.array([1.1, -0.2, -0.4, -0.5])  # asymmetric, sums to 0
    f_true = np.array([0.7, 0.1, -0.8])  # sums to 0
    theta = rng.standard_normal(n)
    tk = np.concatenate([[0.0], np.cumsum(f_true)])
    ks = np.arange(n_cat)

    y = np.full((n, n_items, n_raters), np.nan)
    for p in range(n):
        for i in range(n_items):
            for j in range(n_raters):
                if rng.random() < 0.4:  # sparse plan
                    continue
                psi = ks * theta[p] - ks * (d_true[i] + c_true[j]) - tk
                pr = np.exp(psi - psi.max())
                y[p, i, j] = rng.choice(n_cat, p=pr / pr.sum())

    res = fit_facets(y, n_cat=n_cat)
    assert isinstance(res, FacetsFit) and res.converged and res.connected
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert res.n_parameters == n_items + (n_raters - 1) + (n_cat - 2)
    assert abs(res.rater_severity.sum()) < 1e-6
    assert abs(res.thresholds.sum()) < 1e-6
    assert np.sqrt(np.mean((res.rater_severity - c_true) ** 2)) < 0.12
    assert np.sqrt(np.mean((res.item_difficulty - d_true) ** 2)) < 0.15
    assert np.sqrt(np.mean((res.thresholds - f_true) ** 2)) < 0.12
    assert np.corrcoef(res.theta, theta)[0, 1] > 0.85

    # single-rater reduction: MFRM with J=1 must match RSM (severity absorbed)
    y1 = y[:400, :, :1]
    keep = ~np.isnan(y1).all(axis=(1, 2))
    y1 = y1[keep]
    r_f = fit_facets(y1, n_cat=n_cat)
    r_r = fit_rsm(y1[:, :, 0], n_cat=n_cat, q_theta=41, max_iter=500, tol=1e-6)
    assert np.allclose(r_f.rater_severity, [0.0])
    assert np.allclose(r_f.item_difficulty, r_r.item_location, atol=5e-3)
    assert np.allclose(r_f.thresholds, r_r.thresholds, atol=5e-3)


def test_fit_facets_rejects_malformed_and_flags_disconnected():
    """MFRM input validation plus Linacre's connectedness diagnostic: a judging
    plan whose item-rater graph splits into components must set connected=False
    (asserts read res.connected from the crate). The False assert kills a
    deleted union-find pass; the True assert on the bridged plan kills the
    join-only-item-item-edges mutant, which would leave rater nodes isolated
    and report every design as disconnected."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_facets
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_facets"):
        pytest.skip("compiled core built without fit_facets")

    valid = np.zeros((4, 2, 2))
    valid[::2] = 1.0
    with pytest.raises(ValueError, match="3-D"):
        fit_facets(np.zeros((4, 2)), n_cat=2)
    with pytest.raises(ValueError, match="at least one person"):
        fit_facets(np.empty((0, 2, 2)), n_cat=2)
    with pytest.raises(ValueError, match="integer categories"):
        fit_facets(valid + 0.5, n_cat=2)
    with pytest.raises(ValueError, match="in 0..1"):
        fit_facets(valid * 3, n_cat=2)
    with pytest.raises(ValueError, match="rater 1 has no observed"):
        bad = valid.copy()
        bad[:, :, 1] = np.nan
        fit_facets(bad, n_cat=2)
    # #1929: no node-count cap; q_theta=10 is now accepted, only < 1 is not.
    with pytest.raises(ValueError, match="q_theta"):
        fit_facets(valid, n_cat=2, q_theta=0)
    with pytest.raises(ValueError, match="max_iter"):
        fit_facets(valid, n_cat=2, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        fit_facets(valid, n_cat=2, tol=0.0)

    # disconnected plan: persons 0-19 see (item0, rater0), persons 20-39 see
    # (item1, rater1) -- no shared element links the two components
    rng = np.random.default_rng(7)
    y = np.full((40, 2, 2), np.nan)
    y[:20, 0, 0] = rng.integers(0, 2, 20)
    y[20:, 1, 1] = rng.integers(0, 2, 20)
    y[0, 0, 0], y[1, 0, 0] = 0.0, 1.0
    y[20, 1, 1], y[21, 1, 1] = 0.0, 1.0
    res = fit_facets(y, n_cat=2, max_iter=50)
    assert res.connected is False

    # bridged plan: person 5 also sees (item1, rater1), joining the components
    yb = y.copy()
    yb[5, 1, 1] = 1.0
    yb[6, 1, 1] = 0.0
    resb = fit_facets(yb, n_cat=2, max_iter=50)
    assert resb.connected is True


def test_mokken_analysis_coefficients_and_cluster_recovery():
    """Mokken scale analysis (van der Ark, 2007): every assert reads crate
    outputs (hij/hi/h/zij/scale from mokken_coef_h / mokken_aisp via the
    wrapper). A perfect Guttman scalogram must give H = 1 exactly (kills
    covmax mutants), and a two-cluster simulation must be partitioned into
    exactly two AISP scales (kills selection-logic mutants)."""
    import numpy as np
    import pytest
    from fast_mlsirm import MokkenResult, mokken_analysis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mokken_coef_h"):
        pytest.skip("compiled core built without mokken_coef_h")

    # perfect Guttman scalogram -> H exactly 1
    guttman = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=float
    )
    g = mokken_analysis(guttman)
    assert isinstance(g, MokkenResult)
    assert abs(g.h - 1.0) < 1e-12
    off = ~np.eye(3, dtype=bool)
    assert np.allclose(g.hij[off], 1.0)
    assert np.all(np.isnan(np.diag(g.hij)))

    # two independent Rasch clusters -> two scales
    rng = np.random.default_rng(2013)
    n, per = 1500, 4
    bs = np.array([-0.8, -0.3, 0.3, 0.8])
    t1 = rng.normal(size=(n, 1)) * 1.6
    t2 = rng.normal(size=(n, 1)) * 1.6
    xa = (rng.random((n, per)) < 1.0 / (1.0 + np.exp(-(t1 - bs)))).astype(int)
    xb = (rng.random((n, per)) < 1.0 / (1.0 + np.exp(-(t2 - bs)))).astype(int)
    res = mokken_analysis(np.hstack([xa, xb]), lower_bound=0.3)
    a, b = res.scale[0], res.scale[4]
    assert a > 0 and b > 0 and a != b, res.scale
    assert np.all(res.scale[:4] == a) and np.all(res.scale[4:] == b), res.scale
    # crate zij symmetry read-back on real data
    assert np.allclose(res.zij[off_idx := ~np.eye(8, dtype=bool)],
                       res.zij.T[off_idx])


def test_mokken_analysis_rejects_malformed_input():
    """Wrapper validation: incomplete, non-integer, negative, non-2D data and
    out-of-range c/alpha raise ValueError (each assert exercises the wrapper
    guard in front of the crate; kills guard-deletion mutants)."""
    import numpy as np
    import pytest
    from fast_mlsirm import mokken_analysis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mokken_coef_h"):
        pytest.skip("compiled core built without mokken_coef_h")

    ok = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=float)
    with pytest.raises(ValueError, match="complete"):
        bad = ok.copy()
        bad[0, 0] = np.nan
        mokken_analysis(bad)
    with pytest.raises(ValueError, match="integer"):
        mokken_analysis(ok + 0.5)
    with pytest.raises(ValueError, match="non-negative"):
        mokken_analysis(ok - 1.0)
    with pytest.raises(ValueError, match="2-D"):
        mokken_analysis(ok.reshape(-1))
    with pytest.raises(ValueError, match="lower_bound"):
        mokken_analysis(ok, lower_bound=1.0)
    with pytest.raises(ValueError, match="alpha"):
        mokken_analysis(ok, alpha=0.0)
    # crate-side guard surfaces as ValueError too (zero-variance item)
    cst = ok.copy()
    cst[:, 0] = 1.0
    with pytest.raises(ValueError, match="zero variance"):
        mokken_analysis(cst)

def test_ksirt_analysis_fixture_exact_values():
    """Kernel-smoothing IRT (Ramsay, 1991, as cited in Mazza et al., 2014):
    every assert reads crate outputs (theta/grid/bandwidth/occ/expected/
    expected_total from ksirt_occ via the wrapper) against independently
    derived constants. theta must equal qnorm literals (kills rank-map
    mutants), grid endpoints qnorm(1/5)/qnorm(4/5) (kills grid mutants),
    bandwidth the Silverman literal 1.06*4^(-1/5) (kills bandwidth mutants),
    and a huge bandwidth must flatten every OCC to the marginal proportion
    0.5 (kills NW-normalization mutants)."""
    import numpy as np
    import pytest
    from fast_mlsirm import KsirtResult, analyze_ksirt
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "ksirt_occ"):
        pytest.skip("compiled core built without ksirt_occ")

    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    q02, q04 = -0.8416212335729143, -0.2533471031357997  # qnorm(.2), qnorm(.4)
    r = analyze_ksirt(x, nevalpoints=5)
    assert isinstance(r, KsirtResult)
    # ties.method="first": totals [0,1,0,1] -> ranks [1,3,2,4]
    assert np.allclose(r.theta, [q02, -q04, q04, -q02], atol=1e-8)
    assert abs(r.grid[0] - q02) < 1e-8 and abs(r.grid[-1] + q02) < 1e-8
    assert np.allclose(np.diff(r.grid), np.diff(r.grid)[0], atol=1e-12)
    assert abs(r.bandwidth[0] - 1.06 * 4.0 ** (-0.2)) < 1e-12
    assert len(r.options) == 1 and np.array_equal(r.options[0], [0.0, 1.0])
    assert r.occ[0].shape == (2, 5)
    # huge bandwidth -> equal NW weights -> OCC = marginal proportions
    flat = analyze_ksirt(x, nevalpoints=5, bandwidth=np.array([1e9]))
    assert np.allclose(flat.occ[0], 0.5, atol=1e-12)
    assert np.allclose(flat.expected[0], 0.5, atol=1e-12)
    assert np.allclose(flat.expected_total, 0.5, atol=1e-12)


def test_ksirt_analysis_recovery_and_rejects_malformed_input():
    """Monotone-recovery smoke on a simulated 2PL item (crate expected_total
    must increase over the grid and crate theta must correlate with true
    ability; kills over-collapse/ordering mutants) plus wrapper guard
    checks (each raise exercises a validation branch in front of the
    crate)."""
    import numpy as np
    import pytest
    from fast_mlsirm import analyze_ksirt
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "ksirt_occ"):
        pytest.skip("compiled core built without ksirt_occ")

    rng = np.random.default_rng(1991)
    n = 800
    t = rng.standard_normal(n)
    bs = np.array([-1.0, -0.3, 0.4, 1.1])
    x = (rng.random((n, 4)) < 1.0 / (1.0 + np.exp(-1.5 * (t[:, None] - bs)))).astype(float)
    r = analyze_ksirt(x, kernel="gaussian", nevalpoints=51)
    d = np.diff(r.expected_total)
    assert np.all(d > -1e-9)  # non-decreasing everywhere
    assert np.all(d[10:-10] > 0.0)  # strictly increasing in the interior
    assert r.expected_total[-1] - r.expected_total[0] > 2.0
    assert np.corrcoef(r.theta, t)[0, 1] > 0.7

    ok = x[:8]
    with pytest.raises(ValueError, match="2-D"):
        analyze_ksirt(ok.reshape(-1), nevalpoints=51)
    with pytest.raises(ValueError, match="complete"):
        bad = ok.copy()
        bad[0, 0] = np.nan
        analyze_ksirt(bad, nevalpoints=51)
    with pytest.raises(ValueError, match="kernel"):
        analyze_ksirt(ok, kernel="triangular", nevalpoints=51)
    with pytest.raises(ValueError, match="nevalpoints"):
        analyze_ksirt(ok, nevalpoints=1)
    with pytest.raises(ValueError, match="nevalpoints"):
        analyze_ksirt(ok, nevalpoints=10**11)  # allocation bound
    with pytest.raises(ValueError, match="one value per item"):
        analyze_ksirt(ok, nevalpoints=51, bandwidth=np.array([0.5]))
    with pytest.raises(ValueError, match="positive"):
        analyze_ksirt(ok, nevalpoints=51, bandwidth=np.array([0.5, -0.1, 0.5, 0.5]))


def test_subscore_analysis_matches_independent_reference():
    """Haberman (2008, as cited in Sinharay, 2010) subscore added-value
    analysis: every assert reads the SubscoreResult returned by the crate via
    the PyO3 binding, pinned against literals from an independent NumPy
    transcription of the CRAN subscore R semantics (ddof=1). The fixture is
    asymmetric across subscales (added_value_sx = [False, True]), so
    decision-rule and PRMSE mutants (rowsum including the total column,
    missing true-variance diagonal, dropped +0.01 margin) all fail here."""
    import numpy as np
    import pytest
    from fast_mlsirm import SubscoreResult, subscore_analysis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "subscore_analysis"):
        pytest.skip("compiled core built without subscore_analysis")

    x = np.array(
        [
            [0, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0],
            [1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0],
            [0, 1, 1, 1], [1, 1, 0, 0],
        ],
        dtype=float,
    )
    res = subscore_analysis(x, [0, 0, 1, 1])
    assert isinstance(res, SubscoreResult)
    np.testing.assert_allclose(
        res.alpha, [0.5797101449275359, 0.33333333333333326], atol=1e-10
    )
    assert abs(res.alpha_total - 0.44742729306487705) < 1e-12
    np.testing.assert_allclose(
        res.prmse_x, [0.41946308724832204, 0.3020134228187919], atol=1e-10
    )
    np.testing.assert_allclose(
        res.prmse_sx, [0.5872524752475244, 0.3663366336633662], atol=1e-10
    )
    np.testing.assert_allclose(
        res.tau, [0.1385414635898375, 0.27024443691002326], atol=1e-10
    )
    np.testing.assert_allclose(
        res.beta, [0.49752475247524747, 0.21782178217821788], atol=1e-10
    )
    np.testing.assert_allclose(
        res.gamma, [0.07178217821782161, 0.09900990099009896], atol=1e-10
    )
    assert res.added_value_s.tolist() == [True, True]
    assert res.added_value_sx.tolist() == [False, True]
    assert res.corr.shape == (3, 3)
    assert abs(res.corr[0, 2] - 0.7791290756515445) < 1e-10
    assert np.isnan(res.disattenuated_corr[0, 0])
    assert abs(res.disattenuated_corr[0, 1] - 0.35355339059327384) < 1e-10
    # person-0 estimates from all three estimators
    np.testing.assert_allclose(
        res.subscore_s[0], [0.4623188405797105, 1.0], atol=1e-10
    )
    np.testing.assert_allclose(
        res.subscore_x[0], [0.7308724832214768, 0.778523489932886], atol=1e-10
    )
    np.testing.assert_allclose(
        res.subscore_sx[0], [0.47376237623762407, 0.8910891089108911], atol=1e-10
    )
    assert res.observed.shape == (10, 2) and res.total.shape == (10,)


def test_subscore_analysis_rejects_degenerate_inputs():
    """Guard behavior (spec-review mandated): duplicated subscales make a
    subscore collinear with the total (tau divides by 1 - r^2 = 0) and must
    raise, as must negative within-subscale alpha, bad partitions, and
    incomplete data. Every assert reads the ValueError raised by the crate
    through the binding; removing any Rust-side guard turns these into NaN
    results or panics and fails the test."""
    import numpy as np
    import pytest
    from fast_mlsirm import subscore_analysis

    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "subscore_analysis"):
        pytest.skip("compiled core built without subscore_analysis")

    base = np.array(
        [
            [0, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0],
            [1, 1, 1, 1], [1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0],
            [0, 1, 1, 1], [1, 1, 0, 0],
        ],
        dtype=float,
    )
    dup = base[:, [0, 1, 0, 1]]
    with pytest.raises(ValueError, match="collinear"):
        subscore_analysis(dup, [0, 0, 1, 1])
    rev = base.copy()
    rev[:, 1] = 1.0 - rev[:, 0]  # reversed item -> negative alpha
    with pytest.raises(ValueError, match="outside"):
        subscore_analysis(rev, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        subscore_analysis(base, [0, 0, 1])  # groups length mismatch
    with pytest.raises(ValueError):
        subscore_analysis(base, [0, 0, 0, 0])  # single subscale
    with pytest.raises(ValueError):
        subscore_analysis(base, [0, 1, 1, 1])  # singleton subscale
    nanx = base.copy()
    nanx[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        subscore_analysis(nanx, [0, 0, 1, 1])


def test_detect_analysis_matches_independent_reference():
    """Confirmatory DETECT (Zhang & Stout, 1999, via sirt's sum-score path).

    All asserts read crate outputs (DetectResult fields returned through the
    PyO3 binding). Literals come from an independent NumPy transcription of
    the R semantics that never imports this package. Killed by: delta sign
    swap, dropped bias correction, n-1 covariance divisor, index-scaling
    (x100) errors, label-as-index bugs."""
    from fast_mlsirm import detect_analysis

    x = np.array(
        [
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    res = detect_analysis(x, [0, 0, 1, 1, 0])
    assert res.detect == pytest.approx(-0.2830687830687831, abs=1e-12)
    assert res.assi == pytest.approx(-0.2, abs=1e-12)
    assert res.ratio == pytest.approx(-0.05439755973563803, abs=1e-12)
    assert res.madcov100 == pytest.approx(5.203703703703703, abs=1e-12)
    assert res.mcov100 == pytest.approx(-3.5105820105820107, abs=1e-12)
    assert res.n_pairs == 10
    assert res.ccov[0] == pytest.approx(-0.10317460317460317, abs=1e-12)
    assert res.ccov[3] == 0.0  # constant item 4: exact zero ccov
    # hostile labels behave identically (equality-only comparison)
    big = detect_analysis(
        x, np.array([-(2**63), -(2**63), 2**63 - 1, 2**63 - 1, -(2**63)])
    )
    assert big.detect == pytest.approx(res.detect, abs=1e-12)


def test_detect_analysis_rejects_degenerate_inputs():
    """Trust-boundary rejections; asserts read ValueError raised by the
    wrapper/crate before or at the Rust boundary."""
    from fast_mlsirm import detect_analysis

    base = np.array(
        [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1], [0, 0, 1, 1]], dtype=float
    )
    with pytest.raises(ValueError):
        detect_analysis(base, [0, 0, 1])  # cluster length mismatch
    with pytest.raises(ValueError, match="complete"):
        nanx = base.copy()
        nanx[0, 0] = np.nan
        detect_analysis(nanx, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        half = base.copy()
        half[1, 2] = 0.5  # non-binary
        detect_analysis(half, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        detect_analysis(base[:1], [0, 0, 1, 1])  # 1 person
    with pytest.raises(ValueError):
        detect_analysis(np.ones((4, 2)), [0, 1])  # all-zero ccov: RATIO 0/0
    # float labels beyond i64 must be rejected, not silently wrapped by
    # astype(int64) (which would collapse distinct labels into one cluster)
    with pytest.raises(ValueError, match="64-bit"):
        detect_analysis(base, [2.0**63, 2.0**63, 2.0**63 + 2048, 0.0])


def test_classification_accuracy_matches_independent_reference():
    """Rudner and Lee classification accuracy/consistency (Rudner, 2001,
    2005; Lee, 2010, as cited in Lathrop's cacIRT sources): every assert
    reads crate outputs returned through the wrapper against literals from
    an independent NumPy transcription (exact math.erf / own recursion,
    never this crate). Rudner tolerance 1e-6 covers the crate erfc
    approximation (|err| < 1.2e-7); Lee values are exact-recursion, pinned
    at 1e-12. Weights are deliberately unnormalized (kills a forgot-to-
    normalize mutation); Lee cut 2.4 is non-integer (kills floor-vs-ceil);
    theta[2] == cut -0.4 anchors left-closed categorization."""
    from fast_mlsirm import lee_classification, rudner_classification
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "rudner_classification"):
        pytest.skip("compiled core built without classification")

    r = rudner_classification(
        theta=[-1.7, -0.63, -0.4, 0.11, 0.52, 1.28, 1.95],
        sem=[0.85, 0.42, 0.31, 0.27, 0.33, 0.48, 0.66],
        cutscores=(-0.4, 0.85),
        weights=[0.4, 1.1, 1.7, 2.3, 1.9, 0.8, 0.3],
    )
    tol = 1e-6
    assert abs(r.per_cut_accuracy[0] - 0.8506551287597999) < tol
    assert abs(r.per_cut_consistency[1] - 0.9068646398531202) < tol
    assert abs(r.simultaneous_accuracy - 0.7952699812339523) < tol
    assert abs(r.simultaneous_consistency - 0.7315355182946305) < tol
    assert r.conditional_accuracy.shape == (2, 7)
    assert abs(r.conditional_accuracy[1, 4] - 0.8413447460685428) < tol
    # theta exactly on the cut -> upper category (left-closed)
    assert abs(r.conditional_simultaneous_accuracy[2] - 0.4999723782624206) < tol

    p = np.array(
        [
            [0.08, 0.15, 0.22, 0.31, 0.12, 0.19],
            [0.23, 0.34, 0.41, 0.52, 0.28, 0.37],
            [0.47, 0.55, 0.61, 0.68, 0.51, 0.58],
            [0.66, 0.72, 0.79, 0.83, 0.69, 0.76],
            [0.81, 0.87, 0.90, 0.93, 0.84, 0.88],
        ]
    )
    le = lee_classification(
        p, cutscores=(2.4, 4.0), weights=[0.6, 1.4, 2.0, 1.4, 0.6]
    )
    tol = 1e-12
    assert abs(le.per_cut_accuracy[0] - 0.8221250493919999) < tol
    assert abs(le.per_cut_consistency[1] - 0.7071425914148844) < tol
    assert abs(le.simultaneous_accuracy - 0.6284128106933332) < tol
    assert abs(le.simultaneous_consistency - 0.5754886794891432) < tol
    assert abs(le.conditional_consistency[1, 2] - 0.5012741633381826) < tol
    assert abs(le.conditional_simultaneous_accuracy[2] - 0.2992865452) < tol

    # true score exactly on an integer cut -> upper category (left-closed);
    # dyadic probs make the sum exact in binary64
    edge = lee_classification([[0.5, 0.5, 0.5, 0.5, 0.25, 0.75]], cutscores=(3.0,))
    assert abs(edge.per_cut_accuracy[0] - 0.6640625) < tol
    assert abs(edge.per_cut_consistency[0] - 0.5538330078125) < tol


def test_classification_rejects_degenerate_inputs():
    """Trust-boundary rejections; asserts read the ValueError/RuntimeError
    raised by the wrapper/crate at the Rust boundary."""
    from fast_mlsirm import lee_classification, rudner_classification
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "rudner_classification"):
        pytest.skip("compiled core built without classification")

    with pytest.raises(ValueError):
        rudner_classification([0.0, 1.0], [0.5], (0.0,))  # length mismatch
    with pytest.raises(ValueError):
        rudner_classification([0.0, 1.0], [0.5, 0.0], (0.0,))  # sem <= 0
    with pytest.raises(ValueError):
        rudner_classification([0.0, 1.0], [0.5, 0.5], (0.3, 0.3))  # cuts not increasing
    with pytest.raises(ValueError):
        rudner_classification([0.0, 1.0], [0.5, 0.5], (0.0,), weights=[0.0, 0.0])

    good = [[0.2, 0.4], [0.6, 0.8]]
    with pytest.raises(ValueError):
        lee_classification([[1.0, 0.4], [0.6, 0.8]], (1.5,))  # P == 1
    with pytest.raises(ValueError):
        lee_classification(good, (2.5,))  # raw cut > n_items
    with pytest.raises(ValueError):
        lee_classification(good, (0.2, 0.9))  # ceil-collision at 1
    with pytest.raises(ValueError):
        lee_classification(np.array([0.2, 0.4]), (1.5,))  # 1-D probs



def test_parallel_analysis_matches_numpy_fixture():
    """Horn parallel analysis (paran PCA path): fixture literals computed by
    an independent NumPy replication that mirrors the crate LCG stream with
    np.uint64 arithmetic and uses np.corrcoef + np.linalg.eigvalsh. Every
    assert reads crate outputs returned through the wrapper. Duplicate
    columns give analytic observed eigenvalues {2, 0}; the random benchmark,
    bias, adjustment, and retention are pinned deterministically
    (n=12, p=2, iterations=20, centile=0, seed=7)."""
    from fast_mlsirm import parallel_analysis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "parallel_analysis"):
        pytest.skip("compiled core built without parallel_analysis")

    x = np.arange(1.0, 13.0)
    res = parallel_analysis(
        np.column_stack([x, x]), n_iterations=20, centile=0, seed=7
    )
    tol = 1e-9
    assert abs(res.eigenvalues[0] - 2.0) < tol
    assert abs(res.eigenvalues[1]) < tol
    assert abs(res.random_eigenvalues[0] - 1.265490626983912) < tol
    assert abs(res.random_eigenvalues[1] - 0.734509373016088) < tol
    assert abs(res.bias[0] - 0.265490626983912) < tol
    assert abs(res.adjusted_eigenvalues[0] - 1.734509373016088) < tol
    assert abs(res.adjusted_eigenvalues[1] - 0.265490626983912) < tol
    assert res.retained == 1


def test_parallel_analysis_rejects_degenerate_inputs():
    """Trust-boundary rejections raised at the Rust boundary."""
    from fast_mlsirm import parallel_analysis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "parallel_analysis"):
        pytest.skip("compiled core built without parallel_analysis")

    good = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 4.5]])
    with pytest.raises(ValueError):
        parallel_analysis(good[:2], n_iterations=5)  # n_persons < 3
    with pytest.raises(ValueError):
        parallel_analysis(good[:, :1], n_iterations=5)  # n_items < 2
    with pytest.raises(ValueError):
        parallel_analysis(good, n_iterations=0)
    with pytest.raises(ValueError):
        parallel_analysis(good, n_iterations=5, centile=100)
    bad = good.copy()
    bad[1, 1] = np.nan
    with pytest.raises(ValueError):
        parallel_analysis(bad, n_iterations=5)
    const = good.copy()
    const[:, 0] = 1.0
    with pytest.raises(ValueError):
        parallel_analysis(const, n_iterations=5)  # zero variance
    with pytest.raises(ValueError):
        parallel_analysis(np.array([1.0, 2.0, 3.0]), n_iterations=5)  # 1-D


def test_parallel_analysis_rejects_negative_scalars_and_huge_data():
    """Impl-review regressions: negative scalars raise ValueError (not
    PyO3 OverflowError) via wrapper validation; finite-but-huge data that
    overflows the correlation computation is rejected by the crate."""
    from fast_mlsirm import parallel_analysis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "parallel_analysis"):
        pytest.skip("compiled core built without parallel_analysis")

    good = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 4.5]])
    with pytest.raises(ValueError):
        parallel_analysis(good, n_iterations=-1)
    with pytest.raises(ValueError):
        parallel_analysis(good, n_iterations=5, centile=-3)
    with pytest.raises(ValueError):
        parallel_analysis(good, n_iterations=5, seed=-1)
    big = 1e308
    huge = np.array([[big, big], [big, -big], [-big, big], [-big, -big]])
    with pytest.raises(ValueError):
        parallel_analysis(huge, n_iterations=5)


def _guttman_gen_data(n, p, seed, loadings, means, scales):
    """NumPy mirror of the crate LCG used to build the Guttman fixture data
    (test-input construction only; all asserted values come from the crate)."""
    state = np.uint64(max(seed, 1))
    mul = np.uint64(6364136223846793005)
    add = np.uint64(1442695040888963407)

    def uniform():
        nonlocal state
        with np.errstate(over="ignore"):
            state = state * mul + add
        return float(state >> np.uint64(11)) / float(1 << 53)

    def normal():
        u1 = max(uniform(), 1e-12)
        u2 = uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    out = np.empty((n, p))
    for i in range(n):
        f = normal()
        for j in range(p):
            e = normal()
            out[i, j] = means[j] + scales[j] * (loadings[j] * f + e)
    return out


def test_guttman_lambdas_matches_numpy_fixture():
    """Guttman lambdas (psych 2.6.5 contract): fixture A literals computed by
    an independent NumPy replication (np.corrcoef, np.linalg.inv,
    itertools.combinations). Every assert reads GuttmanResult fields returned
    by the crate through the wrapper (n=30, p=6, exhaustive C(6,3)=20)."""
    from fast_mlsirm import guttman_lambdas
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "guttman_lambdas"):
        pytest.skip("compiled core built without guttman_lambdas")

    data = _guttman_gen_data(
        30,
        6,
        42,
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.2],
        [10.0, -3.0, 5.0, 0.5, 100.0, 7.0],
        [1.0, 2.0, 0.5, 3.0, 10.0, 1.5],
    )
    g = guttman_lambdas(data)
    tol = 1e-9
    assert abs(g.lambda1 - 0.41068157604336863) < tol
    assert abs(g.lambda2 - 0.5561900258418755) < tol
    assert abs(g.lambda3 - 0.49281789125204234) < tol
    assert abs(g.lambda4 - 0.6842463861953535) < tol
    assert abs(g.lambda5 - 0.5602004712206824) < tol
    assert abs(g.lambda6 - 0.59150154331973) < tol
    assert abs(g.beta - 0.1784553694910821) < tol
    assert abs(g.mean_split - 0.4928178912520423) < tol
    assert g.n_splits == 20
    assert g.exhaustive


def test_guttman_lambdas_rejects_degenerate_inputs():
    """Trust-boundary rejections: wrapper scalars raise ValueError (not
    PyO3 OverflowError); shape/finite/variance/singularity checks raise at
    the Rust boundary."""
    from fast_mlsirm import guttman_lambdas
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "guttman_lambdas"):
        pytest.skip("compiled core built without guttman_lambdas")

    rng = np.random.default_rng(3)
    good = rng.normal(size=(10, 4))
    with pytest.raises(ValueError):
        guttman_lambdas(good[:2])  # n_persons < 3
    with pytest.raises(ValueError):
        guttman_lambdas(good[:, :2])  # n_items < 3
    with pytest.raises(ValueError):
        guttman_lambdas(good, n_sample_splits=0)
    with pytest.raises(ValueError):
        guttman_lambdas(good, seed=-1)
    with pytest.raises(ValueError):
        guttman_lambdas(np.array([1.0, 2.0, 3.0]))  # 1-D
    bad = good.copy()
    bad[1, 1] = np.nan
    with pytest.raises(ValueError):
        guttman_lambdas(bad)
    const = good.copy()
    const[:, 0] = 1.0
    with pytest.raises(ValueError):
        guttman_lambdas(const)  # zero variance
    dup = good.copy()
    dup[:, 1] = dup[:, 0]  # singular correlation matrix -> lambda6 inverse
    with pytest.raises(ValueError):
        guttman_lambdas(dup)


def test_tenberge_mu_matches_numpy_fixture():
    """ten Berge & Zegers mu series (psych 2.6.5 tenberge.R contract):
    fixture A literals from an independent NumPy replication (np.corrcoef +
    explicit off-diagonal power sums). Every assert reads TenBergeResult
    fields returned by the crate through the wrapper."""
    from fast_mlsirm import tenberge_mu
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "tenberge_mu"):
        pytest.skip("compiled core built without tenberge_mu")

    data = _guttman_gen_data(
        30,
        6,
        42,
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.2],
        [10.0, -3.0, 5.0, 0.5, 100.0, 7.0],
        [1.0, 2.0, 0.5, 3.0, 10.0, 1.5],
    )
    t = tenberge_mu(data)
    tol = 1e-9
    assert abs(t.mu0 - 0.49281789125204223) < tol
    assert abs(t.mu1 - 0.5561900258418755) < tol
    assert abs(t.mu2 - 0.566334883191414) < tol
    assert abs(t.mu3 - 0.5694865282091219) < tol
    assert t.mu0 <= t.mu1 <= t.mu2 <= t.mu3


def test_tenberge_mu_rejects_degenerate_inputs():
    """Trust-boundary rejections raised at the Rust boundary."""
    from fast_mlsirm import tenberge_mu
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "tenberge_mu"):
        pytest.skip("compiled core built without tenberge_mu")

    rng = np.random.default_rng(9)
    good = rng.normal(size=(10, 4))
    with pytest.raises(ValueError):
        tenberge_mu(good[:2])  # n_persons < 3
    with pytest.raises(ValueError):
        tenberge_mu(good[:, :2])  # n_items < 3
    with pytest.raises(ValueError):
        tenberge_mu(np.array([1.0, 2.0, 3.0]))  # 1-D
    bad = good.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        tenberge_mu(bad)
    const = good.copy()
    const[:, 3] = 0.25
    with pytest.raises(ValueError):
        tenberge_mu(const)  # zero variance


def test_cronbach_alpha_and_feldt_ci_fixture():
    """Pinned against an independent NumPy+scipy replication (scipy.stats.f.ppf).

    Reads crate outputs via fast_mlsirm.cronbach_alpha / feldt_alpha_ci.
    """
    from fast_mlsirm import cronbach_alpha, feldt_alpha_ci
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "feldt_alpha_ci"):
        pytest.skip("compiled core built without feldt_alpha_ci")

    data = _guttman_gen_data(
        30, 6, 42,
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.2],
        [10.0, -3.0, 5.0, 0.5, 100.0, 7.0],
        [1.0, 2.0, 0.5, 3.0, 10.0, 1.5],
    )
    tol = 1e-9
    a = cronbach_alpha(data)
    assert abs(a - 0.26300193786625514) < tol
    ci = feldt_alpha_ci(a, 30, 6, level=0.95)
    assert abs(ci.lower - -0.23710034474579156) < tol
    assert abs(ci.upper - 0.6065060295943752) < tol
    assert abs(ci.r_bar - 0.05613713592263862) < tol
    assert ci.df1 == 29.0 and ci.df2 == 145.0
    assert ci.lower < a < ci.upper


def test_feldt_ci_rejects_degenerate_inputs():
    """Trust-boundary rejections raised at the Rust boundary."""
    from fast_mlsirm import cronbach_alpha, feldt_alpha_ci
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "feldt_alpha_ci"):
        pytest.skip("compiled core built without feldt_alpha_ci")

    rng = np.random.default_rng(11)
    good = rng.normal(size=(10, 4))
    with pytest.raises(ValueError):
        cronbach_alpha(good[:2])  # n_persons < 3
    with pytest.raises(ValueError):
        cronbach_alpha(np.array([1.0, 2.0]))  # 1-D
    flat = good.copy()
    flat[:, 0] = 3.0
    with pytest.raises(ValueError):
        cronbach_alpha(flat)  # zero-variance item
    with pytest.raises(ValueError):
        feldt_alpha_ci(1.0, 30, 6)  # alpha >= 1
    with pytest.raises(ValueError):
        feldt_alpha_ci(0.8, 30, 6, level=1.0)
    with pytest.raises(ValueError):
        feldt_alpha_ci(0.8, 2, 6)


def test_livingston_lewis_matches_independent_reference():
    """Livingston-Lewis single-administration classification (Livingston &
    Lewis, 1995, as implemented in CRAN betafunctions 1.9.0 LL.CA, read in
    full): every assert reads crate outputs returned through the wrapper
    against literals from an independent scipy.integrate.quad replication
    of the pipeline (never this crate). The fixture takes the genuine
    four-parameter path (kills a dropped-failsafe-branch confusion) and has
    round(48 * 0.56) = 27 != floor = 26 (anchors round-ties-even in k).
    Integral tolerance 1e-7 covers the crate Gauss-Legendre quadrature."""
    from fast_mlsirm import livingston_lewis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "livingston_lewis"):
        pytest.skip("compiled core built without livingston_lewis")

    scores = [11, 14, 16, 17, 19, 20, 21, 22, 23, 23, 24, 25, 25, 26, 27,
              27, 28, 29, 30, 31, 31, 32, 33, 34, 35, 36, 38, 39, 41, 44]
    r = livingston_lewis(scores, 0.82, 0.0, 50.0, 28.0)
    assert abs(r.effective_test_length - 47.50171019930535) < 1e-9
    assert r.etl_rounded == 48
    assert not r.used_two_parameter
    assert abs(r.lower - 0.19555999873721175) < 1e-9
    assert abs(r.upper - 0.9652932601665462) < 1e-9
    assert abs(r.alpha - 2.7763142806015297) < 1e-9
    assert abs(r.beta - 3.2986812798601846) < 1e-9
    tol = 1e-7
    assert abs(r.p_tp - 0.40501323923680005) < tol
    assert abs(r.p_fp - 0.07917200341607164) < tol
    assert abs(r.p_tf - 0.46137445417258505) < tol
    assert abs(r.p_ff - 0.05444030317454212) < tol
    assert abs(r.accuracy - 0.866387693409385) < tol
    assert abs(r.sensitivity - 0.8815107553881857) < tol
    assert abs(r.specificity - 0.8535333969826517) < tol
    assert abs(r.p_ii - 0.4232336892048015) < tol
    assert abs(r.p_ij - 0.0925810681423261) < tol
    assert r.p_ji == r.p_ij  # symmetric by construction (single threshold)
    assert abs(r.p_jj - 0.39160417451054624) < tol
    assert abs(r.consistency - 0.8148378637153477) < tol
    assert abs(r.chance_consistency - 0.5005002130998969) < tol
    assert abs(r.kappa - 0.6293048743148242) < tol


def test_livingston_lewis_rejects_degenerate_inputs():
    """Trust-boundary rejections; asserts read the ValueError raised by the
    wrapper/crate at the Rust boundary."""
    from fast_mlsirm import livingston_lewis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "livingston_lewis"):
        pytest.skip("compiled core built without livingston_lewis")

    good = [11.0, 14.0, 20.0, 25.0, 31.0, 36.0, 41.0, 44.0]
    with pytest.raises(ValueError):
        livingston_lewis(good, 1.0, 0.0, 50.0, 28.0)  # reliability >= 1
    with pytest.raises(ValueError):
        livingston_lewis(good, 0.82, 0.0, 50.0, 50.0)  # cut not inside
    with pytest.raises(ValueError):
        livingston_lewis(good, 0.82, 12.0, 50.0, 28.0)  # score below min
    with pytest.raises(ValueError):
        livingston_lewis([20.0, 20.0, 20.0], 0.82, 0.0, 50.0, 10.0)  # var 0

def test_gtheory_pi_matches_paper_tables():
    """Huebner & Lucht (2019) appendix pi_dat; asserts read the crate's
    var/var_raw/d_study outputs (Tables 3-4 reproduced by the Rust tests
    to full precision; pinned literals from the same verified pipeline)."""
    from fast_mlsirm import gtheory_pi
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "gtheory_pi"):
        pytest.skip("compiled core built without gtheory_pi")

    x = np.array(
        [
            [9.0, 9.0, 7.0, 4.0],
            [9.0, 8.0, 4.0, 6.0],
            [8.0, 8.0, 6.0, 2.0],
            [9.0, 8.0, 6.0, 3.0],
            [10.0, 9.0, 8.0, 7.0],
            [6.0, 4.0, 5.0, 1.0],
        ]
    )
    r = gtheory_pi(x, n_i_prime=[4, 8])
    tol = 1e-9
    assert abs(r.var[0] - 1.905555555556) < tol
    assert abs(r.var[1] - 4.027777777778) < tol
    assert abs(r.var[2] - 1.277777777778) < tol
    assert r.var == r.var_raw  # all raw components positive here
    d4 = r.d_study[0]
    assert d4.n_i_prime == 4
    assert abs(d4.rel_error_var - 1.277777777778 / 4.0) < tol
    assert abs(d4.generalizability - 0.856429463171) < tol
    assert abs(d4.dependability - 0.589600343790) < tol
    d8 = r.d_study[1]
    # sigma^2(p) invariant across n' (weak identity); delta halves.
    assert abs(d8.rel_error_var - d4.rel_error_var / 2.0) < tol
    assert d8.generalizability > d4.generalizability


def test_gtheory_pio_matches_paper_tables_and_rejects_bad_input():
    """Huebner & Lucht (2019) appendix pio_cross_dat (6 x 4 x 2); asserts
    read the crate's 7-component var vector (order p,i,o,pi,po,io,pio) and
    D-study coefficients pinned from the verified pipeline (Tables 5-6)."""
    from fast_mlsirm import gtheory_pi, gtheory_pio
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "gtheory_pio"):
        pytest.skip("compiled core built without gtheory_pio")

    # pio_cross_dat in crate layout x[p*n_i*n_o + i*n_o + o], i.e. a
    # (6, 4, 2) persons x items x occasions array.
    scores = np.array(
        [
            9.0, 9.0, 9.0, 8.0, 7.0, 5.0, 4.0, 5.0,
            9.0, 6.0, 8.0, 5.0, 4.0, 3.0, 6.0, 3.0,
            8.0, 8.0, 8.0, 7.0, 6.0, 3.0, 2.0, 2.0,
            9.0, 9.0, 8.0, 6.0, 6.0, 6.0, 3.0, 2.0,
            10.0, 8.0, 9.0, 8.0, 8.0, 9.0, 7.0, 7.0,
            6.0, 3.0, 4.0, 2.0, 5.0, 3.0, 1.0, 2.0,
        ]
    ).reshape(6, 4, 2)
    r = gtheory_pio(scores, n_prime=[(4, 2)])
    tol = 1e-9
    want = [
        2.483333333333, 3.066666666667, 0.577777777778, 0.766666666667,
        0.088888888889, 0.022222222222, 0.831944444444,
    ]
    for got, w in zip(r.var, want):
        assert abs(got - w) < tol
    d = r.d_study[0]
    assert d.n_i_prime == 4 and d.n_o_prime == 2
    # delta = pi/4 + po/2 + pio/8 (kills a dropped-term mutation).
    assert abs(d.rel_error_var - (want[3] / 4.0 + want[4] / 2.0 + want[6] / 8.0)) < tol
    assert 0.0 < d.dependability < d.generalizability < 1.0

    with pytest.raises(ValueError):
        gtheory_pio(scores[:1])  # n_p < 2
    with pytest.raises(ValueError):
        gtheory_pi(np.array([[1.0, np.nan], [2.0, 3.0]]))  # non-finite


def test_minres_fa_oracle_parity_9x2():
    """Rust minres FA matches the pinned scipy L-BFGS-B oracle (session
    oracle_minres.py) on the 9-var / 2-factor exact-structure matrix.
    Asserts read crate outputs via the binding (loadings/uniquenesses/
    objective/converged)."""
    import numpy as np

    from fast_mlsirm import minres_fa

    lam = np.zeros((9, 2))
    lam[:5, 0] = [0.8, 0.7, 0.6, 0.5, 0.4]
    lam[4:, 1] = [0.3, 0.7, 0.6, 0.5, 0.45]
    s = lam @ lam.T
    np.fill_diagonal(s, 1.0)
    r = minres_fa(s, 2)
    assert r.converged
    assert r.objective < 1e-9
    assert r.loadings.shape == (9, 2)
    want_row0 = (0.780565885133, -0.175262374782)
    assert abs(r.loadings[0, 0] - want_row0[0]) < 5e-5
    assert abs(r.loadings[0, 1] - want_row0[1]) < 5e-5
    want_psi = [0.36, 0.51, 0.64, 0.75, 0.75, 0.51, 0.64, 0.75, 0.7975]
    assert np.allclose(r.uniquenesses, want_psi, atol=5e-5)
    assert np.allclose(r.communalities, (r.loadings**2).sum(axis=1), atol=1e-12)


def test_omega_total_1f_oracle_parity():
    """omega_total_1f on the pinned 6-var sampled-data correlation matrix
    reproduces the oracle omega (0.825058734426). Asserts read the crate
    omega and embedded fit via the binding."""
    import numpy as np

    from fast_mlsirm import omega_total_1f

    r6 = np.array(
        [
            [1.0, 0.498568868921, 0.625966200012, 0.370348287807, 0.540999349418, 0.424034177377],
            [0.498568868921, 1.0, 0.511016876110, 0.334681533406, 0.380581491599, 0.347762451610],
            [0.625966200012, 0.511016876110, 1.0, 0.345954222399, 0.593952865689, 0.454414314062],
            [0.370348287807, 0.334681533406, 0.345954222399, 1.0, 0.327886204696, 0.311070632680],
            [0.540999349418, 0.380581491599, 0.593952865689, 0.327886204696, 1.0, 0.415636442046],
            [0.424034177377, 0.347762451610, 0.454414314062, 0.311070632680, 0.415636442046, 1.0],
        ]
    )
    o = omega_total_1f(r6)
    assert o.fa.converged
    assert abs(o.omega_total - 0.825058734426) < 1e-5
    assert abs(o.fa.loadings[0, 0] - 0.774100663868) < 5e-5
    assert abs(o.fa.uniquenesses[2] - 0.345079760758) < 5e-5


def test_separation_reliability_matches_numpy_oracle():
    # Pinned numpy oracle (ddof=1 var; mean se^2); asserts read wrapper
    # outputs which read the Rust crate values.
    from fast_mlsirm import separation_reliability

    x = np.array([-1.8, -0.6, 0.15, 0.9, 2.1, -0.35, 1.4])
    se = np.array([0.45, 0.38, 0.35, 0.37, 0.52, 0.36, 0.41])
    r = separation_reliability(x, se)
    assert abs(r.ssd - 1.743690476190) < 1e-10
    assert abs(r.mse - 0.167771428571) < 1e-10
    assert abs(r.sep_rel - 0.903783709975) < 1e-10
    assert abs(r.sep_index - 3.064841016127) < 1e-10


def test_separation_reliability_negative_unclamped():
    from fast_mlsirm import separation_reliability

    r = separation_reliability(
        np.array([0.0, 0.1, -0.1, 0.05]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    assert r.sep_rel < 0.0
    assert np.isnan(r.sep_index)
def test_glb_fa_oracle_s9():
    """S9 oracle fixture from oracle_glbfa.py (asserts read wrapper->crate
    outputs; mutation kills executed at the Rust level)."""
    import numpy as np
    from fast_mlsirm import glb_fa

    L = np.zeros((9, 2))
    L[:5, 0] = [0.8, 0.7, 0.6, 0.5, 0.4]
    L[4:, 1] = [0.3, 0.7, 0.6, 0.5, 0.45]
    R = L @ L.T
    np.fill_diagonal(R, 1.0)
    res = glb_fa(R)
    assert res.nf == 2
    assert abs(res.glb - 0.730905233399) < 1e-5
    assert res.communalities.shape == (9,)


def test_glb_fa_from_data_and_errors():
    import numpy as np
    import pytest
    from fast_mlsirm import glb_fa, glb_fa_from_data

    rng = np.random.default_rng(7)
    f = rng.standard_normal(500)
    lam = np.array([0.7, 0.6, 0.8, 0.5, 0.65, 0.75])
    x = np.outer(f, lam) + rng.standard_normal((500, 6)) * np.sqrt(1 - lam**2)
    res = glb_fa_from_data(x)
    assert 0.5 < res.glb <= 1.0
    assert 1 <= res.nf < 6
    with pytest.raises(ValueError):
        glb_fa(np.ones((2, 3)))
    with pytest.raises(ValueError):
        glb_fa_from_data(np.full((5, 4), np.nan))


class TestSelectionUtility:
    """Selection utility (Taylor-Russell / Naylor-Shine / BCG) wrappers.

    Oracle values pinned from scipy (tests/oracles/oracle_utility.py); asserts read
    the crate outputs returned through the wrapper. Tolerance 1e-7 reflects
    the crate's Acklam inverse-normal precision, not slack.
    """

    def test_selection_utility_oracle(self):
        import fast_mlsirm as fm

        r = fm.selection_utility(n=1, sdy=10000, rxy=0.5, sr=0.3)
        assert abs(r.pux - 0.579487690333456) < 1e-7
        assert abs(r.utility_gain - 5794.8769033346) < 1e-3
        r2 = fm.selection_utility(n=50, sdy=8000, rxy=0.4, sr=0.2,
                                  cost_total=25000, period=3)
        assert abs(r2.utility_gain - 646908.6089787399) < 1e-2

    def test_taylor_russell_oracle_and_errors(self):
        import pytest

        import fast_mlsirm as fm

        r = fm.taylor_russell(rxy=0.5, sr=0.3, br=0.2)
        assert abs(r.success_ratio - 0.384157434057053) < 1e-7
        assert abs(r.q_joint - 0.115247230217116) < 1e-7
        # rho=0 analytic anchor: success == br
        r0 = fm.taylor_russell(rxy=0.0, sr=0.3, br=0.7)
        assert abs(r0.success_ratio - 0.7) < 1e-6
        with pytest.raises(ValueError):
            fm.taylor_russell(rxy=1.0, sr=0.3, br=0.5)
        with pytest.raises(ValueError):
            fm.selection_utility(n=0, sdy=1, rxy=0.5, sr=0.3)


class TestSympsonHetter:
    """Sympson-Hetter exposure control wrapper (mlsirm_core::exposure).

    Every assert reads crate outputs returned through the wrapper; the sum
    identity sum(exposure) == test_length is an exact counting identity of
    the returned rates, not a local recomputation of the algorithm.
    """

    @staticmethod
    def _pool():
        import numpy as np

        n = 30
        x = np.arange(n, dtype=float)
        a = 0.6 + 1.4 * np.abs(np.sin(x * 0.37))
        b = -2.4 + 4.8 * x / (n - 1) + 0.3 * np.sin(x * 0.71)
        return a, b

    def test_controls_max_exposure_and_identities(self):
        import fast_mlsirm as fm

        a, b = self._pool()
        r = fm.sympson_hetter(
            a, b, r_max=0.25, test_length=5, n_simulees=1500,
            max_iter=12, tol=0.02, seed=42, q_theta=31,
        )
        assert r.converged
        assert r.max_exposure <= 0.25 + 0.02 + 1e-12
        assert abs(float(r.exposure.sum()) - 5.0) < 1e-9
        assert (r.exposure <= r.selection + 1e-12).all()
        assert ((r.k > 0.0) & (r.k <= 1.0)).all()
        assert r.n_iter == len(r.history_max_exposure)
        assert abs(r.max_exposure - float(r.history_max_exposure[-1])) < 1e-15

    def test_rmax_one_unconstrained_and_errors(self):
        import numpy as np
        import pytest

        import fast_mlsirm as fm

        a, b = self._pool()
        r = fm.sympson_hetter(
            a, b, r_max=1.0, test_length=5, n_simulees=400,
            max_iter=6, seed=7, q_theta=21,
        )
        assert r.n_iter == 1
        assert (r.k == 1.0).all()
        assert np.array_equal(r.exposure, r.selection)
        with pytest.raises(ValueError):
            fm.sympson_hetter(a, b, r_max=0.05, test_length=5)  # infeasible
        with pytest.raises(ValueError):
            fm.sympson_hetter(a, b, r_max=0.25, test_length=0)
        # Boundary coercion discipline: non-integral floats, bools, and
        # negative seeds are ValueError, not silent truncation/OverflowError.
        with pytest.raises(ValueError):
            fm.sympson_hetter(a, b, r_max=0.25, test_length=5.9)
        with pytest.raises(ValueError):
            fm.sympson_hetter(a, b, r_max=0.25, test_length=5, n_simulees=True)
        with pytest.raises(ValueError):
            fm.sympson_hetter(a, b, r_max=0.25, test_length=5, seed=-1)
        with pytest.raises(ValueError):
            fm.sympson_hetter(a, b, r_max=0.25, test_length=float("inf"))
        with pytest.raises(ValueError):
            fm.sympson_hetter(a, b, r_max=0.25, test_length=np.float64("nan"))


class TestAStratified:
    """a-stratified multistage CAT wrapper (mlsirm_core::exposure).

    Every assert reads crate outputs returned through the wrapper; the
    per-stratum sum identity sum(exposure[stratum == k]) == stage_lengths[k]
    is an exact counting identity of returned values, not a local
    recomputation of the algorithm.
    """

    @staticmethod
    def _pool():
        n = 30
        x = np.arange(n, dtype=float)
        a = 0.6 + 1.4 * np.abs(np.sin(x * 0.37))
        b = -2.4 + 4.8 * x / (n - 1) + 0.3 * np.sin(x * 0.71)
        return a, b

    def test_per_stratum_identity_and_strata(self):
        import fast_mlsirm as fm

        a, b = self._pool()
        r = fm.a_stratified(
            a, b, n_strata=4, test_length=10, n_simulees=200,
            seed=42, q_theta=31,
        )
        assert list(r.stage_lengths) == [3, 3, 2, 2]
        for k in range(4):
            got = float(r.exposure[r.stratum == k].sum())
            assert abs(got - r.stage_lengths[k]) < 1e-9
        assert abs(float(r.exposure.sum()) - 10.0) < 1e-9
        assert r.max_exposure == float(r.exposure.max())
        assert np.isfinite(r.theta_rmse) and r.theta_rmse >= 0.0
        assert np.isfinite(r.theta_bias)
        # Strata are ascending in a: mean a must not decrease across strata
        # (reads the crate stratum assignment).
        means = [float(a[r.stratum == k].mean()) for k in range(4)]
        assert all(m2 >= m1 for m1, m2 in zip(means, means[1:]))

    def test_validation_and_boundary_coercion(self):
        import fast_mlsirm as fm

        a, b = self._pool()
        with pytest.raises(ValueError):
            fm.a_stratified(a, b, n_strata=0, test_length=10)
        with pytest.raises(ValueError):
            fm.a_stratified(a, b, n_strata=11, test_length=10)
        with pytest.raises(ValueError):
            fm.a_stratified(a, b, n_strata=4, test_length=31)
        with pytest.raises(ValueError):
            fm.a_stratified(a, b, n_strata=4, test_length=10.5)
        with pytest.raises(ValueError):
            fm.a_stratified(a, b, n_strata=True, test_length=10)
        with pytest.raises(ValueError):
            fm.a_stratified(a, b, n_strata=4, test_length=10, seed=-1)
        with pytest.raises(ValueError):
            fm.a_stratified(a, b, n_strata=4, test_length=float("nan"))


class TestVelicerMap:
    """Velicer (1976) minimum average partial test (O'Connor programs READ;
    core oracle: independent NumPy transcription on Harman-8). Every assert
    reads crate outputs through the binding."""

    HARMAN8 = np.array(
        [
            [1.000, 0.846, 0.805, 0.859, 0.473, 0.398, 0.301, 0.382],
            [0.846, 1.000, 0.881, 0.826, 0.376, 0.326, 0.277, 0.415],
            [0.805, 0.881, 1.000, 0.801, 0.380, 0.319, 0.237, 0.345],
            [0.859, 0.826, 0.801, 1.000, 0.436, 0.329, 0.327, 0.365],
            [0.473, 0.376, 0.380, 0.436, 1.000, 0.762, 0.730, 0.629],
            [0.398, 0.326, 0.319, 0.329, 0.762, 1.000, 0.583, 0.577],
            [0.301, 0.277, 0.237, 0.327, 0.730, 0.583, 1.000, 0.539],
            [0.382, 0.415, 0.345, 0.365, 0.629, 0.577, 0.539, 1.000],
        ]
    )

    def test_harman8_oracle(self):
        # Reads crate f2/f4/retained via the binding; kills any wrapper
        # reshaping or dict-key regression on top of the Rust-side kills.
        res = velicer_map(self.HARMAN8)
        np.testing.assert_allclose(
            res.f2,
            [0.312474786, 0.245120888, 0.066444959, 0.127594380,
             0.204202701, 0.271829458, 0.434591269, 1.0],
            atol=1e-7,
        )
        assert res.retained_f2 == 2
        assert res.retained_f4 == 2
        assert res.f4[2] == pytest.approx(0.011930167, abs=1e-7)

    def test_identity_invalid_rows(self):
        res = velicer_map(np.eye(5))
        assert res.f2[0] == 0.0
        assert res.retained_f2 == 0
        assert np.isnan(res.f2[1:]).all()

    def test_data_path_and_default_max_m(self):
        rng = np.random.default_rng(7)
        f = rng.standard_normal((300, 2))
        lam = np.zeros((6, 2))
        lam[:3, 0] = 0.8
        lam[3:, 1] = 0.8
        data = f @ lam.T + np.sqrt(1 - 0.64) * rng.standard_normal((300, 6))
        res = velicer_map_from_data(data)
        assert len(res.f2) == 6  # max_m defaults to p - 1 -> rows 0..5
        assert res.retained_f2 == 2

    def test_validation(self):
        with pytest.raises(ValueError):
            velicer_map(np.ones((3, 2)))
        with pytest.raises(ValueError):
            velicer_map(self.HARMAN8, max_m=8)  # > p - 1


def test_raju_area_unsigned_oracle():
    """Wrapper returns the crate's unsigned areas; pinned to independently
    computed constants (asserts read the crate h/z/p returned via the pyd)."""
    a_f = np.array([1.2, 0.8, 1.5, 0.6])
    b_f = np.array([0.5, 0.0, -1.0, 0.8])
    a_r = np.array([0.8, 1.7, 0.7, 1.4])
    b_r = np.array([-0.3, 0.9, 1.1, -0.4])
    se = np.full(4, 0.1)
    cov = np.zeros(4)
    res = raju_area(a_r, b_r, se, se, cov, a_f, b_f, se, se, cov, alpha=0.05)
    expect = [0.9140059278766983, 1.2023709167732664, 2.193856226242703, 1.6756394651297617]
    assert np.allclose(res["h"], expect, atol=1e-9)
    assert res["signed"] is False
    assert np.all(res["z"] > 0) and np.all(res["p_value"] < 1)
    # Crossing item: unsigned area strictly exceeds |signed area| = 0.8.
    assert res["h"][0] > 0.8 + 1e-3


def test_raju_area_signed_orientation_and_3pl():
    """Signed h = b_foc - b_ref (positive = harder for focal); common-c 3PL
    scales h/se by (1 - c) and leaves z untouched (crate values throughout)."""
    a_r, b_r = np.array([0.8]), np.array([-0.3])
    a_f, b_f = np.array([1.2]), np.array([0.5])
    se, cov = np.array([0.1]), np.array([0.0])
    s = raju_area(a_r, b_r, se, se, cov, a_f, b_f, se, se, cov, signed=True, alpha=0.05)
    assert abs(s["h"][0] - 0.8) < 1e-12
    two = raju_area(a_r, b_r, se, se, cov, a_f, b_f, se, se, cov, alpha=0.05)
    three = raju_area(a_r, b_r, se, se, cov, a_f, b_f, se, se, cov, guess=np.array([0.35]), alpha=0.05)
    assert abs(three["h"][0] - 0.65 * two["h"][0]) < 1e-12
    assert abs(three["se"][0] - 0.65 * two["se"][0]) < 1e-12
    assert abs(three["z"][0] - two["z"][0]) < 1e-12
    # Detection flag reads the crate dif_items: strong DIF must be flagged.
    tiny = np.array([0.01])
    strong = raju_area(a_r, b_r, tiny, tiny, cov, a_f, b_f, tiny, tiny, cov, signed=True, alpha=0.05)
    assert list(strong["dif_items"]) == [0]


def test_raju_area_error_paths():
    ok, se, cov = np.array([1.0]), np.array([0.1]), np.array([0.0])
    b = np.array([0.0])
    with pytest.raises(ValueError, match="discriminations"):
        raju_area(np.array([-1.0]), b, se, se, cov, ok, b, se, se, cov, alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        raju_area(ok, b, se, se, cov, ok, b, se, se, cov, alpha=0.0)
    with pytest.raises(ValueError, match="guessing"):
        raju_area(ok, b, se, se, cov, ok, b, se, se, cov, guess=np.array([1.0]), alpha=0.05)
    with pytest.raises(ValueError, match="length"):
        raju_area(ok, b, se, se, cov, np.array([1.0, 1.0]), b, se, se, cov, alpha=0.05)
    with pytest.raises(ValueError, match="semi-definite"):
        raju_area(ok, b, se, se, np.array([-0.011]), ok, b, se, se, cov, alpha=0.05)
    with pytest.raises(ValueError, match="1-D"):
        raju_area(np.ones((1, 1)), b, se, se, cov, ok, b, se, se, cov, alpha=0.05)


class TestKlInformation:
    """Chang-Ying (1996) KL global-information CAT selection (paper READ via
    Digital Conservancy; cross-checked against catR KL.R). Pinned oracles were
    computed twice independently (Decimal 60-digit Simpson). Every assert
    reads crate outputs returned through the binding."""

    def test_pinned_oracle_and_select(self):
        import fast_mlsirm as fm

        # 2PL pinned area oracle (kills sign/Q-term/quadrature mutations).
        v = fm.kl_information(
            np.array([1.2]), np.array([0.3]), theta0=0.5, delta=1.0
        )
        assert abs(float(v[0]) - 0.114454883565329) < 1e-12

        # Selection-pool oracle: full index vector + argmax + delta = r/sqrt(n).
        a = np.array([0.8, 1.5, 1.0, 2.0, 0.6])
        b = np.array([-1.0, 0.4, 0.0, 1.2, 0.3])
        c = np.array([0.0, 0.0, 0.15, 0.1, 0.0])
        adm = np.zeros(5, dtype=bool)
        r = fm.kl_select(
            a, b, c, administered=adm, theta0=0.25, n_administered=4, r=3.0
        )
        assert r["selected"] == 1
        assert abs(r["delta"] - 1.5) < 1e-15
        expected = np.array([
            0.139656556289429, 0.562390595050374, 0.194256509461839,
            0.349021047637362, 0.0992541089721506,
        ])
        assert np.max(np.abs(r["index"] - expected)) < 1e-12
        # Masking applies to selection only: administered item keeps its index.
        adm2 = adm.copy(); adm2[1] = True
        r2 = fm.kl_select(
            a, b, c, administered=adm2, theta0=0.25, n_administered=4, r=3.0
        )
        assert r2["selected"] == 3
        assert abs(r2["index"][1] - expected[1]) < 1e-12

    def test_validation(self):
        import fast_mlsirm as fm

        a = np.array([1.0, 1.2]); b = np.array([0.0, 0.5])
        adm = np.zeros(2, dtype=bool)
        with pytest.raises(ValueError):
            fm.kl_information(a.reshape(1, 2), b, theta0=0.0, delta=1.0)
        with pytest.raises(ValueError):
            fm.kl_select(a, b, administered=adm, theta0=0.0, n_administered=0)
        with pytest.raises(ValueError):
            fm.kl_select(
                a, b, administered=np.ones(2, dtype=bool),
                theta0=0.0, n_administered=3,
            )


class TestOwenCat:
    """Owen (1975) approximate Bayesian sequential CAT (Rust core).

    Every assert reads crate outputs returned through the binding; oracles
    are the adversarial-spec-review pinned values (exact-posterior numerical
    integration). Killing mutations: update sign flips / wrong denominators
    change the pinned moments; argmin->argmax changes the administered order.
    """

    def test_pinned_update_oracles(self):
        import fast_mlsirm as fm

        mu, s2 = fm.owen_update(1.5, 0.3, 0.0, correct=True, mu=0.2, sig2=1.2)
        assert abs(mu - 0.993708628794091) < 5e-7
        assert abs(s2 - 0.627945890895211) < 5e-7
        mu, s2 = fm.owen_update(1.5, 0.3, 0.0, correct=False, mu=0.2, sig2=1.2)
        assert abs(mu - -0.500813523713129) < 5e-7
        assert abs(s2 - 0.657719958655775) < 5e-7
        mu, s2 = fm.owen_update(1.5, 0.3, 0.2, correct=True, mu=0.2, sig2=1.2)
        assert abs(mu - 0.717701908086462) < 5e-7
        assert abs(s2 - 0.969762981710486) < 5e-7

    def test_cat_trajectory_oracle(self):
        import fast_mlsirm as fm

        a = np.array([0.9, 1.4, 1.1, 2.0, 0.7, 1.6])
        b = np.array([-1.2, 0.5, 0.0, 1.0, -0.4, 0.2])
        c = np.array([0.0, 0.1, 0.0, 0.15, 0.0, 0.0])
        resp = np.array([1, 0, 1, 0, 1, 1])
        r = fm.owen_cat(a, b, c, responses=resp, test_length=4)
        assert r["administered"] == [2, 1, 5, 3]
        assert abs(r["mu"] - 0.33188033525266003) < 1e-5
        assert abs(r["sig2"] - 0.20724352639310067) < 1e-5
        assert abs(r["mu_trace"][0] - 0.5903867604819626) < 1e-5
        assert abs(r["sig2_trace"][1] - 0.4123387131860715) < 1e-5
        # Variance-threshold stopping (Owen's rule).
        r = fm.owen_cat(a, b, c, responses=resp, test_length=4, sig2_stop=0.7)
        assert len(r["administered"]) == 1
        assert r["sig2"] <= 0.7

    def test_validation(self):
        import fast_mlsirm as fm

        a = np.array([1.0, 1.2])
        b = np.array([0.0, 0.5])
        resp = np.array([1, 0])
        with pytest.raises(ValueError):
            fm.owen_update(1.0, 0.0, 1.0, correct=True, mu=0.0, sig2=1.0)
        with pytest.raises(ValueError):
            fm.owen_cat(a, b, responses=resp, test_length=3)
        with pytest.raises(ValueError):
            fm.owen_cat(a, b, responses=np.array([2, 0]), test_length=1)
        with pytest.raises(ValueError):
            fm.owen_cat(a, b, responses=resp, test_length=1, sig2_stop=0.0)

    def test_response_validation_before_cast(self):
        # Impl-review finding: the uint8 cast used to launder 1.2 -> 1,
        # 0.9/-0.2/nan -> 0, and -1 -> 255 past validation.
        import fast_mlsirm as fm

        a = np.array([1.0, 1.2])
        b = np.array([0.0, 0.5])
        for bad in ([1.2, 0], [0.9, 1], [-0.2, 1], [np.nan, 1], [-1, 0]):
            with pytest.raises(ValueError):
                fm.owen_cat(a, b, responses=np.array(bad), test_length=1)

    def test_complex_responses_rejected(self):
        # Round-2 impl-review finding: astype(float64) silently dropped the
        # imaginary part, laundering complex inputs past validation.
        import fast_mlsirm as fm

        with pytest.raises(ValueError):
            fm.owen_cat(
                np.array([1.0, 1.2]),
                np.array([0.0, 0.5]),
                responses=np.array([1 + 2j, 0 + 0j]),
                test_length=1,
            )


class TestCcatSelect:
    """Kingsbury & Zara (1989) CCAT content balancing (Rust core parity)."""

    def _pool(self):
        import numpy as np

        return dict(
            a=np.array([1.0, 1.5, 0.8, 2.0, 1.2, 0.9]),
            b=np.array([-0.5, 0.2, 0.0, 0.8, -0.2, 0.4]),
            c=np.array([0.0, 0.1, 0.0, 0.2, 0.0, 0.0]),
            groups=np.array([0, 0, 1, 1, 0, 1]),
            targets=np.array([0.6, 0.4]),
        )

    def test_pinned_oracle(self):
        # Every assert reads the dict returned by the crate binding.
        import numpy as np

        from fast_mlsirm import ccat_select

        p = self._pool()
        r = ccat_select(
            p["a"],
            p["b"],
            p["c"],
            groups=p["groups"],
            targets=p["targets"],
            administered=np.array([True, False, False, True, False, False]),
            theta0=0.1,
        )
        assert r["group"] == 0
        assert r["selected"] == 1
        assert abs(r["discrepancy"][0] - 0.1) < 1e-12
        assert abs(r["info"][1] - 0.451012779418390198) < 1e-12
        assert abs(r["info"][4] - 0.348583393587808097) < 1e-12

    def test_balancing_overrides_global_max_info(self):
        import numpy as np

        from fast_mlsirm import ccat_select

        p = self._pool()
        r = ccat_select(
            p["a"],
            p["b"],
            p["c"],
            groups=p["groups"],
            targets=p["targets"],
            administered=np.array([True, False, False, False, False, False]),
            theta0=0.1,
        )
        assert r["group"] == 1
        assert r["selected"] == 3
        # unconstrained max-info would have picked item 1
        assert int(np.argmax(np.where([True] + [False] * 5, -np.inf, r["info"]))) == 1

    def test_group_validation_before_cast(self):
        import numpy as np
        import pytest

        from fast_mlsirm import ccat_select

        p = self._pool()
        adm = np.array([False] * 6)
        for bad in (
            np.array([0, 0, 1, 1, 0, -1]),
            np.array([0.0, 0.5, 1.0, 1.0, 0.0, 1.0]),
            np.array([0, 0, 1, 1, 0, 1], dtype=complex),
        ):
            with pytest.raises(ValueError):
                ccat_select(
                    p["a"],
                    p["b"],
                    p["c"],
                    groups=bad,
                    targets=p["targets"],
                    administered=adm,
                    theta0=0.0,
                )

    def test_error_paths(self):
        import numpy as np
        import pytest

        from fast_mlsirm import ccat_select

        p = self._pool()
        with pytest.raises(ValueError):
            ccat_select(
                p["a"],
                p["b"],
                p["c"],
                groups=p["groups"],
                targets=np.array([0.5, 0.6]),
                administered=np.array([False] * 6),
                theta0=0.0,
            )
        with pytest.raises(ValueError):
            ccat_select(
                p["a"],
                p["b"],
                p["c"],
                groups=p["groups"],
                targets=p["targets"],
                administered=np.array([True] * 6),
                theta0=0.0,
            )


class TestEpvSelect:
    """Owen-approximate posterior-predictive EPV selection (Rust core parity)."""

    def test_pinned_oracle(self):
        # Every assert reads the dict returned by the crate binding.
        import numpy as np

        from fast_mlsirm import epv_select

        r = epv_select(
            np.array([1.0, 1.5, 0.8, 2.0, 1.2]),
            np.array([-0.5, 0.2, 0.0, 0.8, -0.2]),
            np.array([0.0, 0.1, 0.0, 0.2, 0.0]),
            administered=np.array([False] * 5),
            mu=0.3,
            sig2=0.8,
        )
        assert r["selected"] == 1
        predictive_oracle = [
            0.72450752942669350,
            0.58214281295605108,
            0.57737397511781730,
            0.45023407596279880,
            0.65873253189075243,
        ]
        epv_oracle = [
            0.60131771422419289,
            0.52962761721829099,
            0.62991332336282158,
            0.60231616694192736,
            0.54351330867142256,
        ]
        for i in range(5):
            assert abs(r["predictive"][i] - predictive_oracle[i]) < 5e-7
            assert abs(r["epv"][i] - epv_oracle[i]) < 5e-7

    def test_discriminates_from_max_info_and_b_matching(self):
        # Spec-review delegation discriminator: argmin EPV = 2 while
        # max-Fisher-info = 4 and Owen b-matching = 1 in this pool.
        import numpy as np

        from fast_mlsirm import epv_select

        r = epv_select(
            np.array([
                0.71127789879824621,
                0.64854445799599558,
                1.7654112699885889,
                0.83742872778938249,
                0.88322558143136509,
            ]),
            np.array([
                -1.1985815760644314,
                0.13372726004391877,
                -1.4174170834573148,
                -2.1649158509201518,
                -0.7232085493428142,
            ]),
            np.array([0.0, 0.0017367688601781506, 0.032298449910040355, 0.10888354386121427, 0.0]),
            administered=np.array([False] * 5),
            mu=-0.18196673524946427,
            sig2=0.5152733572547918,
        )
        assert r["selected"] == 2
        assert r["selected"] != 1  # b-matching pick
        assert r["selected"] != 4  # max-info pick
        assert abs(r["epv"][2] - 0.41771608214541328) < 5e-7

    def test_administered_masking(self):
        import numpy as np

        from fast_mlsirm import epv_select

        a = np.array([1.0, 1.5, 0.8, 2.0, 1.2])
        b = np.array([-0.5, 0.2, 0.0, 0.8, -0.2])
        c = np.array([0.0, 0.1, 0.0, 0.2, 0.0])
        full = epv_select(a, b, c, administered=np.array([False] * 5), mu=0.3, sig2=0.8)
        masked = epv_select(
            a, b, c, administered=np.array([False, True, False, False, False]), mu=0.3, sig2=0.8
        )
        assert masked["selected"] == 4
        # scoring covers the whole pool; masking affects selection only
        assert np.allclose(masked["epv"], full["epv"], atol=1e-15, rtol=0.0)
        assert np.allclose(masked["predictive"], full["predictive"], atol=1e-15, rtol=0.0)

    def test_administered_degenerate_item_is_ignored(self):
        import numpy as np

        from fast_mlsirm import epv_select

        r = epv_select(
            np.array([1.0, 1.2]),
            np.array([-1000.0, 0.5]),
            np.array([0.0, 0.1]),
            administered=np.array([True, False]),
            mu=0.0,
            sig2=1.0,
        )
        assert r["selected"] == 1
        assert np.isnan(r["epv"][0])

    def test_error_paths(self):
        import numpy as np
        import pytest

        from fast_mlsirm import epv_select

        a = np.array([1.0, 1.2])
        b = np.array([0.0, 0.5])
        c = np.array([0.0, 0.1])
        adm = np.array([False, False])
        with pytest.raises(ValueError):
            epv_select(a, b[:1], c, administered=adm, mu=0.0, sig2=1.0)
        with pytest.raises(ValueError):
            epv_select(np.array([1.0, -0.5]), b, c, administered=adm, mu=0.0, sig2=1.0)
        with pytest.raises(ValueError):
            epv_select(a, b, c, administered=adm, mu=0.0, sig2=0.0)
        with pytest.raises(ValueError):
            epv_select(a, b, c, administered=np.array([True, True]), mu=0.0, sig2=1.0)
        with pytest.raises(ValueError):
            epv_select(a, b, c, administered=np.array([0, 0]), mu=0.0, sig2=1.0)
        with pytest.raises(ValueError):
            epv_select(a, b, c, administered=np.array([[False, False]]), mu=0.0, sig2=1.0)
        # complex laundering guard: rejected BEFORE the float64 cast
        with pytest.raises(ValueError):
            epv_select(np.array([1 + 2j, 1.2]), b, c, administered=adm, mu=0.0, sig2=1.0)
        with pytest.raises(ValueError):
            epv_select(a, np.array([0.0 + 1j, 0.5]), c, administered=adm, mu=0.0, sig2=1.0)
        with pytest.raises(ValueError):
            epv_select(a, b, np.array([0.0, 0.1 + 1j]), administered=adm, mu=0.0, sig2=1.0)

class TestSprtClassify:
    """sprt_classify wrapper: every assert reads crate-returned dict values."""

    def test_pinned_oracle(self):
        from fast_mlsirm import sprt_classify

        r = sprt_classify(
            np.full(5, 2.0),
            np.zeros(5),
            np.full(5, 0.1),
            responses=np.array([1, 1, 1, 1, 0]),
            theta_cut=0.0,
            delta=0.5,
            alpha=0.05,
            beta=0.05,
        )
        assert r["decision"] == "above"
        assert r["n_used"] == 4
        trace = np.array(
            [
                0.79567203915954553,
                1.5913440783190911,
                2.3870161174786366,
                3.1826881566381821,
                2.1826881566381821,
            ]
        )
        np.testing.assert_allclose(r["llr_trace"], trace, rtol=0, atol=5e-15)
        assert abs(r["llr"] - 2.1826881566381821) < 5e-15

    def test_default_c_and_bool_responses(self):
        from fast_mlsirm import sprt_classify

        r = sprt_classify(
            np.array([1.0, 1.1]),
            np.array([0.0, 0.2]),
            responses=np.array([True, False]),
            theta_cut=0.0,
            delta=0.3,
        )
        assert r["decision"] == "continue"
        assert r["n_used"] == 2
        assert r["llr_trace"].shape == (2,)
        assert r["llr"] == r["llr_trace"][-1]

    def test_rejects_bad_responses(self):
        from fast_mlsirm import sprt_classify

        with pytest.raises(ValueError, match="0 or 1"):
            sprt_classify(
                np.array([1.0, 1.0]),
                np.zeros(2),
                responses=np.array([1, 2]),
                theta_cut=0.0,
                delta=0.5,
            )
        with pytest.raises(ValueError, match="0 or 1"):
            sprt_classify(
                np.array([1.0, 1.0]),
                np.zeros(2),
                responses=np.array([1.0, 0.5]),
                theta_cut=0.0,
                delta=0.5,
            )

    def test_rejects_complex_input(self):
        from fast_mlsirm import sprt_classify

        with pytest.raises(ValueError, match="real numeric array"):
            sprt_classify(
                np.array([1.0 + 1j, 1.0]),
                np.zeros(2),
                responses=np.array([1, 0]),
                theta_cut=0.0,
                delta=0.5,
            )

    def test_core_validation_propagates(self):
        from fast_mlsirm import sprt_classify

        with pytest.raises(ValueError, match="alpha \\+ beta"):
            sprt_classify(
                np.array([1.0, 1.0]),
                np.zeros(2),
                responses=np.array([1, 0]),
                theta_cut=0.0,
                delta=0.5,
                alpha=0.6,
                beta=0.5,
            )

class TestCiClassify:
    """ci_classify wrapper: every assert reads crate-returned dict values."""

    def test_pinned_oracle(self):
        from fast_mlsirm import ci_classify

        r = ci_classify(
            np.full(6, 1.5),
            np.array([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]),
            np.zeros(6),
            responses=np.array([1, 1, 1, 1, 1, 0]),
            theta_cut=0.0,
            z_crit=1.6448536269514722,
        )
        assert r["decision"] == "above"
        assert r["n_used"] == 5
        theta = np.array(
            [
                0.18783548849905624,
                0.40433637208107137,
                0.65453031321107147,
                0.93795666218057705,
                1.251068565832161,
                1.004851105902542,
            ]
        )
        se = np.array(
            [
                0.91459937771477151,
                0.84249780260178286,
                0.78082991898905685,
                0.72897935456728113,
                0.68628322205747161,
                0.60091214158918393,
            ]
        )
        np.testing.assert_allclose(r["theta_trace"], theta, rtol=0, atol=1e-12)
        np.testing.assert_allclose(r["se_trace"], se, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            r["lower_trace"], theta - 1.6448536269514722 * se, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            r["upper_trace"], theta + 1.6448536269514722 * se, rtol=0, atol=1e-12
        )
        # First-strict-crossing anchor: k=5 is the first lower bound > cut.
        assert r["lower_trace"][3] <= 0.0 < r["lower_trace"][4]

    def test_default_c_and_bool_responses(self):
        from fast_mlsirm import ci_classify

        r = ci_classify(
            np.array([1.0, 1.1]),
            np.array([0.0, 0.2]),
            responses=np.array([True, False]),
            theta_cut=0.0,
            z_crit=1.96,
        )
        assert r["decision"] == "continue"
        assert r["n_used"] == 2
        assert r["theta_trace"].shape == (2,)
        assert np.all(r["se_trace"] > 0.0)
        assert np.all(r["lower_trace"] <= 0.0)
        assert np.all(r["upper_trace"] >= 0.0)

    def test_rejects_bad_responses(self):
        from fast_mlsirm import ci_classify

        with pytest.raises(ValueError, match="0 or 1"):
            ci_classify(
                np.array([1.0, 1.0]),
                np.zeros(2),
                responses=np.array([1, 2]),
                theta_cut=0.0,
                z_crit=1.96,
            )
        with pytest.raises(ValueError, match="0 or 1"):
            ci_classify(
                np.array([1.0, 1.0]),
                np.zeros(2),
                responses=np.array([1.0, 0.5]),
                theta_cut=0.0,
                z_crit=1.96,
            )

    def test_rejects_complex_input(self):
        from fast_mlsirm import ci_classify

        with pytest.raises(ValueError, match="real numeric array"):
            ci_classify(
                np.array([1.0 + 1j, 1.0]),
                np.zeros(2),
                responses=np.array([1, 0]),
                theta_cut=0.0,
                z_crit=1.96,
            )

    def test_core_validation_propagates(self):
        from fast_mlsirm import ci_classify

        with pytest.raises(ValueError, match="z_crit"):
            ci_classify(
                np.array([1.0, 1.0]),
                np.zeros(2),
                responses=np.array([1, 0]),
                theta_cut=0.0,
                z_crit=0.0,
            )

class TestDimtest:
    """Stout-style DIMTEST (Nandakumar & Stout, 1993 formulas).

    The oracle values were computed by an independent NumPy script (never
    imports the crate); the dataset is regenerated here from the same seed.
    Every assert reads values returned by the crate via the wrapper.
    """

    @staticmethod
    def _fixture():
        import math

        rng = np.random.default_rng(20260726)
        n = 500
        z1 = rng.standard_normal(n)
        z2 = 0.30 * z1 + math.sqrt(1 - 0.09) * rng.standard_normal(n)
        items = [(0, 1.40, b) for b in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        items += [(1, 1.40, b) for b in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        items += [(1, 1.20, b) for b in (-1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4)]
        p = np.empty((n, len(items)))
        for j, (dim, a, b) in enumerate(items):
            th = z1 if dim == 0 else z2
            p[:, j] = 1 / (1 + np.exp(-a * (th - b)))
        return (rng.random(p.shape) < p).astype(np.float64)

    def test_pinned_oracle(self):
        from fast_mlsirm import dimtest

        y = self._fixture()
        r = dimtest(y, at1=np.arange(5), at2=np.arange(5, 10))
        assert abs(r.t_l - 8.5848469411043151) < 1e-12
        assert abs(r.t_b - 3.6579307315481961) < 1e-12
        assert abs(r.t - 3.4838558621150524) < 1e-12
        # crate erfc is a 1.2e-7-accurate approximation; anchor p at 5e-7.
        assert abs(r.p_value - 0.000247122791999742) < 5e-7
        assert r.groups_used == 8
        assert r.n_discarded == 17
        assert r.retained_pt_scores.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_rejects_overlap_and_short_at(self):
        from fast_mlsirm import dimtest

        y = self._fixture()
        with pytest.raises(ValueError, match="duplicates"):
            dimtest(y, at1=[0, 1, 2, 3], at2=[3, 4, 5, 6])
        with pytest.raises(ValueError, match=">= 4"):
            dimtest(y, at1=[0, 1, 2], at2=[3, 4, 5])

    def test_rejects_non_binary_and_complex(self):
        from fast_mlsirm import dimtest

        y = self._fixture()
        bad = y.copy()
        bad[0, 0] = 0.5
        with pytest.raises(ValueError, match="exactly 0 or 1"):
            dimtest(bad, at1=[0, 1, 2, 3, 4], at2=[5, 6, 7, 8, 9])
        with pytest.raises(ValueError, match="real-valued"):
            dimtest(y.astype(complex), at1=[0, 1, 2, 3, 4], at2=[5, 6, 7, 8, 9])

    def test_rejects_fractional_indices(self):
        from fast_mlsirm import dimtest

        y = self._fixture()
        with pytest.raises(ValueError, match="integers"):
            dimtest(y, at1=[0.5, 1, 2, 3], at2=[4, 5, 6, 7])

    def test_rejects_string_responses_and_indices(self):
        # Regression: dtype-kind check must run BEFORE astype casts, so
        # string arrays are rejected rather than silently coerced.
        from fast_mlsirm import dimtest

        y = self._fixture()
        with pytest.raises(ValueError, match="numeric"):
            dimtest(y.astype(str), at1=[0, 1, 2, 3, 4], at2=[5, 6, 7, 8, 9])
        with pytest.raises(ValueError, match="numeric"):
            dimtest(y, at1=["0", "1", "2", "3", "4"], at2=[5, 6, 7, 8, 9])

    def test_too_few_groups_errors(self):
        from fast_mlsirm import dimtest

        y = self._fixture()[:30]
        with pytest.raises(ValueError, match="need at least 2"):
            dimtest(y, at1=[0, 1, 2, 3, 4], at2=[5, 6, 7, 8, 9])

class TestWollackOmega:
    """Omega answer-copying statistic (Wollack, 1997, as implemented by
    CopyDetect/aberrance; oracle values pinned from an independent Python
    computation in the adversarial spec review)."""

    @staticmethod
    def _fixture():
        import numpy as np

        probs = np.array(
            [
                [0.05, 0.10, 0.70, 0.10, 0.05],
                [0.20, 0.20, 0.20, 0.20, 0.20],
                [0.60, 0.15, 0.10, 0.10, 0.05],
                [0.12, 0.38, 0.25, 0.15, 0.10],
                [0.30, 0.25, 0.20, 0.15, 0.10],
                [0.08, 0.12, 0.16, 0.24, 0.40],
                [0.45, 0.05, 0.25, 0.15, 0.10],
                [0.11, 0.22, 0.33, 0.22, 0.12],
                [0.18, 0.32, 0.22, 0.18, 0.10],
                [0.07, 0.14, 0.21, 0.28, 0.30],
            ]
        )
        source = np.array([2, 3, 0, 1, 4, 4, 2, 1, 0, 3])
        copier = np.array([2, 1, 0, 1, 3, 4, 0, 1, 2, 3])
        return copier, source, probs

    def test_pinned_oracle(self):
        # Asserts read WollackOmegaResult fields returned by the crate.
        # Killed by: sqrt(V) vs V, copier-prob lookup, two-sided p,
        # continuity correction (mutation values in the Rust test file).
        from fast_mlsirm import wollack_omega

        copier, source, probs = self._fixture()
        r = wollack_omega(copier, source, probs, 5)
        assert r.observed_matches == 6
        assert abs(r.expected_matches - 3.3100000000000001) < 1e-12
        assert abs(r.variance - 1.8839000000000001) < 1e-12
        assert abs(r.omega - 1.9598523632230238) < 1e-12
        assert abs(r.p_value - 0.02500652442931299) < 5e-7

    def test_flat_probs_equivalent(self):
        from fast_mlsirm import wollack_omega

        copier, source, probs = self._fixture()
        r2 = wollack_omega(copier, source, probs, 5)
        r1 = wollack_omega(copier, source, probs.ravel(), 5)
        assert r1.omega == r2.omega
        assert r1.p_value == r2.p_value

    def test_rejects_bad_inputs(self):
        import numpy as np
        import pytest

        from fast_mlsirm import wollack_omega

        copier, source, probs = self._fixture()
        with pytest.raises(ValueError):
            wollack_omega(np.array(["2", "1"]), source[:2], probs[:2], 5)
        with pytest.raises(ValueError):
            wollack_omega(copier.astype(complex), source, probs, 5)
        with pytest.raises(ValueError):
            wollack_omega(copier + 0.5, source, probs, 5)
        with pytest.raises(ValueError):
            wollack_omega(copier, source, probs.astype(complex), 5)
        with pytest.raises(ValueError):
            wollack_omega(copier, source[:9], probs, 5)
        with pytest.raises(ValueError):
            wollack_omega(copier, source, probs[:, :4], 5)
        with pytest.raises(ValueError):
            wollack_omega(copier, source, probs, 0)
        bad = copier.copy()
        bad[0] = 5
        with pytest.raises(ValueError):
            wollack_omega(bad, source, probs, 5)
        badp = probs.copy()
        badp[0, 0] = 0.5
        with pytest.raises(ValueError):
            wollack_omega(copier, source, badp, 5)

    def test_accepts_float_integer_indices(self):
        from fast_mlsirm import wollack_omega

        copier, source, probs = self._fixture()
        r = wollack_omega(copier.astype(float), source.astype(float), probs, 5)
        assert r.observed_matches == 6
    def test_rejects_bool_indices(self):
        import numpy as np
        import pytest

        from fast_mlsirm import wollack_omega

        copier, source, probs = self._fixture()
        with pytest.raises(ValueError):
            wollack_omega((copier == 2), source, probs, 5)
        with pytest.raises(ValueError):
            wollack_omega(copier, (source == 1), probs, 5)

class TestKIndex:
    def _fixture(self):
        import numpy as np

        return np.array(
            [
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
            ],
            dtype=float,
        )

    def test_pinned_oracle(self):
        # Asserts read KIndexResult fields returned by the crate through the
        # wrapper; oracle computed independently with exact math.comb.
        import numpy as np

        from fast_mlsirm import k_index

        r = k_index(self._fixture(), copier=2, source=7)
        assert r.wc == 6
        assert r.ws == 5
        assert r.m == 2
        assert list(r.subgroup) == [1, 2, 5]
        assert list(r.emp_agg) == [3, 2, 3]
        assert abs(r.p - 0.53333333333333333) < 1e-12
        assert abs(r.k_index - 0.85139489711934158) < 1e-12
        assert isinstance(r.subgroup, np.ndarray)

    def test_validation(self):
        import numpy as np
        import pytest

        from fast_mlsirm import k_index

        x = self._fixture()
        with pytest.raises(ValueError):
            k_index(x, copier=2, source=2)  # identical indices
        with pytest.raises(ValueError):
            k_index(x, copier=2, source=10)  # out of range
        with pytest.raises(ValueError):
            k_index(x, copier=True, source=7)  # bool index
        bad = x.copy()
        bad[0, 0] = 0.5
        with pytest.raises(ValueError):
            k_index(bad, copier=2, source=7)  # non-binary
        with pytest.raises(ValueError):
            k_index(x + 0j, copier=2, source=7)  # complex laundering
        with pytest.raises(ValueError):
            k_index(x.astype(bool), copier=2, source=7)  # bool matrix
        allc = x.copy()
        allc[4, :] = 1.0
        with pytest.raises(ValueError):
            k_index(allc, copier=2, source=4)  # ws == 0 degenerate

    def test_m_zero_gives_k_one(self):
        import numpy as np

        from fast_mlsirm import k_index

        x = np.ones((3, 10))
        x[0, 9] = 0.0
        x[1, 0] = 0.0
        x[1, 1] = 0.0
        r = k_index(x, copier=0, source=1)
        assert r.m == 0
        assert r.k_index == 1.0

class TestGbt:
    """GBT tail kernel: asserts read fast_mlsirm.gbt outputs (crate-backed).

    Oracle: exact Python fractions.Fraction reference from the adversarial
    spec review (files/gbt_spec_review.md); kills tail-direction,
    off-by-one, and convolution-mixing mutants.
    """

    def test_pinned_oracle(self):
        import numpy as np

        from fast_mlsirm import gbt

        matches = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int64)
        probs = np.array(
            [0.62, 0.55, 0.48, 0.71, 0.52, 0.66, 0.43, 0.58, 0.73, 0.49]
        )
        r = gbt(matches, probs)
        assert r.observed_matches == 7
        assert abs(r.p_value - 0.32225898631286054) < 1e-12
        assert r.match_dist.shape == (11,)
        assert abs(r.match_dist[0] - 0.00013873169507258882) < 1e-12
        assert abs(r.match_dist[7] - 0.19449839960529633) < 1e-12
        assert abs(r.match_dist.sum() - 1.0) < 1e-12

    def test_structural_obs_zero_tail_is_one(self):
        import numpy as np

        from fast_mlsirm import gbt

        r = gbt(np.zeros(4), np.array([0.3, 0.9, 0.5, 0.1]))
        assert r.observed_matches == 0
        assert r.p_value == 1.0

    def test_error_paths(self):
        import numpy as np
        import pytest

        from fast_mlsirm import gbt

        with pytest.raises(ValueError):
            gbt(np.array([1.0, 0.5]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            gbt(np.array([1.0, 0.0]), np.array([0.5, 1.5]))
        with pytest.raises(ValueError):
            gbt(np.array([1.0, 0.0]), np.array([0.5]))
        with pytest.raises(ValueError):
            gbt(np.array([1.0 + 0j, 0.0]), np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            gbt(np.array([True, False]), np.array([0.5, 0.5]))

class TestKVariants:
    """K1/K2/S1/S2 indices: asserts read fast_mlsirm.k_variants outputs
    (crate-backed).

    Oracle: independent Python reference confirmed by the adversarial spec
    review (files/k_variants_spec_review.md); the pinned values kill
    source-inclusion, weight-exponent-sign, and strict-tail mutants.
    """

    def _fixture(self):
        import numpy as np

        rng = np.random.default_rng(42)
        x = (rng.random((15, 12)) < 0.6).astype(float)
        x[2, :5] = x[7, :5]
        return x

    def test_pinned_oracle(self):
        import numpy as np

        from fast_mlsirm import k_variants

        r = k_variants(self._fixture(), copier=2, source=7)
        assert r.wc == 4
        assert r.ws == 3
        assert r.m == 2
        assert r.mm == 3
        assert abs(r.p1 - 0.43862433862433853) < 1e-10
        assert abs(r.p2 - 0.38571428571428595) < 1e-10
        assert abs(r.s1 - 1.2577145969460732) < 1e-8
        assert abs(r.s2 - 2.2808598308064885) < 1e-8
        assert abs(r.k1 - 0.4083989087088663) < 1e-10
        assert abs(r.k2 - 0.33155685131195367) < 1e-10
        assert abs(r.s1_index - 0.3191324670108421) < 1e-8
        assert abs(r.s2_index - 0.39887838757590066) < 1e-8
        assert r.pr.shape == (13,)
        assert r.pj.shape == (13,)
        # source excluded from its own subgroup: pr[3] uses rows other
        # than the source (independent oracle value 2/9, not 1)
        assert abs(r.pr[3] - 2.0 / 9.0) < 1e-12
        assert np.isnan(r.pr[0])
        assert isinstance(r.pr, np.ndarray)

    def test_error_paths(self):
        import numpy as np
        import pytest

        from fast_mlsirm import k_variants

        x = self._fixture()
        with pytest.raises(ValueError):
            k_variants(x, copier=2, source=2)  # identical indices
        with pytest.raises(ValueError):
            k_variants(x, copier=2, source=15)  # out of range
        with pytest.raises(ValueError):
            k_variants(x, copier=True, source=7)  # bool index
        bad = x.copy()
        bad[0, 0] = 0.5
        with pytest.raises(ValueError):
            k_variants(bad, copier=2, source=7)  # non-binary
        with pytest.raises(ValueError):
            k_variants(x + 0j, copier=2, source=7)  # complex laundering
        with pytest.raises(ValueError):
            k_variants(x.astype(bool), copier=2, source=7)  # bool matrix
        allc = x.copy()
        allc[7, :] = 1.0
        with pytest.raises(ValueError):
            k_variants(allc, copier=2, source=7)  # ws == 0 degenerate

    def test_result_fields(self):
        from fast_mlsirm import KVariantsResult, k_variants

        r = k_variants(self._fixture(), copier=2, source=7)
        assert isinstance(r, KVariantsResult)
        for field in ("wc", "ws", "m", "mm"):
            assert isinstance(getattr(r, field), int)
        for field in (
            "p1",
            "p2",
            "s1",
            "s2",
            "k1",
            "k2",
            "s1_index",
            "s2_index",
        ):
            v = getattr(r, field)
            assert isinstance(v, float)
            assert 0.0 <= v or field in ("s1", "s2")

class TestHofstee:
    def test_pinned_oracle_main(self):
        import numpy as np

        from fast_mlsirm import hofstee

        rng = np.random.default_rng(2026)
        scores = np.round(np.clip(rng.normal(68, 9, 40), 0, 100), 1)
        r = hofstee(scores, 62.5, 75.0, 0.0, 20.0)
        assert r.failed is False
        assert abs(r.cut_score - 62.804878048780488) < 1e-12
        assert abs(r.fail_rate - 19.512195121951219) < 1e-12
        assert r.cum_freq_percent.shape == (101,)
        assert r.cum_freq_percent[62] == 17.5
        assert r.cum_freq_percent[63] == 20.0
        # divide-first arithmetic order (23/40)*100:
        assert r.cum_freq_percent[70] == 57.49999999999999

    def test_fallback_and_reduced_scope(self):
        import numpy as np

        from fast_mlsirm import hofstee

        r = hofstee(np.full(10, 30.0), 62.5, 75.0, 0.0, 20.0)
        assert (r.cut_score, r.fail_rate, r.failed) == (62.5, 100.0, True)
        r = hofstee([30.0, 30.0, 90.0], 62.5, 75.0, 0.0, 20.0)
        assert (r.cut_score, r.fail_rate, r.failed) == (62.5, 66.67, True)
        import pytest

        with pytest.raises(ValueError, match="zero-length"):
            hofstee([50.0], 70.0, 70.0, 20.0, 20.0)
        with pytest.raises(ValueError, match="collinear overlap"):
            hofstee([10.0, 90.0], 40.0, 60.0, 50.0, 50.0)

    def test_error_paths(self):
        import numpy as np
        import pytest

        from fast_mlsirm import hofstee

        with pytest.raises(ValueError):
            hofstee([], 62.5, 75.0, 0.0, 20.0)
        with pytest.raises(ValueError):
            hofstee([np.nan], 62.5, 75.0, 0.0, 20.0)
        with pytest.raises(ValueError):
            hofstee([50.0 + 1j], 62.5, 75.0, 0.0, 20.0)
        with pytest.raises(ValueError):
            hofstee([[50.0]], 62.5, 75.0, 0.0, 20.0)
        with pytest.raises(ValueError):
            hofstee([50.0], 62.5, 75.0, 0.0, True)
        with pytest.raises(ValueError):
            hofstee([101.0], 62.5, 75.0, 0.0, 20.0)
        with pytest.raises(ValueError):
            hofstee([50.0], 75.0, 62.5, 0.0, 20.0)

class TestPersonFitNp:
    def test_pinned_main_fixture(self):
        import numpy as np

        from fast_mlsirm import person_fit_np

        rows = [
            "11000000", "01010110", "11010000", "11111011", "11001100",
            "11111010", "11101101", "11100001", "01000101", "00001111",
            "00000000", "11111111",
        ]
        x = np.array([[int(c) for c in r] for r in rows], dtype=float)
        res = person_fit_np(x)
        assert res.g.tolist() == [0, 10, 4, 4, 0, 6, 0, 4, 4, 10, 0, 0]
        assert abs(res.gnormed[1] - 0.625) < 1e-12
        assert abs(res.nci[3] - (-0.1428571428571428)) < 1e-12
        assert abs(res.u3[9] - 0.7968163928347531) < 1e-12
        assert abs(res.zu3[9] - 2.315959529860312) < 1e-12
        assert abs(res.c_sato[9] - 1.5555555555555556) < 1e-12
        assert abs(res.cstar[9] - 0.7777777777777777) < 1e-12
        # Perfect rows: G/Gnormed/NCI = 0; U3/ZU3/C/C* NaN.
        for p in (10, 11):
            assert res.g[p] == 0 and res.gnormed[p] == 0 and res.nci[p] == 0
            assert np.isnan(res.u3[p]) and np.isnan(res.zu3[p])
            assert np.isnan(res.c_sato[p]) and np.isnan(res.cstar[p])

    def test_validation(self):
        import numpy as np
        import pytest

        from fast_mlsirm import person_fit_np

        with pytest.raises(ValueError):
            person_fit_np(np.array([0.0, 1.0]))  # 1-D
        with pytest.raises(ValueError):
            person_fit_np(np.array([[1.0, 0.5]]))  # non-binary
        with pytest.raises(ValueError):
            person_fit_np(np.array([[1.0, np.nan]]))  # missing out of scope
        with pytest.raises(ValueError):
            person_fit_np(np.array([[1.0 + 0j, 0.0]]))  # complex laundering
        with pytest.raises(ValueError):
            person_fit_np(np.array([[1.0], [0.0]]))  # single item

    def test_bool_input_accepted(self):
        import numpy as np

        from fast_mlsirm import person_fit_np

        x = np.array([[True, True, False], [False, True, True]])
        res = person_fit_np(x)
        assert res.g.shape == (2,)

class TestDeltaPlot:
    """Angoff Delta plot DIF (deltaPlotR response-type port).

    Every assertion reads values returned by the crate through the
    Python wrapper; pinned values come from a NumPy oracle
    transcription of deltaPlot.R with exact normal quantiles (crate
    uses Acklam's approximation, hence 1e-6 tolerances)."""

    @staticmethod
    def _main_fixture():
        import numpy as np

        rng = np.random.default_rng(2037)
        n, ni = 80, 10
        theta = rng.normal(0, 1, n)
        group = np.array([0] * 40 + [1] * 40)
        bd = np.linspace(-1.5, 1.5, ni)
        eta = theta[:, None] - bd[None, :]
        eta[group == 1, 3] -= 2.2
        resp = (rng.random((n, ni)) < 1 / (1 + np.exp(-eta))).astype(float)
        return resp, group

    def test_main_fixture_pins(self):
        import numpy as np

        from fast_mlsirm import delta_plot

        resp, group = self._main_fixture()
        r = delta_plot(resp, group, alpha=0.05, max_iter=10)
        assert abs(r.props[3, 0] - 0.625) < 1e-15
        assert abs(r.props[3, 1] - 0.225) < 1e-15
        assert abs(r.deltas[3, 0] - 11.7254425441425) < 1e-6
        assert abs(r.axis_par[0, 1] - 1.0498623805872538) < 1e-6
        assert abs(r.dist[0, 3] - -2.968831851786941) < 1e-6
        assert abs(r.thresholds[0] - 2.4061275783230864) < 1e-6
        assert list(r.dif_items) == [3]
        assert r.n_iter == 1 and r.converged
        assert r.dist.shape == (1, 10)
        assert np.isfinite(r.dist).all()

    def test_purification_and_fixed_threshold(self):
        from fast_mlsirm import delta_plot

        resp, group = self._main_fixture()
        r3 = delta_plot(resp, group, purify="IPP3", alpha=0.05, max_iter=10)
        assert r3.n_iter == 2 and r3.converged
        assert list(r3.dif_items) == [3]
        assert abs(r3.thresholds[1] - 1.2874360054118958) < 1e-6
        rf = delta_plot(resp, group, threshold="fixed", fixed_threshold=1.0, alpha=0.05, max_iter=10)
        assert list(rf.dif_items) == [3, 7, 8]

    def test_input_validation(self):
        import numpy as np
        import pytest

        from fast_mlsirm import delta_plot

        resp, group = self._main_fixture()
        with pytest.raises(ValueError):
            delta_plot(resp[:, :1], group, alpha=0.05, max_iter=10)
        with pytest.raises(ValueError):
            delta_plot(resp, group[:-1], alpha=0.05, max_iter=10)
        with pytest.raises(ValueError):
            delta_plot(resp + 0j, group, alpha=0.05, max_iter=10)  # complex laundering
        bad = resp.copy()
        bad[0, 0] = 2.0
        with pytest.raises(ValueError):
            delta_plot(bad, group, alpha=0.05, max_iter=10)
        with pytest.raises(ValueError):
            delta_plot(resp, np.full_like(group, 2), alpha=0.05, max_iter=10)
        with pytest.raises(ValueError):
            delta_plot(resp, group, threshold="bogus", alpha=0.05, max_iter=10)
        with pytest.raises(ValueError):
            delta_plot(resp, group, purify="IPP9", alpha=0.05, max_iter=10)
        with pytest.raises(ValueError):
            delta_plot(resp, group, alpha=0.0, max_iter=10)
        with pytest.raises(ValueError):
            delta_plot(resp, group, extreme="add", nr_add=0, alpha=0.05, max_iter=10)

    def test_missing_and_add_mode(self):
        import numpy as np

        from fast_mlsirm import delta_plot

        resp, group = self._main_fixture()
        miss = resp.copy()
        miss[0, 0] = np.nan
        r = delta_plot(miss, group, alpha=0.05, max_iter=10)
        expect = resp[1:40, 0].sum() / 39.0
        assert abs(r.props[0, 0] - expect) < 1e-15
        # add-mode: exact-1 ref proportion becomes (sum+1)/(n+2)
        small = np.array(
            [[1, 1], [1, 0], [1, 1], [1, 0], [0, 0], [1, 1], [1, 1], [0, 1]],
            dtype=float,
        )
        g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ra = delta_plot(
            small, g, extreme="add", nr_add=1, threshold="fixed", fixed_threshold=9.9,
            alpha=0.05, max_iter=10,
        )
        assert abs(ra.adj_props[0, 0] - 5.0 / 6.0) < 1e-15

class TestEbMhDif:
    """Empirical Bayes MH DIF (Zwick & Thayer, 2003, ED481063). Pins read
    crate-returned values; oracle in the session spec artifacts."""

    def test_pinned_main_fixture(self):
        import numpy as np

        from fast_mlsirm import eb_mh_dif

        r = eb_mh_dif([1.2, -0.4, 0.3, -2.1], [0.5, 0.8, 0.4, 1.0])
        assert abs(r.mu - (-0.25)) < 1e-12
        assert abs(r.tau2 - 1.4375) < 1e-12
        assert abs(r.weight[0] - 23.0 / 27.0) < 1e-12
        assert abs(r.post_mean[3] - (-523.0 / 390.0)) < 1e-12
        assert abs(r.post_var[0] - 23.0 / 108.0) < 1e-12
        assert r.cat_probs.shape == (4, 5)
        # Item 4 C- probability (independent math.erfc oracle).
        assert abs(r.cat_probs[3, 0] - 0.4180002533326439) < 5e-7
        assert np.allclose(r.cat_probs.sum(axis=1), 1.0, atol=1e-9)

    def test_degenerate_floor(self):
        from fast_mlsirm import eb_mh_dif

        r = eb_mh_dif([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        assert r.tau2 == 0.0
        assert abs(r.tau2_raw - (-1.0)) < 1e-12
        assert list(r.cat_probs[0]) == [0.0, 0.0, 1.0, 0.0, 0.0]

    def test_validation_errors(self):
        import numpy as np
        import pytest

        from fast_mlsirm import eb_mh_dif

        with pytest.raises(ValueError):
            eb_mh_dif([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            eb_mh_dif([1.0], [1.0])
        with pytest.raises(ValueError):
            eb_mh_dif([np.nan, 1.0], [1.0, 1.0])
        with pytest.raises(ValueError):
            eb_mh_dif([1.0, 2.0], [0.0, 1.0])
        with pytest.raises(ValueError):
            eb_mh_dif(np.array([1 + 2j, 1.0]), [1.0, 1.0])
        with pytest.raises(ValueError):
            eb_mh_dif([[1.0, 2.0]], [1.0, 1.0])

class TestMantelSmd:
    """Mantel (1963) polytomous DIF + SMD (Zwick, Donoghue & Grima, 1993)."""

    def _fixture(self):
        import numpy as np

        y = np.array(
            [
                [2, 1, 0], [1, 0, 2], [2, 2, 1], [0, 1, 1], [1, 2, 0], [2, 0, 0],
                [1, 1, 1], [0, 0, 0], [2, 1, 0], [0, 2, 2], [0, 1, 0], [1, 0, 1],
            ]
        )
        group = np.array([0] * 6 + [1] * 6)
        return y, group

    def test_pinned_fixture(self):
        import numpy as np
        from fast_mlsirm import mantel_smd_dif

        y, group = self._fixture()
        res = mantel_smd_dif(y, group)
        # Exact-rational pins from the session oracle (crate values asserted).
        np.testing.assert_allclose(res["chi2"], [3 / 77, 5 / 37, 2 / 133], atol=1e-12)
        np.testing.assert_allclose(res["smd"], [1 / 9, -1 / 6, 1 / 18], atol=1e-12)
        assert list(res["n_strata_used"]) == [2, 2, 2]
        assert np.all((res["p_value"] > 0) & (res["p_value"] < 1))

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import mantel_smd_dif

        y, group = self._fixture()
        with pytest.raises(ValueError):
            mantel_smd_dif(y.astype(complex), group)
        with pytest.raises(ValueError):
            mantel_smd_dif(y + 0.5, group)
        with pytest.raises(ValueError):
            mantel_smd_dif(-y, group)
        yn = y.astype(float)
        yn[0, 0] = np.nan
        with pytest.raises(ValueError):
            mantel_smd_dif(yn, group)
        with pytest.raises(ValueError):
            mantel_smd_dif(y, np.zeros(12, dtype=int))
        with pytest.raises(ValueError):
            mantel_smd_dif(y, group[:5])
        with pytest.raises(ValueError):
            mantel_smd_dif(y.ravel(), group)
        big = y.astype(np.int64).copy()
        big[0, 0] = 2**53 + 1  # would round to 2**53 under a float64 cast
        with pytest.raises(ValueError):
            mantel_smd_dif(big, group)

class TestGmhDif:
    """GMH nominal DIF (Eq. 10; Zwick, Donoghue & Grima, 1993)."""

    def _fixture(self):
        import numpy as np

        y = np.array(
            [
                [2, 1, 0], [1, 0, 2], [2, 2, 1], [0, 1, 1], [1, 2, 0], [2, 0, 0],
                [1, 1, 1], [0, 0, 0], [2, 1, 0], [0, 2, 2], [0, 1, 0], [1, 0, 1],
            ]
        )
        group = np.array([0] * 6 + [1] * 6)
        return y, group

    def test_pinned_fixture(self):
        import numpy as np
        from fast_mlsirm import gmh_dif

        y, group = self._fixture()
        res = gmh_dif(y, group)
        # Exact-rational pins from the session oracle (crate values asserted).
        np.testing.assert_allclose(
            res["chi2"], [53 / 79, 352 / 483, 1072 / 483], atol=1e-12
        )
        assert list(res["df"]) == [2, 2, 2]
        assert list(res["n_strata_used"]) == [2, 2, 2]
        assert np.all((res["p_value"] > 0) & (res["p_value"] < 1))

    def test_dichotomous_matches_mantel(self):
        import numpy as np
        from fast_mlsirm import gmh_dif, mantel_smd_dif

        y = np.array(
            [
                [1, 0], [1, 1], [1, 0], [0, 1], [1, 1], [1, 0], [0, 0], [1, 1],
                [0, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [0, 1], [1, 0],
            ]
        )
        group = np.array([0] * 8 + [1] * 8)
        g = gmh_dif(y, group)
        m = mantel_smd_dif(y, group)
        np.testing.assert_allclose(g["chi2"], m["chi2"], atol=1e-12)
        np.testing.assert_allclose(g["chi2"], [0.98, 0.98], atol=1e-12)
        assert list(g["df"]) == [1, 1]

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import gmh_dif

        y, group = self._fixture()
        with pytest.raises(ValueError):
            gmh_dif(y.astype(complex), group)
        with pytest.raises(ValueError):
            gmh_dif(y + 0.5, group)
        with pytest.raises(ValueError):
            gmh_dif(-y, group)
        yn = y.astype(float)
        yn[0, 0] = np.nan
        with pytest.raises(ValueError):
            gmh_dif(yn, group)
        with pytest.raises(ValueError):
            gmh_dif(y, np.zeros(12, dtype=int))
        with pytest.raises(ValueError):
            gmh_dif(y, group[:5])
        with pytest.raises(ValueError):
            gmh_dif(y.ravel(), group)
        big = y.astype(np.int64).copy()
        big[0, 0] = 2**53 + 1  # would round to 2**53 under a float64 cast
        with pytest.raises(ValueError):
            gmh_dif(big, group)

class TestBreslowDay:
    """Breslow-Day (1980, Eq. 4.30) odds-ratio homogeneity DIF test."""

    def _fixture(self):
        import numpy as np

        # 24x4 fixture from the session oracle; first 12 rows reference,
        # last 12 focal. Same fixture as the Rust bd_fixture().
        y = np.array(
            [
                [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1],
                [1, 1, 1, 0], [0, 0, 1, 0], [1, 1, 0, 1], [0, 1, 0, 0],
                [1, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 0, 0],
                [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0],
                [1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1],
                [0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0],
            ]
        )
        group = np.array([0] * 12 + [1] * 12)
        return y, group

    def test_pinned_fixture(self):
        import numpy as np
        from fast_mlsirm import breslow_day_dif

        y, group = self._fixture()
        res = breslow_day_dif(y, group)
        # Exact pins from the session oracle (crate values asserted). The
        # chi2 pins discriminate the MH plug-in from an unconditional-MLE
        # psi-hat (item 0: 0.3760173495 vs 0.3740794014) and kill wrong-root,
        # dropped-variance-reciprocal, and stratum-OR mutations; the df pins
        # kill df = K.
        np.testing.assert_allclose(
            res["alpha_mh"], [14 / 3, 23 / 26, 39 / 49, 5 / 9], atol=1e-12
        )
        np.testing.assert_allclose(
            res["chi2"],
            [
                0.37601734952436118,
                1.59543251534123905,
                0.41398523247143159,
                3.98273278105957981,
            ],
            atol=1e-10,
        )
        np.testing.assert_allclose(res["df"], [1.0, 2.0, 2.0, 2.0], atol=0)
        np.testing.assert_allclose(
            res["p_value"],
            [
                0.5397424284850997,
                0.4503562883177753,
                0.8130256531551879,
                0.1365087736593863,
            ],
            atol=1e-10,
        )
        assert list(res["n_strata_used"]) == [2, 3, 3, 3]
        assert res["flagged_bh"].dtype == bool
        assert not res["flagged_bh"].any()

    def test_psi_one_mirrored_groups(self):
        import numpy as np
        from fast_mlsirm import breslow_day_dif

        # Focal block identical to the reference block -> alpha_mh = 1
        # exactly (psi = 1 linear branch) and chi2 = 0, p = 1. Same mirrored
        # fixture as the Rust breslow_day_psi_one_linear_branch test.
        y_ref, _ = self._fixture()
        y_half = y_ref[:12]
        y = np.vstack([y_half, y_half])
        group = np.array([0] * 12 + [1] * 12)
        res = breslow_day_dif(y, group)
        np.testing.assert_allclose(res["alpha_mh"], [1.0] * 4, atol=0)
        np.testing.assert_allclose(res["chi2"], [0.0] * 4, atol=1e-12)
        np.testing.assert_allclose(res["p_value"], [1.0] * 4, atol=1e-12)

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import breslow_day_dif

        y, group = self._fixture()
        with pytest.raises(ValueError):
            breslow_day_dif(y.astype(complex), group)
        with pytest.raises(ValueError):
            breslow_day_dif(y + 0.5, group)
        with pytest.raises(ValueError):
            breslow_day_dif(y * 2, group)  # codes other than 0/1
        yn = y.astype(float)
        yn[0, 0] = np.nan
        with pytest.raises(ValueError):
            breslow_day_dif(yn, group)
        with pytest.raises(ValueError):
            breslow_day_dif(y, np.zeros(24, dtype=int))
        with pytest.raises(ValueError):
            breslow_day_dif(y, group[:5])
        with pytest.raises(ValueError):
            breslow_day_dif(y.ravel(), group)
        with pytest.raises(ValueError):
            breslow_day_dif(y, group, fdr_q=0.0)
        big = y.astype(np.int64).copy()
        big[0, 0] = 2**53 + 1  # would round to 2**53 under a float64 cast
        with pytest.raises(ValueError):
            breslow_day_dif(big, group)

class TestFlexilevel:
    def test_worked_example_routing_pin(self):
        from fast_mlsirm import flexilevel_administer

        # Lord (1971) worked example RWWRWRRRWR on N = 19: administered
        # columns [9, 10, 8, 7, 11, 6, 12, 13, 14, 5]; blue, r = 6, x = 6.
        cols = [9, 10, 8, 7, 11, 6, 12, 13, 14, 5]
        answers = [1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
        row = np.zeros((1, 19))
        for c, y in zip(cols, answers):
            row[0, c] = y
        r = flexilevel_administer(row, n_persons=1, n_items=19)
        assert r["n_administered"] == 10
        assert r["items"].tolist() == cols
        assert r["number_right"].tolist() == [6]
        assert r["is_red"].tolist() == [0]
        assert r["score"].tolist() == [6.0]

    def test_distribution_exact_pin(self):
        from fast_mlsirm import flexilevel_score_distribution

        # N = 5 exact oracle pin (enumeration == recursion in the spec
        # oracle): mean 7/4, variance 71/240.
        d = flexilevel_score_distribution([4 / 5, 2 / 3, 1 / 2, 1 / 3, 1 / 5])
        assert d["scores"].tolist() == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        expect = [1 / 30, 2 / 15, 1 / 3, 1 / 3, 2 / 15, 1 / 30]
        assert np.allclose(d["probs"], expect, atol=1e-15)
        assert abs(d["mean"] - 1.75) < 1e-15
        assert abs(d["variance"] - 71 / 240) < 1e-15

    def test_validation_errors(self):
        from fast_mlsirm import (
            flexilevel_administer,
            flexilevel_score_distribution,
        )

        with pytest.raises(ValueError, match="odd"):
            flexilevel_administer(np.zeros((1, 4)), n_persons=1, n_items=4)
        with pytest.raises(ValueError, match="0 and 1"):
            flexilevel_administer(
                np.full((1, 5), 2.0), n_persons=1, n_items=5
            )
        with pytest.raises(ValueError, match="real-valued"):
            flexilevel_administer(
                np.zeros((1, 5), dtype=complex), n_persons=1, n_items=5
            )
        with pytest.raises(ValueError, match="shape"):
            flexilevel_administer(np.zeros((2, 5)), n_persons=1, n_items=5)
        with pytest.raises(ValueError, match=r"p\[1\]"):
            flexilevel_score_distribution([0.5, 1.5, 0.5])
        with pytest.raises(ValueError, match="real-valued"):
            flexilevel_score_distribution(np.array([0.5, 0.5j, 0.5]))
        # Object-dtype complex bypasses np.iscomplexobj; the float64
        # coercion backstop must still raise the documented ValueError.
        with pytest.raises(ValueError, match="real-valued"):
            flexilevel_administer(
                np.array([0, 1 + 0j, 0, 1, 0], dtype=object),
                n_persons=1,
                n_items=5,
            )
        with pytest.raises(ValueError, match="real-valued"):
            flexilevel_score_distribution(
                np.array([0.5, 0.5 + 0j, 0.5], dtype=object)
            )

class TestStradaptive:
    def test_person_d_lower_step_pin(self):
        from fast_mlsirm import stradaptive_administer

        # Below-chance-side m7 anchor (spec Person D): p = 1/3 < 1/2 at hnc
        # with UNEQUAL adjacent gaps (D = [-2, 0, 3]); the derived lower
        # step gives 0 + (0 - (-2))(1/3 - 1/2) = -1/3, while an
        # always-upper mutant gives -1/2. Every assert reads crate output.
        r = stradaptive_administer(
            [0, 0, 1, 1, 1, 2, 2],
            [-2.0, -2.0, -1.0, 0.0, 1.0, 3.0, 3.0],
            [1, 1, 1, 0, 0, 0, 0],
            entry_stratum=1,
            chance=0.25,
            min_items=2,
            max_items=100,
        )
        assert r["administered"].tolist() == [2, 5, 3, 0, 4, 1]
        assert r["reason"] == "pool_exhausted"
        assert r["ceiling"] == -1
        assert r["hnc"] == 1
        assert abs(r["scores"][6] - (-1.0 / 3.0)) < 1e-12
        assert abs(r["scores"][6] - (-0.5)) > 0.1

    def test_person_e_boundary_pin(self):
        from fast_mlsirm import stradaptive_administer

        # p == chance exactly must terminate (<=, not <); chance = 1/2.
        r = stradaptive_administer(
            [0, 0, 0, 1, 1],
            [-1.0, -1.0, -1.0, 1.0, 1.0],
            [1, 0, 1, 0, 0],
            entry_stratum=0,
            chance=0.5,
            min_items=2,
            max_items=100,
        )
        assert r["administered"].tolist() == [0, 3, 1]
        assert r["responses_taken"].tolist() == [1, 0, 0]
        assert r["reason"] == "criterion"
        assert r["ceiling"] == 0
        assert r["hnc"] == -1
        assert r["scores"][0] == -1.0
        assert r["scores"][1] == -1.0
        assert np.isnan(r["scores"][2])
        assert np.isnan(r["scores"][8])

    def test_validation_errors(self):
        from fast_mlsirm import stradaptive_administer

        kw = dict(entry_stratum=0, chance=0.25, min_items=1, max_items=10)
        with pytest.raises(ValueError, match="chance"):
            stradaptive_administer(
                [0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], [1, 0, 1, 0],
                entry_stratum=0, chance=1.0, min_items=1, max_items=10,
            )
        with pytest.raises(ValueError, match="0 and 1"):
            stradaptive_administer(
                [0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], [1, 2, 1, 0], **kw
            )
        with pytest.raises(ValueError, match="real-valued"):
            stradaptive_administer(
                [0, 0, 1, 1],
                np.array([0.0, 1j, 1.0, 1.0]),
                [1, 0, 1, 0],
                **kw,
            )
        # Object-dtype complex bypasses np.iscomplexobj; the float64
        # coercion backstop must still raise the documented ValueError.
        with pytest.raises(ValueError, match="real-valued"):
            stradaptive_administer(
                [0, 0, 1, 1],
                np.array([0.0, 1 + 0j, 1.0, 1.0], dtype=object),
                [1, 0, 1, 0],
                **kw,
            )
        with pytest.raises(ValueError, match="non-negative integers"):
            stradaptive_administer(
                [0, -1, 1, 1], [0.0, 0.0, 1.0, 1.0], [1, 0, 1, 0], **kw
            )
        # 2**53 + 1 rounds to 2**53 under float64; the len-based bound must
        # reject it anyway (regression for the lossy > 2**53 guard).
        with pytest.raises(ValueError, match="below len"):
            stradaptive_administer(
                [0, 2**53 + 1, 1, 1], [0.0, 0.0, 1.0, 1.0], [1, 0, 1, 0],
                **kw,
            )
        with pytest.raises(ValueError, match="stratum"):
            stradaptive_administer(
                [0, 0, 2, 2], [0.0, 0.0, 1.0, 1.0], [1, 0, 1, 0], **kw
            )

class TestPyramidal:
    """Larkin & Weiss (1974) pyramidal adaptive testing wrapper.

    Anchors from files/pyramidal_oracle.py (exact Fractions, executed);
    every assert reads values returned by the crate through the wrapper.
    """

    def _pyr4(self):
        import numpy as np

        b = np.array([0, -1, 1, -2, 0, 2, -3, -1, 1, 3], dtype=np.float64)
        b_next = np.array([-4, -2, 0, 2, 4], dtype=np.float64)
        return b, b_next

    def test_anchor_a_exact(self):
        import numpy as np
        from fast_mlsirm import pyramidal_administer

        b, bn = self._pyr4()
        r = pyramidal_administer(b, 4, [1, 0, 1, 1], b_next=bn)
        assert r["path"].tolist() == [0, 2, 4, 8]
        assert r["positions"].tolist() == [0, 1, 1, 2]
        assert r["number_correct"] == 3.0
        assert r["mean_b_attempted"] == 0.5
        assert abs(r["mean_b_correct"] - 1.0 / 3.0) < 1e-15
        assert r["final_b"] == 1.0
        assert r["final_difficulty"] == 2.0
        assert r["all_item_score"] == 15.0
        # M5 unavailable without b_next
        r2 = pyramidal_administer(b, 4, [1, 0, 1, 1])
        assert np.isnan(r2["final_difficulty"])
        assert r2["all_item_score"] == 15.0

    def test_all_wrong_and_paper_range(self):
        import numpy as np
        from fast_mlsirm import pyramidal_administer

        b, bn = self._pyr4()
        r = pyramidal_administer(b, 4, np.zeros(4), b_next=bn)
        assert r["path"].tolist() == [0, 1, 3, 6]
        assert np.isnan(r["mean_b_correct"])
        assert r["final_difficulty"] == -4.0
        assert r["all_item_score"] == 0.0
        # Paper-printed 15-stage all-item range 0..240 (p. 16).
        n = 15
        big = np.zeros(n * (n + 1) // 2)
        assert pyramidal_administer(big, n, np.ones(n))["all_item_score"] == 240.0

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import pyramidal_administer

        b, bn = self._pyr4()
        with pytest.raises(ValueError, match="0 and 1"):
            pyramidal_administer(b, 4, [1, 0, 2, 1], b_next=bn)
        with pytest.raises(ValueError, match="real-valued"):
            pyramidal_administer(b.astype(np.complex128), 4, [1, 0, 1, 1])
        with pytest.raises(ValueError, match="real-valued"):
            pyramidal_administer(
                np.array([1j if i == 0 else 0.0 for i in range(10)], dtype=object),
                4,
                [1, 0, 1, 1],
            )
        with pytest.raises(ValueError, match="n\\(n\\+1\\)/2"):
            pyramidal_administer(b[:9], 4, [1, 0, 1, 1])
        with pytest.raises(ValueError, match="n_stages \\+ 1"):
            pyramidal_administer(b, 4, [1, 0, 1, 1], b_next=bn[:4])

class TestTwoStage:
    """Betz & Weiss (1973, 1974) two-stage adaptive testing wrappers.

    Anchors from files/twostage_oracle.py (exact Fractions through the
    p-computation, executed; Acklam inverse-CDF tolerance 1e-7); every
    assert reads values returned by the crate through the wrapper.
    """

    def _ts(self):
        import numpy as np

        a_meas = np.array([0.53, 0.55, 0.61, 0.68])
        b_meas = np.array([1.73, 0.35, -0.71, -1.60])
        return 10, 0.70, -0.23, a_meas, b_meas

    def test_anchor_pipeline(self):
        from fast_mlsirm import two_stage_route, two_stage_score

        m1, a1, b1, a_meas, b_meas = self._ts()
        theta1, assigned = two_stage_route(7, m1, a1, b1, b_meas, 0.2)
        assert abs(theta1 - 0.22519909137767882) < 1e-7
        assert assigned == 1
        r = two_stage_score(7, m1, a1, b1, 20, 30, assigned, a_meas, b_meas, 0.2)
        assert r["theta1"] == theta1
        assert r["assigned"] == 1
        assert abs(r["theta2"] - 0.7325970804507724) < 1e-7
        assert abs(r["composite"] - 0.605747583182499) < 1e-7

    def test_truncation_and_x9(self):
        from fast_mlsirm import two_stage_route

        m1, a1, b1, _a, b_meas = self._ts()
        # Perfect score truncates to x' = m - 1/2; chance-or-below to
        # x' = c*m + 1/2 (Betz & Weiss, 1974).
        t10, _ = two_stage_route(10, m1, a1, b1, b_meas, 0.2)
        t9, _ = two_stage_route(9, m1, a1, b1, b_meas, 0.2)
        assert abs(t9 - 1.4133562576800114) < 1e-7
        assert t10 > t9
        t2, _ = two_stage_route(2, m1, a1, b1, b_meas, 0.2)
        t0, _ = two_stage_route(0, m1, a1, b1, b_meas, 0.2)
        assert t2 == t0

    def test_error_contract(self):
        import numpy as np
        import pytest
        from fast_mlsirm import two_stage_route, two_stage_score

        m1, a1, b1, a_meas, b_meas = self._ts()
        with pytest.raises(ValueError, match="wrong test"):
            two_stage_score(7, m1, a1, b1, 20, 30, 0, a_meas, b_meas, 0.2)
        with pytest.raises(ValueError, match="must be an integer"):
            two_stage_route(7.5, m1, a1, b1, b_meas, 0.2)
        with pytest.raises(ValueError, match="out of range"):
            two_stage_route(-1, m1, a1, b1, b_meas, 0.2)
        with pytest.raises(ValueError, match="exceeds"):
            two_stage_route(11, m1, a1, b1, b_meas, 0.2)
        with pytest.raises(ValueError, match="real-valued"):
            two_stage_route(7, m1, a1, b1, b_meas.astype(np.complex128), 0.2)
        with pytest.raises(ValueError, match="real-valued"):
            two_stage_score(
                7, m1, a1, b1, 20, 30, 1,
                np.array([1j, 0, 0, 0], dtype=object), b_meas, 0.2,
            )
        with pytest.raises(ValueError, match=r"m\*\(1-c\)"):
            two_stage_route(1, 1, a1, b1, b_meas, 0.2)
        # Object-dtype arrays are rejected outright, not coerced.
        with pytest.raises(ValueError, match="numeric"):
            two_stage_route(7, m1, a1, b1, b_meas.astype(object), 0.2)
        with pytest.raises(ValueError, match="numeric"):
            two_stage_score(
                7, m1, a1, b1, 20, 30, 1, a_meas.astype(object), b_meas, 0.2
            )
        # Huge m collapses the f64 truncation endpoints -> ValueError,
        # never a silent NaN (core-side runtime guard).
        with pytest.raises(ValueError, match="degenerate"):
            two_stage_route(2**53, 2**53, a1, b1, b_meas, 0.2)

class TestHansonBrennan:
    """Hanson-Brennan compound binomial classification (Hanson, 1991, ACT
    RR 91-5, read in full; cross-checked against CRAN betafunctions 1.9.0
    HB.CA): every assert reads crate outputs returned through the wrapper
    against literals from an independent exact-Fraction stdlib oracle
    (never this crate). Params fixture pins are exact rationals (1e-12);
    the data fixture takes the genuine 4P path with negative Lord's k and
    both beta shapes < 1 (tolerance 1e-7 per spec)."""

    def test_params_path_matches_exact_oracle(self):
        from fast_mlsirm import hanson_brennan_from_params
        from fast_mlsirm.fitstats import _core_module

        core = _core_module()
        if core is None or not hasattr(core, "hanson_brennan_from_params"):
            pytest.skip("compiled core built without hanson_brennan")
        r = hanson_brennan_from_params(8, 0.5, 0.0, 1.0, 2.0, 3.0, 5)
        tol = 1e-12
        assert abs(r.consistency - 0.7882233865206) < tol
        assert abs(r.kappa - 0.459362303476315) < tol
        assert abs(r.p_tp - 0.1271510992216) < tol
        assert abs(r.accuracy - 0.835374853002183) < tol
        assert abs(r.sensitivity - 0.838664899213647) < tol
        assert abs(r.specificity - 0.834786905175617) < tol
        assert r.p_ji == r.p_ij
        assert not r.used_two_parameter
        import numpy as np

        assert np.isnan(r.true_score_moments).all()

    def test_data_path_four_parameter_matches_oracle(self):
        from fast_mlsirm import hanson_brennan
        from fast_mlsirm.fitstats import _core_module

        core = _core_module()
        if core is None or not hasattr(core, "hanson_brennan"):
            pytest.skip("compiled core built without hanson_brennan")
        scores = [7, 1, 11, 2, 9, 10, 6, 6, 12, 6, 10, 8, 10, 8, 2, 2, 5,
                  8, 12, 11, 2, 1]
        r = hanson_brennan(scores, 12, 0.85, 7)
        assert abs(r.lords_k - -0.428757070304408) < 1e-12
        assert abs(r.true_score_moments[0] - 0.564393939393939) < 1e-12
        assert abs(r.true_score_moments[3] - 0.245696195187782) < 1e-12
        assert not r.used_two_parameter
        assert abs(r.lower - 0.132649605850816) < 1e-11
        assert abs(r.upper - 0.888674369964715) < 1e-11
        assert abs(r.alpha - 0.45726578470813) < 1e-11
        assert abs(r.beta - 0.343449430670119) < 1e-11
        tol = 1e-7
        assert abs(r.accuracy - 0.910755004973576) < tol
        assert abs(r.consistency - 0.877865550806179) < tol
        assert abs(r.kappa - 0.752703324849248) < tol
        assert abs(r.sensitivity - 0.941718760516029) < tol
        assert abs(r.specificity - 0.876176500427409) < tol

    def test_rejects_degenerate_inputs(self):
        import numpy as np

        from fast_mlsirm import hanson_brennan, hanson_brennan_from_params
        from fast_mlsirm.fitstats import _core_module

        core = _core_module()
        if core is None or not hasattr(core, "hanson_brennan"):
            pytest.skip("compiled core built without hanson_brennan")
        good = [7, 1, 11, 2, 9, 10, 6, 6, 12, 6, 10, 8, 10, 8, 2, 2, 5,
                8, 12, 11, 2, 1]
        with pytest.raises(ValueError):
            hanson_brennan(good[:5], 12, 0.85, 7)  # too few scores
        with pytest.raises(ValueError):
            hanson_brennan(good, 12, 1.0, 7)  # reliability >= 1
        with pytest.raises(ValueError):
            hanson_brennan(good, 12, 0.85, 0)  # cut below 1
        with pytest.raises(ValueError):
            hanson_brennan(good, 12, 0.85, 13)  # cut above n_items
        with pytest.raises(ValueError):
            hanson_brennan([6.5] + good[1:], 12, 0.85, 7)  # non-integer
        with pytest.raises(ValueError):
            hanson_brennan(np.array(good) * 1j, 12, 0.85, 7)  # complex
        with pytest.raises(ValueError):
            hanson_brennan(
                np.array(["x"] * 22, dtype=object), 12, 0.85, 7
            )  # non-numeric objects
        with pytest.raises(ValueError):
            hanson_brennan_from_params(8, 0.0, 0.5, 0.4, 2.0, 2.0, 4)
        with pytest.raises(ValueError):
            hanson_brennan_from_params(8, 0.0, 0.0, 1.0, -2.0, 2.0, 4)


class TestPhiLambda:
    def test_fixture_a_exact(self):
        # Oracle pins (phi_lambda_oracle.py, exact Fraction); asserts read
        # wrapper outputs that come straight from the Rust core.
        import numpy as np

        from fast_mlsirm import phi_lambda

        x = np.array(
            [[1, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0]],
            dtype=float,
        )
        r = phi_lambda(x, 0.25, n_i_prime=[4, 8])
        assert abs(r.grand_mean - 0.6) < 1e-12
        assert abs(r.var_xbar - 11.0 / 600.0) < 1e-12
        assert abs(r.signal - 5.0 / 48.0) < 1e-12
        assert abs(r.phi[0] - 0.75) < 1e-12
        assert abs(r.phi[1] - 6.0 / 7.0) < 1e-12

    def test_negative_signal_at_mean_and_fixture_d(self):
        import numpy as np

        from fast_mlsirm import phi_lambda

        x = np.array(
            [[1, 1, 1, 0], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 1, 0]],
            dtype=float,
        )
        r = phi_lambda(x, 0.6, n_i_prime=[4])
        assert abs(r.signal - (-11.0 / 600.0)) < 1e-12
        assert abs(r.phi[0] - 48.0 / 113.0) < 1e-12
        d = np.array([[5, 3, 1], [4, 3, 2], [6, 4, 2], [3, 1, 0]], dtype=float)
        rd = phi_lambda(d, 2.0, n_i_prime=[3, 6])
        assert abs(rd.phi[0] - 12.0 / 29.0) < 1e-12
        assert abs(rd.phi[1] - 24.0 / 41.0) < 1e-12

    def test_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import phi_lambda

        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            phi_lambda(x.astype(complex), 0.5, n_i_prime=[2])
        with pytest.raises(ValueError):
            phi_lambda(np.array([1.0, 0.0]), 0.5, n_i_prime=[2])
        with pytest.raises(ValueError):
            phi_lambda(x, float("nan"), n_i_prime=[2])
        with pytest.raises(ValueError):
            phi_lambda(x, 0.5, n_i_prime=[0])
        with pytest.raises(ValueError):
            phi_lambda(np.array([[1.0, np.nan], [0.0, 1.0]]), 0.5, n_i_prime=[2])

class TestLivingston:
    def test_fixture_a_exact(self):
        # Oracle pins (livingston_oracle.py, exact Fraction): k2 = 5/6,
        # SB(2) = 10/11; asserts read wrapper outputs from the Rust core.
        import numpy as np

        from fast_mlsirm import livingston_k2

        r = livingston_k2(np.array([2.0, 4.0, 4.0, 6.0]), 2.0, 0.5, [1.0, 2.0])
        assert abs(r.mean - 4.0) < 1e-15
        assert abs(r.var - 2.0) < 1e-15
        assert abs(r.msd - 6.0) < 1e-15
        assert abs(r.k2[0] - 5.0 / 6.0) < 1e-15
        assert abs(r.k2[1] - 10.0 / 11.0) < 1e-15
        # Equality anchor: cut at the mean gives k2 == rho2.
        at_mean = livingston_k2(np.array([2.0, 4.0, 4.0, 6.0]), 4.0, 0.5)
        assert abs(at_mean.k2[0] - 0.5) < 1e-15

    def test_correlation_sign_flip_and_asymmetric(self):
        import math

        import numpy as np

        from fast_mlsirm import livingston_correlation

        # Norm rho = -1 but k = +5/7 (oracle fixture B).
        k = livingston_correlation(
            np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), 0.0, 0.0
        )
        assert abs(k - 5.0 / 7.0) < 1e-15
        # Asymmetric offsets (oracle fixture E): k = 22/(7 sqrt(10)).
        ke = livingston_correlation(
            np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), 0.0, 1.0
        )
        assert abs(ke - 22.0 / (7.0 * math.sqrt(10.0))) < 1e-15

    def test_error_contract_and_nan(self):
        import math

        import numpy as np
        import pytest

        from fast_mlsirm import livingston_correlation, livingston_k2

        x = np.array([2.0, 4.0, 4.0, 6.0])
        with pytest.raises(ValueError):
            livingston_k2(x.astype(complex), 2.0, 0.5)
        with pytest.raises(ValueError):
            livingston_k2(np.array(["a", "b"], dtype=object), 2.0, 0.5)
        with pytest.raises(ValueError):
            livingston_k2(x, 2.0, 1.5)
        with pytest.raises(ValueError):
            livingston_k2(x, 2.0, 0.5, [])
        with pytest.raises(ValueError):
            livingston_k2(x, 2.0, 0.5, [0.0])
        with pytest.raises(ValueError):
            livingston_correlation(x.astype(complex), x, 0.0, 0.0)
        with pytest.raises(ValueError):
            livingston_correlation(x, x[:2], 0.0, 0.0)
        # NaN only in the exact degenerate case (var 0 AND mean == cut).
        const = np.array([3.0, 3.0, 3.0])
        assert math.isnan(livingston_k2(const, 3.0, 0.5).k2[0])
        assert livingston_k2(const, 1.0, 0.0).k2[0] == 1.0
        assert math.isnan(livingston_correlation(const, x[:3], 3.0, 0.0))

    def test_review_regressions(self):
        """Regression pins for impl-review findings: element-wise degenerate
        detection at a non-representable decimal cut, overflow limit for
        k^2, overflow error for the correlation, and non-None __doc__.

        Every assert reads crate values via the wrappers; killed mutants:
        removing the element-wise all-equal check (first assert), removing
        the off2-overflow limit branch (second), removing the overflow Err
        (third)."""
        import math

        import numpy as np
        import pytest

        import fast_mlsirm as fm
        from fast_mlsirm import livingston_correlation, livingston_k2

        const = np.array([0.1, 0.1, 0.1])
        assert math.isnan(livingston_k2(const, 0.1, 0.5).k2[0])
        assert math.isnan(
            livingston_correlation(const, np.array([1.0, 2.0, 3.0]), 0.1, 0.0)
        )
        x = np.array([1.0, 2.0, 3.0])
        res = livingston_k2(x, 1e308, 0.5, [1.0, 2.0])
        assert res.k2[0] == 1.0
        assert res.k2[1] == 1.0
        with pytest.raises((ValueError, RuntimeError)):
            livingston_correlation(x, x, 1e308, 0.0)
        assert fm.livingston_k2.__doc__ is not None
        assert "References" in fm.livingston_k2.__doc__
        assert fm.livingston_correlation.__doc__ is not None
        assert "References" in fm.livingston_correlation.__doc__


class TestSubkoviak:
    def test_table1_alpha_supplied_exact(self):
        # Oracle pins (subkoviak_oracle.py, exact Fraction); asserts read
        # wrapper outputs that come straight from the Rust core.
        import numpy as np

        from fast_mlsirm import subkoviak_agreement

        x = np.array([0, 4, 2, 0, 2, 2, 1, 3, 4, 5], dtype=float)
        r = subkoviak_agreement(x, 5, [4], alpha=0.58)
        assert abs(r.alpha - 0.58) < 1e-15
        assert abs(r.p_hat[0] - 0.1932) < 1e-12
        assert abs(r.per_person[0] - 0.988290295814609) < 1e-12
        assert abs(r.agreement - 0.754404497506925) < 1e-12
        assert abs(r.chance_agreement - 0.659131077528193) < 1e-12
        assert abs(r.kappa - 0.279501631559306) < 1e-12

    def test_derived_kr21_and_multi_cut(self):
        import numpy as np

        from fast_mlsirm import subkoviak_agreement

        x = np.array([0, 4, 2, 0, 2, 2, 1, 3, 4, 5], dtype=float)
        r = subkoviak_agreement(x, 5, [4])
        assert abs(r.alpha - 19.0 / 29.0) < 1e-12
        assert abs(r.kappa - 0.343090844315065) < 1e-12
        c = np.array([1, 2, 3, 4, 5, 6, 0, 3], dtype=float)
        rc = subkoviak_agreement(c, 6, [2, 5], alpha=1.0)
        assert abs(rc.per_person[0] - 0.611776411438135) < 1e-12
        assert abs(rc.kappa - 0.53024146176074) < 1e-12

    def test_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import subkoviak_agreement

        x = np.array([0, 4, 2, 0, 2, 2, 1, 3, 4, 5], dtype=float)
        with pytest.raises(ValueError):
            subkoviak_agreement(x.astype(complex), 5, [4], alpha=0.5)
        with pytest.raises(ValueError):
            subkoviak_agreement(np.array(["a", "b"], dtype=object), 5, [4])
        with pytest.raises(ValueError):
            subkoviak_agreement(x, 5, [], alpha=0.5)
        with pytest.raises(ValueError):
            subkoviak_agreement(x, 5, [6], alpha=0.5)
        with pytest.raises(ValueError):
            subkoviak_agreement(x, 5, [4], alpha=1.5)
        with pytest.raises(ValueError):
            subkoviak_agreement(np.array([3.0, 3.0, 3.0]), 5, [4])

class TestWoodruffSawyer:
    """Woodruff & Sawyer (1988, ERIC ED292877) pass-fail reliability.

    Every assert reads values returned by the crate through the Python
    wrapper (WoodruffSawyerResult fields); expected values from the
    exact-Fraction / mpmath session oracle. Rust-side tests carry the
    mutation-kill provenance (MU1-MU6 all EXECUTED and killed).
    """

    def test_sb_fixture_a_exact(self):
        from fast_mlsirm import woodruff_sawyer_sb

        r = woodruff_sawyer_sb([2, 1, 3, 10])
        assert abs(r.pass_rate - 0.75) < 1e-15
        assert abs(r.phi_half - 1 / 3) < 1e-15
        assert abs(r.theta_half - 0.75) < 1e-15
        assert abs(r.phi - 0.5) < 1e-15
        assert abs(r.theta - 13 / 16) < 1e-15
        assert abs(r.pi00 - 5 / 32) < 1e-15
        assert abs(r.pi01 - 3 / 32) < 1e-15
        assert abs(r.pi11 - 21 / 32) < 1e-15

    def test_normal_orthant_and_pins(self):
        import math

        from fast_mlsirm import woodruff_sawyer_normal

        r = woodruff_sawyer_normal(0.0, 1.0, 0.0, 1 / 3)
        assert abs(r.pi00 - 1 / 3) < 1e-6
        assert abs(r.phi - 1 / 3) < 1e-6
        assert math.isnan(r.phi_half) and math.isnan(r.theta_half)
        n = woodruff_sawyer_normal(100.0, 15.0, 85.0, 0.6)
        assert abs(n.theta - 0.86360339357711063) < 1e-6
        assert abs(n.phi - 0.48908915212993364) < 1e-6

    def test_validation(self):
        import numpy as np
        import pytest

        from fast_mlsirm import woodruff_sawyer_normal, woodruff_sawyer_sb

        with pytest.raises(ValueError):
            woodruff_sawyer_sb([1, 2, 3])
        with pytest.raises(ValueError):
            woodruff_sawyer_sb([1, -1, 2, 3])
        with pytest.raises(ValueError):
            woodruff_sawyer_sb(np.array([1 + 2j, 0, 0, 1]))
        with pytest.raises(ValueError):
            woodruff_sawyer_sb(np.array([object(), 1, 1, 1], dtype=object))
        with pytest.raises(ValueError):
            woodruff_sawyer_normal(0.0, 0.0, 0.0, 0.5)
        with pytest.raises(ValueError):
            woodruff_sawyer_normal(0.0, 1.0, 0.0, -0.5)
        with pytest.raises(ValueError):
            woodruff_sawyer_normal(0.0, 1.0, 0.0, 1.0)

class TestCircleArc:
    def test_paper_method1(self):
        from fast_mlsirm import circle_arc_equate

        r = circle_arc_equate(
            [5.0, 12.0, 20.0, 10.0], (5.0, 5.0), (12.0, 14.0), (20.0, 20.0),
            method="arc1",
        )
        # crate-returned circle: paper worked example center (40, -15), r^2=1625
        assert r.xc == 40.0 and r.yc == -15.0 and r.r2 == 1625.0
        assert not r.collinear and r.middle == (12.0, 14.0)
        assert r.equated[0] == 5.0 and r.equated[1] == 14.0 and r.equated[2] == 20.0
        assert abs(r.equated[3] - 11.925824035672519) < 1e-12

    def test_paper_method2_and_anchor(self):
        from fast_mlsirm import circle_arc_equate, circle_arc_middle_anchor

        r = circle_arc_equate(
            [10.0], (5.0, 5.0), (12.0, 14.0), (20.0, 20.0), method="arc2",
        )
        assert r.xc == 12.5 and r.yc == -13.0 and r.r2 == 225.25
        assert abs(r.equated[0] - 11.798648586948742) < 1e-12
        x2, y2 = circle_arc_middle_anchor(73.62, 30.60, 77.47, 10.83, 30.46, 5.09)
        assert x2 == 73.62  # Table 1 new-form test mean m_XA
        assert abs(y2 - 77.76787819253438) < 1e-12  # paper Table 1 pin

    def test_errors(self):
        import numpy as np
        import pytest

        from fast_mlsirm import circle_arc_equate, circle_arc_middle_anchor

        with pytest.raises(ValueError):
            circle_arc_equate([1.0], (0.0, 0.0), (4.0, 2.0), (10.0, 10.0), method="nope")
        with pytest.raises(ValueError):
            circle_arc_equate([-1.0], (0.0, 0.0), (4.0, 2.0), (10.0, 10.0))
        with pytest.raises(ValueError):
            circle_arc_equate(np.array([1 + 2j]), (0.0, 0.0), (4.0, 2.0), (10.0, 10.0))
        with pytest.raises(ValueError):
            circle_arc_equate(np.array(["a"], dtype=object), (0.0, 0.0), (4.0, 2.0), (10.0, 10.0))
        with pytest.raises(ValueError):
            circle_arc_equate([1.0], (0.0, 0.0), 4.0, (10.0, 10.0))
        with pytest.raises(ValueError):
            # mixed-branch triple: no single-branch arc through all three
            circle_arc_equate([1.0], (0.0, 0.0), (1.0, 3.0), (2.0, 1.0))
        with pytest.raises(ValueError):
            circle_arc_middle_anchor(1.0, 1.0, 1.0, 0.0, 1.0, 1.0)

class TestNominalWeights:
    """Nominal weights mean equating (Babcock et al., 2012; Albano, 2016).

    Oracle: exact-Fraction hand computation of Albano (2016) eqs. 37-40/42/10
    plus an executed cross-check against the method authors' R package
    equate 2.0.8. Every assert reads crate outputs through the wrapper.
    """

    XT = [10.0, 12.0, 14.0, 16.0]
    XA = [4.0, 5.0, 5.0, 6.0]
    YT = [8.0, 9.0, 11.0, 12.0]
    YA = [3.0, 4.0, 4.0, 5.0]

    def test_fixture_a_exact(self):
        import numpy as np

        from fast_mlsirm import nominal_weights_mean_equate

        r = nominal_weights_mean_equate(
            self.XT, self.XA, self.YT, self.YA, 20, 20, 8, w1=1.0
        )
        assert r.intercept == -0.5
        assert r.slope == 1.0
        assert r.x_scores.shape == (21,)
        assert r.y_equivalents[0] == -0.5
        assert r.y_equivalents[20] == 19.5
        # equal gammas: intercept is w1-invariant
        half = nominal_weights_mean_equate(
            self.XT, self.XA, self.YT, self.YA, 20, 20, 8, w1=0.5
        )
        assert half.intercept == -0.5
        assert np.isfinite(r.moments["sigma_x"]) and r.moments["sigma_x"] > 0

    def test_fixture_b_exact(self):
        from fast_mlsirm import nominal_weights_mean_equate

        r = nominal_weights_mean_equate(
            self.XT, self.XA, self.YT, self.YA, 20, 30, 10, w1=0.25
        )
        assert r.intercept == -0.75
        assert r.moments["mu_x"] == 11.5
        assert r.moments["mu_y"] == 10.75
        assert r.moments["sigma_x"] == (23.0 / 4.0) ** 0.5
        assert r.moments["sigma_y"] == (67.0 / 16.0) ** 0.5

    def test_errors(self):
        import numpy as np
        import pytest

        from fast_mlsirm import nominal_weights_mean_equate

        with pytest.raises(ValueError):
            nominal_weights_mean_equate(
                self.XT, self.XA, self.YT, self.YA, 0, 20, 8
            )
        with pytest.raises(ValueError):
            nominal_weights_mean_equate(
                self.XT, self.XA, self.YT, self.YA, 20, 20, 8, w1=1.5
            )
        with pytest.raises(ValueError):
            nominal_weights_mean_equate(
                self.XT, self.XA[:3], self.YT, self.YA, 20, 20, 8
            )
        with pytest.raises(ValueError):
            nominal_weights_mean_equate(
                np.array(self.XT) * 1j, self.XA, self.YT, self.YA, 20, 20, 8
            )
        with pytest.raises(ValueError):
            nominal_weights_mean_equate(
                np.array(["a"], dtype=object), ["b"], self.YT, self.YA, 20, 20, 8
            )

class TestCompositeLinking:
    def test_fixture_a_exact(self):
        # Oracle pins (files/composite_oracle.py): identity + 3x-2 over
        # [0,1,2,10], wc=(1/2,1/2), slopes (1,3), p=1 -> W=(2/3,1/3),
        # composite (5/3)x - 2/3 = [-2/3,1,8/3,16]. Asserts read crate dict.
        import numpy as np
        from fast_mlsirm import composite_linking

        grid = np.array([0.0, 1.0, 2.0, 10.0])
        r = composite_linking(
            [grid, 3.0 * grid - 2.0], [0.5, 0.5], slopes=[1.0, 3.0], p=1.0
        )
        assert r["symmetric"] is True
        np.testing.assert_allclose(
            r["adjusted_weights"], [2.0 / 3.0, 1.0 / 3.0], rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            r["composite"], [-2.0 / 3.0, 1.0, 8.0 / 3.0, 16.0], rtol=0, atol=1e-14
        )

    def test_raw_weights_nonsymmetric(self):
        # Non-symmetric path: W=(1/2,1/2), composite [-1,1,3,19]; scale
        # invariance of the normalization checked via weights (2,2).
        import numpy as np
        from fast_mlsirm import composite_linking

        grid = np.array([0.0, 1.0, 2.0, 10.0])
        tabs = [grid, 3.0 * grid - 2.0]
        r = composite_linking(tabs, [0.5, 0.5])
        assert r["symmetric"] is False
        np.testing.assert_allclose(
            r["composite"], [-1.0, 1.0, 3.0, 19.0], rtol=0, atol=1e-14
        )
        r2 = composite_linking(tabs, [2.0, 2.0])
        np.testing.assert_allclose(
            r2["composite"], r["composite"], rtol=0, atol=1e-15
        )

    def test_errors(self):
        import numpy as np
        import pytest
        from fast_mlsirm import composite_linking

        t = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            composite_linking([], [])
        with pytest.raises(ValueError):
            composite_linking([t], [-1.0])
        with pytest.raises(ValueError):
            composite_linking([t], [1.0], slopes=[0.0])
        with pytest.raises(ValueError):
            composite_linking([t], [1.0], slopes=[1.0], p=0.5)
        with pytest.raises(ValueError):
            composite_linking([t + 1j], [1.0])
        with pytest.raises(ValueError):
            composite_linking([np.array([1.0, "a"], dtype=object)], [1.0])

class TestThurstoneCaseV:
    """Thurstone (1927) Case V scaling via psych's thurstone() algorithm.

    Pins from files/thurstone_oracle.py (mpmath 50-digit, EXECUTED).
    All asserts read the ThurstoneResult returned by the crate.
    """

    def test_fixture_a_scale_gof_model(self):
        # Kills sign/colmean/min-shift mutants: scale + GF + off-diagonal
        # model cells pinned against the 50-digit oracle (asymmetric,
        # nonzero-residual fixture).
        import numpy as np
        from fast_mlsirm import thurstone_case_v

        choice = np.array(
            [[0.5, 0.7, 0.9], [0.3, 0.5, 0.8], [0.1, 0.2, 0.5]]
        )
        r = thurstone_case_v(choice)
        np.testing.assert_allclose(
            r.scale,
            [0.0, 0.49624378579592261, 1.3097082924567186],
            rtol=0,
            atol=1e-6,
        )
        assert abs(r.gof - 0.99986967677023893) < 1e-6
        assert abs(r.model[0, 2] - 0.9048527313906016) < 1e-6
        assert abs(r.model[2, 0] - 0.095147268609398397) < 1e-6
        assert abs(r.residual[0, 1] - (-0.00986121100602988)) < 1e-6
        assert r.model.shape == (3, 3) and r.residual.shape == (3, 3)

    def test_fixture_b_consistent_roundtrip(self):
        # Exactly Case-V-consistent 2x2: GF must be 1, model reproduces
        # the input (round-trip anchor; variance mutations are covered by
        # fixture A/C pins, not this zero-residual fixture).
        import numpy as np
        from fast_mlsirm import thurstone_case_v

        choice = np.array([[0.5, 0.75], [0.25, 0.5]])
        r = thurstone_case_v(choice)
        assert abs(r.gof - 1.0) < 1e-9
        np.testing.assert_allclose(
            r.scale, [0.0, 0.67448975019608174], rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(r.model, choice, rtol=0, atol=1e-6)

    def test_errors(self):
        import numpy as np
        import pytest
        from fast_mlsirm import thurstone_case_v

        good = np.array([[0.5, 0.75], [0.25, 0.5]])
        with pytest.raises(ValueError):
            thurstone_case_v(good[:1])  # non-square
        with pytest.raises(ValueError):
            thurstone_case_v(np.array([[0.5]]))  # n < 2
        bad = good.copy()
        bad[0, 1] = 1.0  # boundary excluded: strictly in (0, 1)
        with pytest.raises(ValueError):
            thurstone_case_v(bad)
        with pytest.raises(ValueError):
            thurstone_case_v(good + 0j)  # complex rejected
        with pytest.raises(ValueError):
            thurstone_case_v(
                np.array([[0.5, "a"], [0.25, 0.5]], dtype=object)
            )

class TestBradleyTerry:
    """Bradley-Terry MM fit (choix 0.4.1 opt.mm pairwise path, source READ).

    Pins from files/bt_oracle.py (mpmath 50-digit, EXECUTED; choix
    cross-checked to <= 1.4e-12). All asserts read the BradleyTerryResult
    returned by the crate.
    """

    def test_fixture_a_params_weights_iterations(self):
        # Asymmetric 3x3; kills winner-accumulation / denominator /
        # centering mutants; iterations == 18 pins the tol*n convergence
        # semantics (reviewer-executed MU5 evidence).
        import numpy as np
        from fast_mlsirm import bradley_terry_mm

        wins = np.array([[0.0, 3.0, 1.0], [2.0, 0.0, 4.0], [5.0, 1.0, 0.0]])
        r = bradley_terry_mm(wins)
        np.testing.assert_allclose(
            r.params,
            [-0.378869072353494149, 0.274223212389322699,
             0.104645859964171450],
            rtol=0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            r.weights,
            [0.660321972720804678, 1.268791102424604100,
             1.070886924854591222],
            rtol=0,
            atol=1e-7,
        )
        assert r.iterations == 18
        assert abs(float(np.mean(r.params))) < 1e-12
        assert abs(float(np.sum(r.weights)) - 3.0) < 1e-9

    def test_fixture_b_closed_form_and_alpha(self):
        # 2x2 closed form: params = +/- ln(3)/2 (verified symbolically in
        # the oracle); fixture D pins the alpha=0.5 MAP path, which kills
        # the weight-normalization mutant (alpha=0 is scale-invariant).
        import numpy as np
        from fast_mlsirm import bradley_terry_mm

        r = bradley_terry_mm(np.array([[0.0, 3.0], [1.0, 0.0]]))
        half_ln3 = 0.549306144334054846
        np.testing.assert_allclose(
            r.params, [half_ln3, -half_ln3], rtol=0, atol=1e-7
        )
        np.testing.assert_allclose(r.weights, [1.5, 0.5], rtol=0, atol=1e-7)

        wins = np.array([[0.0, 3.0, 1.0], [2.0, 0.0, 4.0], [5.0, 1.0, 0.0]])
        d = bradley_terry_mm(wins, alpha=0.5)
        np.testing.assert_allclose(
            d.params,
            [-0.337946615223381393, 0.240502957855231605,
             0.097443657368149788],
            rtol=0,
            atol=1e-7,
        )

    def test_errors(self):
        import numpy as np
        import pytest
        from fast_mlsirm import bradley_terry_mm

        good = np.array([[0.0, 3.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            bradley_terry_mm(good[:1])  # non-square
        with pytest.raises(ValueError):
            bradley_terry_mm(np.array([[0.0]]))  # n < 2
        with pytest.raises(ValueError):
            bradley_terry_mm(np.zeros((2, 2)))  # all-zero rejected
        with pytest.raises(ValueError):
            bradley_terry_mm(np.zeros((2, 2)), alpha=0.5)  # even with alpha
        bad = good.copy()
        bad[0, 0] = 1.0
        with pytest.raises(ValueError):
            bradley_terry_mm(bad)  # nonzero diagonal
        with pytest.raises(ValueError):
            bradley_terry_mm(-good)  # negative counts
        with pytest.raises(ValueError):
            bradley_terry_mm(good + 0j)  # complex rejected
        with pytest.raises(ValueError):
            bradley_terry_mm(
                np.array([[0.0, "a"], [1.0, 0.0]], dtype=object)
            )
        with pytest.raises(ValueError):
            bradley_terry_mm(good, tol=0.0)
        with pytest.raises(ValueError):
            bradley_terry_mm(good, max_iter=0)
        # zero-wins item at alpha=0: no finite log-worth
        winless = np.array(
            [[0.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        with pytest.raises(ValueError):
            bradley_terry_mm(winless)


class TestLsr:
    """Luce Spectral Ranking (choix 0.4.1 lsr.py; Maystre & Grossglauser,
    2015). Pins from the EXECUTED exact-Fraction/mpmath oracle
    (session files lsr_oracle.py), cross-checked against pip choix 0.4.1.
    atol 1e-7 on I-LSR pins: converged iterate at tol=1e-8 sits ~5e-9 from
    the fixed point; mutant deviations are >= 3e-2."""

    def test_lsr_one_shot_exact(self):
        import numpy as np

        from fast_mlsirm import lsr_pairwise

        a = np.array([[0, 3, 1], [2, 0, 4], [5, 1, 0]], dtype=float)
        r = lsr_pairwise(a)
        assert r.iterations == 1
        np.testing.assert_allclose(
            r.weights, [12 / 17, 45 / 34, 33 / 34], atol=1e-13
        )
        np.testing.assert_allclose(
            r.params,
            [-0.31568746351363625, 0.31292119590873789, 0.00276626760489836],
            atol=1e-13,
        )
        # n = 2 closed form: +/- ln(3)/2.
        b = np.array([[0, 3], [1, 0]], dtype=float)
        rb = lsr_pairwise(b)
        np.testing.assert_allclose(
            rb.params, [np.log(3) / 2, -np.log(3) / 2], atol=1e-14
        )

    def test_ilsr_matches_bradley_terry(self):
        import numpy as np

        from fast_mlsirm import bradley_terry_mm, ilsr_pairwise

        a = np.array([[0, 3, 1], [2, 0, 4], [5, 1, 0]], dtype=float)
        r = ilsr_pairwise(a)
        np.testing.assert_allclose(
            r.params,
            [-0.37886907235349415, 0.27422321238932270, 0.10464585996417145],
            atol=1e-7,
        )
        assert r.iterations == 15
        bt = bradley_terry_mm(a)
        np.testing.assert_allclose(r.params, bt.params, atol=1e-6)
        assert abs(float(np.sum(r.weights)) - 3.0) < 1e-9

    def test_lsr_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import ilsr_pairwise, lsr_pairwise

        a = np.array([[0, 3, 1], [2, 0, 4], [5, 1, 0]], dtype=float)
        with pytest.raises(ValueError):
            lsr_pairwise(np.ones((2, 3)))
        with pytest.raises(ValueError):
            lsr_pairwise(a + 1j * a)
        with pytest.raises(ValueError):
            lsr_pairwise(np.array([["x"] * 3] * 3, dtype=object))
        with pytest.raises(ValueError):
            lsr_pairwise(-a)
        with pytest.raises(ValueError):
            lsr_pairwise(np.zeros((3, 3)))
        with pytest.raises(ValueError):
            lsr_pairwise(a, alpha=-1.0)
        # Disconnected comparison graph at alpha = 0.
        d = np.array(
            [[0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 3], [0, 0, 1, 0]],
            dtype=float,
        )
        with pytest.raises(ValueError):
            lsr_pairwise(d)
        with pytest.raises(ValueError):
            ilsr_pairwise(d)
        # ... but estimable with alpha > 0.
        rd = lsr_pairwise(d, alpha=0.5)
        assert np.all(np.isfinite(rd.params))
        # Overflow must raise, never return NaN. n = 4 at 1.7e308 makes
        # each generator row sum overflow to inf (an n = 3 all-1e308
        # matrix stays finite and is now correctly accepted).
        huge = np.full((4, 4), 1.7e308)
        np.fill_diagonal(huge, 0.0)
        with pytest.raises(ValueError):
            lsr_pairwise(huge)
        with pytest.raises(ValueError):
            lsr_pairwise(a, alpha=1e308)
        # Scale invariance: globally rescaled huge counts must match.
        base = lsr_pairwise(a)
        big = lsr_pairwise(a * 1e20)
        assert np.allclose(big.params, base.params, atol=1e-12)
        with pytest.raises(ValueError):
            ilsr_pairwise(a, max_iter=0)
        with pytest.raises(ValueError):
            ilsr_pairwise(a, tol=0.0)

class TestRankCentrality:
    """Rank Centrality (choix 0.4.1 rank_centrality; Negahban, Oh, & Shah,
    2017, as ported by choix -- continuous-time win-ratio chain). Pins from
    the EXECUTED exact-Fraction/mpmath oracle (session files
    rank_centrality_oracle.py), cross-checked against pip choix 0.4.1."""

    def test_rank_centrality_exact(self):
        import numpy as np

        from fast_mlsirm import lsr_pairwise, rank_centrality

        a = np.array([[0, 3, 1], [2, 0, 4], [5, 1, 0]], dtype=float)
        r = rank_centrality(a)
        assert r.iterations == 1
        np.testing.assert_allclose(
            r.weights, [138 / 181, 237 / 181, 168 / 181], atol=1e-13
        )
        np.testing.assert_allclose(
            r.params,
            [-0.2458389167413269, 0.2949675392365996, -0.0491286224952727],
            atol=1e-13,
        )
        # RC is the ratio chain, NOT the LSR rate chain (kills a
        # dispatch-to-lsr mutant): weights differ on the same input.
        rl = lsr_pairwise(a)
        assert np.max(np.abs(r.weights - rl.weights)) > 1e-2
        # alpha regularizes the ratio numerator and denominator.
        ra = rank_centrality(a, alpha=0.5)
        np.testing.assert_allclose(
            ra.weights, [207 / 265, 333 / 265, 255 / 265], atol=1e-13
        )
        np.testing.assert_allclose(
            ra.params,
            [-0.2279894828693773, 0.2474342138456975, -0.0194447309763202],
            atol=1e-13,
        )

    def test_rank_centrality_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import rank_centrality

        a = np.array([[0, 3, 1], [2, 0, 4], [5, 1, 0]], dtype=float)
        with pytest.raises(ValueError):
            rank_centrality(np.ones((2, 3)))
        with pytest.raises(ValueError):
            rank_centrality(a + 1j * a)
        with pytest.raises(ValueError):
            rank_centrality(-a)
        with pytest.raises(ValueError):
            rank_centrality(np.zeros((3, 3)))
        with pytest.raises(ValueError):
            rank_centrality(a, alpha=-1.0)
        # Disconnected comparison graph at alpha = 0; estimable at 0.5.
        d = np.array(
            [[0, 2, 0, 0], [1, 0, 0, 0], [0, 0, 0, 3], [0, 0, 1, 0]],
            dtype=float,
        )
        with pytest.raises(ValueError):
            rank_centrality(d)
        rd = rank_centrality(d, alpha=0.5)
        np.testing.assert_allclose(
            rd.weights, [9 / 8, 7 / 8, 6 / 5, 4 / 5], atol=1e-13
        )
        # Denominator overflow (2 * alpha -> inf) must raise, not NaN.
        with pytest.raises(ValueError):
            rank_centrality(a, alpha=1e308)
        # Exact scale invariance at alpha = 0 only.
        base = rank_centrality(a)
        big = rank_centrality(a * 1e20)
        assert np.allclose(big.params, base.params, atol=1e-12)

class TestPlackettLuceRankings:
    """Plackett-Luce rankings LSR/I-LSR (choix 0.4.1 lsr_rankings /
    ilsr_rankings). Pins from the executed exact-Fraction oracle."""

    def test_lsr_rankings_fixtures(self):
        import numpy as np
        from fast_mlsirm import lsr_rankings

        ra = [[1, 0, 2], [2, 1, 0], [0, 1, 2], [1, 2, 0]]
        r = lsr_rankings(ra, 3)
        np.testing.assert_allclose(
            r.weights, [6 / 11, 21 / 11, 6 / 11], rtol=0, atol=1e-14
        )
        assert r.iterations == 1
        r5 = lsr_rankings(ra, 3, alpha=0.5)
        np.testing.assert_allclose(r5.weights, [0.75, 1.5, 0.75], rtol=0, atol=1e-14)
        # Partial rankings (kills the all-items-sum mutant).
        rb = [[0, 1, 2], [3, 2], [1, 3], [2, 0, 3], [3, 1, 0]]
        rp = lsr_rankings(rb, 4)
        np.testing.assert_allclose(
            rp.weights,
            [172 / 175, 12 / 7, 16 / 35, 148 / 175],
            rtol=0,
            atol=1e-14,
        )

    def test_ilsr_rankings_fixed_point(self):
        import numpy as np
        from fast_mlsirm import ilsr_rankings

        ra = [[1, 0, 2], [2, 1, 0], [0, 1, 2], [1, 2, 0]]
        r = ilsr_rankings(ra, 3)
        np.testing.assert_allclose(
            r.params,
            [-0.39145300187318292, 0.78290600374636584, -0.39145300187318292],
            rtol=0,
            atol=1e-7,
        )
        assert r.iterations == 8
        assert (r.weights > 0).all()
        assert abs(r.weights.sum() - 3.0) < 1e-9

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import ilsr_rankings, lsr_rankings

        with pytest.raises(ValueError):
            lsr_rankings([[0]], 2)  # length-1 ranking (choix would no-op)
        with pytest.raises(ValueError):
            lsr_rankings([[0, -1]], 2)  # negative index must NOT wrap
        with pytest.raises(ValueError):
            lsr_rankings([[0, 1.5]], 2)  # non-integer item
        with pytest.raises(ValueError):
            lsr_rankings([[0, 2]], 2)  # out of range
        with pytest.raises(ValueError):
            lsr_rankings([[0, 1, 0]], 2)  # within-ranking duplicate
        with pytest.raises(ValueError):
            lsr_rankings([[0, 1]], 1)  # n < 2
        with pytest.raises(ValueError):
            lsr_rankings([[0, 1], [1, 0], [2, 3], [3, 2]], 4)  # disconnected
        with pytest.raises(ValueError):
            ilsr_rankings([[0, 1], [1, 0]], 2, max_iter=0)
        # object-dtype backstop: a non-numeric object item must raise
        # ValueError, not TypeError.
        with pytest.raises(ValueError):
            lsr_rankings([[0, object()]], 2)
        with pytest.raises(ValueError):
            lsr_rankings([[0, np.complex128(1)]], 2)
        # Impl-review regressions: np.bool_ items must be rejected like
        # Python bool; infinite items must raise ValueError (not
        # OverflowError); huge n must Err (not abort the process).
        with pytest.raises(ValueError):
            lsr_rankings([[0, np.bool_(True)]], 2, alpha=0.5)
        with pytest.raises(ValueError):
            lsr_rankings([[0, np.float64("inf")]], 2)
        with pytest.raises(ValueError):
            lsr_rankings([[0, 1]], 1_000_000)
        # Huge n / indices beyond u64 must raise ValueError before any
        # unsigned cast, never a raw OverflowError (impl-review #295).
        with pytest.raises(ValueError):
            lsr_rankings([[0, 1]], 2**81)
        with pytest.raises(ValueError):
            lsr_rankings([[0, 2**80]], 3)

class TestPlackettLuceTop1:
    """Plackett-Luce top-1 LSR/I-LSR (choix 0.4.1 lsr_top1 / ilsr_top1).
    Pins from the executed exact-Fraction oracle."""

    def test_lsr_top1_fixtures(self):
        import numpy as np
        from fast_mlsirm import lsr_top1

        ta = [(1, [0, 2]), (1, [0, 2]), (0, [1, 2]), (2, [0, 1]), (1, [0, 2])]
        r = lsr_top1(ta, 3)
        np.testing.assert_allclose(r.weights, [0.6, 1.8, 0.6], rtol=0, atol=1e-14)
        assert r.iterations == 1
        r5 = lsr_top1(ta, 3, alpha=0.5)
        np.testing.assert_allclose(
            r5.weights, [15 / 19, 27 / 19, 15 / 19], rtol=0, atol=1e-14
        )
        # Partial choice sets (kills the winner-excluded / all-items
        # denominator mutants, both invisible on TA).
        tb = [(0, [1, 2]), (3, [2]), (1, [3]), (2, [0, 3]), (3, [1, 0]), (1, [2])]
        rp = lsr_top1(tb, 4)
        np.testing.assert_allclose(
            rp.weights,
            [38 / 41, 54 / 41, 22 / 41, 50 / 41],
            rtol=0,
            atol=1e-14,
        )

    def test_ilsr_top1_fixed_point(self):
        import numpy as np
        from fast_mlsirm import ilsr_top1

        tb = [(0, [1, 2]), (3, [2]), (1, [3]), (2, [0, 3]), (3, [1, 0]), (1, [2])]
        r = ilsr_top1(tb, 4)
        np.testing.assert_allclose(
            r.params,
            [
                0.038399971408422691,
                0.26272439694352421,
                -0.56384876453083402,
                0.26272439617888712,
            ],
            rtol=0,
            atol=1e-7,
        )
        assert r.iterations == 12
        assert (r.weights > 0).all()
        assert abs(r.weights.sum() - 4.0) < 1e-9

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import ilsr_top1, lsr_top1

        with pytest.raises(ValueError):
            lsr_top1([(0, [])], 2)  # empty loser set (choix would no-op)
        with pytest.raises(ValueError):
            lsr_top1([(0, [-1])], 2)  # negative index must NOT wrap
        with pytest.raises(ValueError):
            lsr_top1([(0, [1.5])], 2)  # non-integral loser
        with pytest.raises(ValueError):
            lsr_top1([(0.5, [1])], 2)  # non-integral winner
        with pytest.raises(ValueError):
            lsr_top1([(0, [2])], 2)  # loser out of range
        with pytest.raises(ValueError):
            lsr_top1([(2, [0])], 2)  # winner out of range
        with pytest.raises(ValueError):
            lsr_top1([(0, [0, 1])], 2)  # winner in its own loser set
        with pytest.raises(ValueError):
            lsr_top1([(0, [1, 1])], 3)  # duplicate loser
        with pytest.raises(ValueError):
            lsr_top1([(0, [1])], 1)  # n < 2
        with pytest.raises(ValueError):
            lsr_top1([(0, [1]), (1, [0]), (2, [3]), (3, [2])], 4)  # disconnected
        with pytest.raises(ValueError):
            ilsr_top1([(0, [1]), (1, [0])], 2, max_iter=0)
        # Object-dtype backstop: a non-numeric object must raise
        # ValueError, not TypeError.
        with pytest.raises(ValueError):
            lsr_top1([(0, [object()])], 2)
        with pytest.raises(ValueError):
            lsr_top1([(0, [np.complex128(1)])], 2)
        with pytest.raises(ValueError):
            lsr_top1([(np.complex128(0), [1])], 2)
        # bool / np.bool_ are not item indices; inf must raise ValueError
        # (not OverflowError); huge n must Err (not abort the process).
        with pytest.raises(ValueError):
            lsr_top1([(0, [np.bool_(True)])], 2, alpha=0.5)
        with pytest.raises(ValueError):
            lsr_top1([(True, [1])], 2, alpha=0.5)
        with pytest.raises(ValueError):
            lsr_top1([(0, [np.float64("inf")])], 2)
        with pytest.raises(ValueError):
            lsr_top1([(0, [1])], 1_000_000)
        # Huge n / indices beyond u64 must raise ValueError before any
        # unsigned cast, never a raw OverflowError (impl-review finding).
        with pytest.raises(ValueError):
            lsr_top1([(0, [1])], 2**81)
        with pytest.raises(ValueError):
            lsr_top1([(0, [2**80])], 3)
        with pytest.raises(ValueError):
            lsr_top1([(2**80, [1])], 3)

class TestKendallCircular:
    """circular_triads wrapper. Every assert reads crate-returned fields.

    Fixture pins are exact-Fraction oracle values (files/kendall_oracle.py,
    EXECUTED): dog example T=5, T_max=8, zeta=3/8, p_less=1043/2048;
    exact p-values are dyadic rationals so == comparison is exact.
    """

    DOG = [
        [0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0],
    ]

    def test_dog_exact(self):
        # Mutation kill: dropping the pairing C(d,2) shifts T off 5.
        from fast_mlsirm import circular_triads

        r = circular_triads(self.DOG, alternative="less")
        assert r.t == 5.0
        assert r.t_max == 8.0
        assert r.t_exp == 5.0
        assert r.zeta == 3.0 / 8.0
        assert r.p_value == 1043.0 / 2048.0
        assert r.exact is True
        import math

        assert math.isnan(r.chi2) and math.isnan(r.df)

    def test_two_sided_and_greater(self):
        from fast_mlsirm import circular_triads

        r2 = circular_triads(self.DOG, alternative="two.sided")
        assert r2.p_value == 1.0
        rg = circular_triads(self.DOG, alternative="greater")
        assert rg.p_value == 1233.0 / 2048.0

    def test_chi2_path_n12(self):
        # n=12 tournament: i beats j (i<j) iff (i+j)%3 != 0; oracle pins
        # T=60, zeta=1/7, corrected less-tail chi2 uses corr=-0.5.
        import numpy as np

        from fast_mlsirm import circular_triads

        n = 12
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if (i + j) % 3 != 0:
                    m[i, j] = 1.0
                else:
                    m[j, i] = 1.0
        r = circular_triads(m, alternative="less")
        assert r.t == 60.0
        assert abs(r.zeta - 1.0 / 7.0) < 1e-15
        assert r.exact is False
        assert abs(r.chi2 - 121.0 / 8.0) < 1e-12
        assert abs(r.df - 165.0 / 8.0) < 1e-12
        assert abs(r.p_value - 0.7996995989775597) < 1e-9

    def test_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import circular_triads

        ok = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        with pytest.raises(ValueError):
            circular_triads([[0, 1], [0, 0]])  # incomplete pair
        with pytest.raises(ValueError):
            circular_triads([[0, 1], [1, 0]])  # n=2 rejected
        with pytest.raises(ValueError):
            circular_triads(ok, alternative="sideways")
        with pytest.raises(ValueError):
            circular_triads(np.array(ok)[:2, :])  # non-square
        with pytest.raises(ValueError):
            circular_triads(np.array(ok) + 0.5)  # non-binary
        with pytest.raises(ValueError):
            circular_triads(np.array(ok, dtype=complex))
        bad_diag = np.array(ok, dtype=float)
        bad_diag[0, 0] = 1.0
        with pytest.raises(ValueError):
            circular_triads(bad_diag)
        with pytest.raises(ValueError):
            circular_triads(np.array([["a", "b"], ["c", "d"]], dtype=object))


class TestKendallU:
    """kendall_u wrapper. Oracle pins (files/kendall_oracle.py, EXECUTED):
    KU-A Sigma=11, u=2/9, chi2=11, df=9; KU-B perfect u=1."""

    def test_fixture_a(self):
        from fast_mlsirm import kendall_u

        r = kendall_u([[0, 3, 4], [1, 0, 2], [0, 2, 0]])
        assert r.sigma == 11.0
        assert abs(r.u - 2.0 / 9.0) < 1e-15
        assert r.min_u == -1.0 / 3.0
        assert r.chi2 == 11.0
        assert r.df == 9.0
        assert abs(r.p_value - 0.27570893677222197) < 1e-9

    def test_perfect_agreement(self):
        from fast_mlsirm import kendall_u

        n, m = 4, 5
        mat = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = m
        r = kendall_u(mat)
        assert r.sigma == 60.0
        assert r.u == 1.0
        assert r.min_u == -1.0 / 5.0
        assert abs(r.chi2 - 52.0) < 1e-12
        assert abs(r.df - 40.0 / 3.0) < 1e-12

    def test_negative_chi2_raw(self):
        # n=2, m=4, perfect split: raw chi2 = -1 must NOT be clamped in
        # the returned statistic; only the p-value clamps (p = 1).
        from fast_mlsirm import kendall_u

        r = kendall_u([[0, 2], [2, 0]])
        assert r.chi2 == -1.0
        assert r.min_u == -1.0 / 3.0
        assert abs(r.u - r.min_u) < 1e-15
        assert r.p_value == 1.0

    def test_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import kendall_u

        with pytest.raises(ValueError):
            kendall_u([[0, 3, 3], [1, 0, 3], [1, 2, 0]])  # unequal m across pairs
        with pytest.raises(ValueError):
            kendall_u([[0, 1], [1, 0]])  # m=2 < 3
        with pytest.raises(ValueError):
            kendall_u([[0, 1.5], [1.5, 0]])  # non-integer
        with pytest.raises(ValueError):
            kendall_u([[0, -1], [4, 0]])  # negative
        with pytest.raises(ValueError):
            kendall_u([[5]])  # n=1 and nonzero diagonal
        with pytest.raises(ValueError):
            kendall_u(np.array([[0, 2], [2, 0]], dtype=complex))


class TestElo:
    """Elo rating (PlayerRatings elo() semantics); pins from the executed
    exact-rational oracle. Every assert reads wrapper/crate outputs."""

    def test_ea_exact_anchor(self):
        import numpy as np

        from fast_mlsirm import elo_rating

        r = elo_rating(
            [[1, 0, 1, 1.0], [1, 0, 2, 0.5]], 3, init=2000, kfac=32, gamma=[0.0, 400.0]
        )
        assert abs(r.ratings[0] - 22032 / 11) < 1e-12
        assert r.ratings[1] == 1984.0  # dyadic, exact
        assert abs(r.ratings[2] - 22144 / 11) < 1e-12
        assert list(r.games) == [2, 1, 1]
        assert list(r.wins) == [1, 0, 0]
        assert list(r.draws) == [1, 0, 1]
        assert list(r.losses) == [0, 1, 0]
        assert list(r.lag) == [0, 0, 0]
        assert isinstance(r.ratings, np.ndarray)

    def test_eb_batch_and_lag(self):
        from fast_mlsirm import elo_rating

        r = elo_rating(
            [[1, 0, 1, 1.0], [1, 0, 2, 0.5], [2, 0, 1, 1.0]], 3, init=2000, kfac=400
        )
        assert abs(r.ratings[0] - 24600 / 11) < 1e-12
        assert abs(r.ratings[1] - 19400 / 11) < 1e-12
        assert r.ratings[2] == 2000.0
        assert list(r.lag) == [0, 0, 1]

    def test_gamma_scalar_broadcast(self):
        from fast_mlsirm import elo_rating

        # ED closed-form nonzero-gamma pin; differs from gamma=0 (2016/1984).
        r = elo_rating([[1, 0, 1, 1.0]], 2, init=2000, kfac=32, gamma=100.0)
        assert abs(r.ratings[0] - 2011.5179200063076) < 1e-12
        assert abs(r.ratings[1] - 1988.4820799936924) < 1e-12
        r0 = elo_rating([[1, 0, 1, 1.0]], 2, init=2000, kfac=32)
        assert r0.ratings[0] == 2016.0
        assert r0.ratings[1] == 1984.0

    def test_defaults_and_conservation(self):
        from fast_mlsirm import elo_rating

        # PlayerRatings defaults init=2200, kfac=27; rating-sum conservation.
        r = elo_rating([[1, 0, 1, 1.0], [2, 1, 2, 0.5]], 3)
        assert abs(float(r.ratings.sum()) - 3 * 2200.0) < 1e-9

    def test_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import elo_rating

        with pytest.raises(ValueError):
            elo_rating([[1, 0, 0, 1.0]], 2)  # self-play
        with pytest.raises(ValueError):
            elo_rating([[1, 0, 2, 1.0]], 2)  # index out of range
        with pytest.raises(ValueError):
            elo_rating([[1, 0, 1, 1.5]], 2)  # score out of range
        with pytest.raises(ValueError):
            elo_rating([[1, 0.5, 1, 1.0]], 2)  # non-integral player index
        with pytest.raises(ValueError):
            elo_rating([[-1, 0, 1, 1.0]], 2)  # negative period
        with pytest.raises(ValueError):
            elo_rating([[1, 0, 1]], 2)  # wrong shape
        with pytest.raises(ValueError):
            elo_rating(np.zeros((0, 4)), 2)  # no games
        with pytest.raises(ValueError):
            elo_rating([[1, 0, 1, np.nan]], 2)  # non-finite
        with pytest.raises(ValueError):
            elo_rating(np.array([[1, 0, 1, 1.0]], dtype=complex), 2)  # complex
        with pytest.raises(ValueError):
            elo_rating([[1, 0, 1, 1.0]], 2, gamma=[1.0, 2.0])  # gamma length
        with pytest.raises(ValueError):
            elo_rating([["a", 0, 1, 1.0]], 2)  # object dtype backstop

    def test_large_period_labels_exact(self):
        import numpy as np
        import pytest

        from fast_mlsirm import elo_rating

        # Integer-array path keeps u64 period fidelity above 2**53: the two
        # labels are DISTINCT periods, so sequential (not batch) updates
        # apply and player 0 loses rating (impl-review finding fix).
        big = 2**53
        g = np.array(
            [[big, 0, 1, 1], [big + 1, 1, 0, 1]], dtype=np.uint64
        )
        r = elo_rating(g, 2, init=2000, kfac=400)
        assert abs(r.ratings[0] - (2200 - 400 * 10 / 11)) < 1e-9
        assert abs(float(r.ratings.sum()) - 4000.0) < 1e-9
        # Float path must REJECT unrepresentable labels, not merge them.
        with pytest.raises(ValueError):
            elo_rating([[float(big + 1), 0, 1, 1.0]], 2)
        with pytest.raises(ValueError):
            elo_rating([[1e300, 0, 1, 1.0]], 2)

    def test_float32_period_labels_rejected(self):
        import numpy as np
        import pytest

        from fast_mlsirm import elo_rating

        # float32 loses integer fidelity above 2**24 (round-2 review):
        # labels 2**24 and 2**24+1 merge in the raw array, so the wrapper
        # must reject float32 periods at/above 2**24 rather than batch them.
        g = np.array(
            [[2**24, 0, 1, 1.0], [2**24 + 1, 1, 0, 1.0]], dtype=np.float32
        )
        with pytest.raises(ValueError):
            elo_rating(g, 2)
        # below the bound the float32 path still works (crate output read)
        g_ok = np.array([[1, 0, 1, 1.0], [2, 1, 0, 1.0]], dtype=np.float32)
        r = elo_rating(g_ok, 2, init=2000, kfac=400)
        assert abs(r.ratings[0] - (2200 - 400 * 10 / 11)) < 1e-9
        # boundary: 2**24 - 1 is exactly representable and must be accepted
        # (glicko impl-review: nmant excludes hidden bit; bound is nmant + 1)
        g_edge = np.array([[2**24 - 1, 0, 1, 1.0]], dtype=np.float32)
        r_edge = elo_rating(g_edge, 2)
        assert r_edge.games.tolist() == [1, 1]
        r53 = elo_rating([[float(2**53 - 1), 0, 1, 1.0]], 2)
        assert r53.games.tolist() == [1, 1]


class TestGlicko:
    """Glicko rating wrapper (Glickman's note + PlayerRatings glicko();
    every assert reads crate-returned values via the Python wrapper)."""

    def test_paper_anchor_ga(self):
        import numpy as np

        from fast_mlsirm import glicko_rating

        # Glickman's worked example: heterogeneous per-player init arrays.
        g = [[1, 0, 1, 1.0], [1, 0, 2, 0.0], [1, 0, 3, 0.0]]
        r = glicko_rating(
            g,
            4,
            init=(
                np.array([1500.0, 1400.0, 1550.0, 1700.0]),
                np.array([200.0, 30.0, 100.0, 300.0]),
            ),
            cval=0.0,
        )
        assert abs(r.ratings[0] - 1464.1064627569112) < 1e-12
        assert abs(r.deviations[0] - 151.39890244796933) < 1e-12
        assert r.games.tolist() == [3, 1, 1, 1]
        assert r.wins.tolist() == [1, 0, 1, 1]
        assert r.losses.tolist() == [2, 1, 0, 0]

    def test_two_period_gb(self):
        from fast_mlsirm import glicko_rating

        # Scalar-pair init broadcast; inflation + lag + idle player pins.
        g = [[1, 0, 1, 1.0], [1, 1, 2, 0.5], [2, 2, 0, 1.0]]
        r = glicko_rating(g, 3, init=(2200.0, 300.0), cval=15.0)
        er = [2190.0061185685217, 2094.5895980175137, 2345.066036571678]
        ed = [223.91939158372585, 224.94066563436596, 223.91939158372585]
        for p in range(3):
            assert abs(r.ratings[p] - er[p]) < 1e-12
            assert abs(r.deviations[p] - ed[p]) < 1e-12
        assert r.lag.tolist() == [0, 1, 0]
        assert r.draws.tolist() == [0, 1, 1]
        # Documented non-identity: Glicko does NOT conserve the rating sum.
        assert abs(float(r.ratings.sum()) - 6600.0) > 1e-3

    def test_gamma_and_clamp(self):
        from fast_mlsirm import glicko_rating

        # GD gamma pins (kills a wrapper-level gamma mixup).
        r = glicko_rating(
            [[1, 0, 1, 1.0]], 2, init=(2000.0, 200.0), gamma=30.0, cval=0.0
        )
        assert abs(r.ratings[0] - 2072.980891506514) < 1e-12
        assert abs(r.ratings[1] - 1927.019108493486) < 1e-12
        # GC rdmax clamp active.
        rc = glicko_rating(
            [[1, 0, 1, 0.5]], 2, init=(2200.0, 300.0), cval=400.0, rdmax=350.0
        )
        assert abs(rc.deviations[0] - 290.2305060910912) < 1e-12
        assert rc.ratings.tolist() == [2200.0, 2200.0]

    def test_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import glicko_rating

        ok = [[1, 0, 1, 1.0]]
        with pytest.raises(ValueError):
            glicko_rating([], 2)
        with pytest.raises(ValueError):
            glicko_rating([[1, 0, 1]], 2)  # wrong width
        with pytest.raises(ValueError):
            glicko_rating([[1.5, 0, 1, 1.0]], 2)  # fractional period
        with pytest.raises(ValueError):
            glicko_rating([[1, 0, 0, 1.0]], 2)  # self-play
        with pytest.raises(ValueError):
            glicko_rating(ok, 1)  # n < 2
        with pytest.raises(ValueError):
            glicko_rating(np.array(ok, dtype=complex), 2)
        with pytest.raises(ValueError):
            glicko_rating(ok, 2, init=(2200.0,))  # not a pair
        with pytest.raises(ValueError):
            glicko_rating(ok, 2, init=(np.array([2200.0]), np.array([300.0])))
        with pytest.raises(ValueError):
            glicko_rating(ok, 2, init=(2200.0, 400.0), rdmax=350.0)  # dev>rdmax
        with pytest.raises(ValueError):
            glicko_rating(ok, 2, cval=-1.0)
        with pytest.raises(ValueError):
            glicko_rating(ok, 2, gamma=[1.0, 2.0])  # wrong gamma length
        # float-path period-fidelity bound (inherited elo contract)
        with pytest.raises(ValueError):
            glicko_rating([[float(2**53), 0, 1, 1.0]], 2)
        g32 = np.array([[2**24, 0, 1, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            glicko_rating(g32, 2)
        # boundary acceptance: labels below the exact-integer bound must pass
        # (impl-review: nmant excludes the hidden bit, bound is nmant + 1).
        # Asserts read crate outputs (games tally) — kills a 2**nmant mutant.
        r53 = glicko_rating([[float(2**53 - 1), 0, 1, 1.0]], 2)
        assert r53.games.tolist() == [1, 1]
        g32ok = np.array([[2**24 - 1, 0, 1, 1.0]], dtype=np.float32)
        r24 = glicko_rating(g32ok, 2)
        assert r24.games.tolist() == [1, 1]
        # integer-dtype lossless path (crate output read)
        gi = np.array([[2**53, 0, 1, 1], [2**53 + 1, 1, 0, 1]], dtype=np.uint64)
        r = glicko_rating(gi, 2, init=(2200.0, 300.0), cval=0.0)
        assert r.lag.tolist() == [0, 0]
        assert r.games.tolist() == [2, 2]


class TestGlicko2:
    """Glicko-2 rating wrapper (Glickman 2022 + PlayerRatings glicko2();
    every assert reads crate-returned values via the Python wrapper)."""

    def test_paper_anchor_g2a(self):
        import numpy as np

        from fast_mlsirm import glicko2_rating

        # Glickman's (2022) worked example, heterogeneous init triple.
        g = [[1, 0, 1, 1.0], [1, 0, 2, 0.0], [1, 0, 3, 0.0]]
        r = glicko2_rating(
            g,
            4,
            init=(
                np.array([1500.0, 1400.0, 1550.0, 1700.0]),
                np.array([200.0, 30.0, 100.0, 300.0]),
                np.array([0.06] * 4),
            ),
            tau=0.5,
        )
        assert abs(r.ratings[0] - 1464.0506708196929) < 1e-7
        assert abs(r.deviations[0] - 151.51652192592556) < 1e-7
        assert abs(r.volatilities[0] - 0.05999598428664987) < 1e-7
        assert r.games.tolist() == [3, 1, 1, 1]
        assert r.wins.tolist() == [1, 0, 1, 1]
        assert r.losses.tolist() == [2, 1, 0, 0]

    def test_two_period_g2b(self):
        from fast_mlsirm import glicko2_rating

        # Scalar-triple init broadcast; inflation + lag + idle player pins.
        g = [[1, 0, 1, 1.0], [1, 1, 2, 0.5], [2, 0, 2, 0.0]]
        r = glicko2_rating(g, 3, init=(2200.0, 300.0, 0.15), tau=1.2)
        er = [2189.2376693682463, 2094.286091224066, 2346.354869237831]
        ev = [0.15002063478252045, 0.1498718983333671, 0.14993884055293957]
        for p in range(3):
            assert abs(r.ratings[p] - er[p]) < 1e-7
            assert abs(r.volatilities[p] - ev[p]) < 1e-7
        assert r.lag.tolist() == [0, 1, 0]
        # Documented non-identity: rating sum is NOT conserved.
        assert abs(float(r.ratings.sum()) - 6600.0) > 1e-3

    def test_gamma_and_tau0(self):
        from fast_mlsirm import glicko2_rating

        # G2D: favored white draws with gamma=30 and DROPS (crate ratings).
        r = glicko2_rating(
            [[1, 0, 1, 0.5]], 2, init=(2200.0, 300.0, 0.15), gamma=30.0, tau=1.2
        )
        assert abs(r.ratings[0] - 2191.52226104287) < 1e-7
        assert abs(r.ratings[1] - 2208.47773895713) < 1e-7
        # tau = 0 freezes volatility EXACTLY (crate volatilities read).
        rf = glicko2_rating(
            [[1, 0, 1, 0.25], [2, 1, 2, 1.0]], 3, init=(2200.0, 300.0, 0.15), tau=0.0
        )
        assert rf.volatilities.tolist() == [0.15, 0.15, 0.15]
        assert rf.wins.tolist() == [0, 1, 0]
        assert rf.games.tolist() == [1, 2, 1]

    def test_error_contract(self):
        import numpy as np
        import pytest

        from fast_mlsirm import glicko2_rating

        ok = [[1, 0, 1, 1.0]]
        with pytest.raises(ValueError):
            glicko2_rating([], 2)
        with pytest.raises(ValueError):
            glicko2_rating([[1, 0, 1]], 2)  # wrong width
        with pytest.raises(ValueError):
            glicko2_rating([[1.5, 0, 1, 1.0]], 2)  # fractional period
        with pytest.raises(ValueError):
            glicko2_rating([[1, 0, 0, 1.0]], 2)  # self-play
        with pytest.raises(ValueError):
            glicko2_rating(ok, 1)  # n < 2
        with pytest.raises(ValueError):
            glicko2_rating(np.array(ok, dtype=complex), 2)
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, init=(2200.0, 300.0))  # pair, not triple
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, init=(2200.0, 300.0, 0.0))  # vol <= 0
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, init=(2200.0, 300.0, 3.0))  # vol > q*rdmax
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, tau=-0.5)  # negative tau
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, rdmax=0.0)
        with pytest.raises(ValueError):
            # mixed scalar/array init rejected
            glicko2_rating(
                ok, 2, init=(np.array([2200.0, 2200.0]), 300.0, 0.15)
            )
        with pytest.raises(ValueError):
            glicko2_rating([[1, 0, 1, "x"]], 2)  # non-numeric
        with pytest.raises(ValueError):
            g = np.array(ok, dtype=object)
            g[0][3] = object()
            glicko2_rating(g, 2)
        # --- coverage added in impl-review round 2 (all must be ValueError,
        # not TypeError/OverflowError leaks) ---
        with pytest.raises(ValueError):
            glicko2_rating(ok, float("inf"))  # int(n_players) overflow
        with pytest.raises(ValueError):
            glicko2_rating(ok, None)  # int(None) TypeError
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, gamma=object())  # asarray TypeError
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, tau=None)  # float(None) TypeError
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, rdmax=None)
        with pytest.raises(ValueError):
            glicko2_rating([[1, 0, 2, 1.0]], 2)  # player index out of range
        with pytest.raises(ValueError):
            glicko2_rating([[1, -1, 1, 1.0]], 2)  # negative player index
        with pytest.raises(ValueError):
            glicko2_rating([[1, 0, 1, 1.5]], 2)  # score > 1
        with pytest.raises(ValueError):
            glicko2_rating([[1, 0, 1, -0.5]], 2)  # score < 0
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, gamma=[10.0, 20.0])  # gamma length != g
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, gamma=float("nan"))  # non-finite gamma
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, gamma=1j)  # complex gamma
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, tau=float("nan"))
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, rdmax=float("inf"))
        with pytest.raises(ValueError):
            glicko2_rating(ok, 2, rdmax=-350.0)
        with pytest.raises(ValueError):
            glicko2_rating(
                ok, 2, init=(np.array([np.nan, 2200.0]), np.array([300.0] * 2),
                             np.array([0.15] * 2))
            )
        with pytest.raises(ValueError):
            glicko2_rating(
                ok, 2, init=(np.array([2200.0]), np.array([300.0]),
                             np.array([0.15]))
            )  # init length != n_players
        with pytest.raises(ValueError):
            glicko2_rating(ok, 10001)  # n_players cap

    def test_period_fidelity(self):
        import numpy as np

        from fast_mlsirm import glicko2_rating

        # float64 2**53 rejected (labels at/above the exact-integer bound)
        import pytest

        with pytest.raises(ValueError):
            glicko2_rating([[float(2**53), 0, 1, 1.0]], 2)
        g32 = np.array([[2**24, 0, 1, 1.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            glicko2_rating(g32, 2)
        # Largest exactly-representable labels ACCEPTED; asserts read crate
        # outputs (games tally) — kills a 2**nmant mutant.
        r53 = glicko2_rating([[float(2**53 - 1), 0, 1, 1.0]], 2)
        assert r53.games.tolist() == [1, 1]
        g32ok = np.array([[2**24 - 1, 0, 1, 1.0]], dtype=np.float32)
        r24 = glicko2_rating(g32ok, 2)
        assert r24.games.tolist() == [1, 1]
        # integer-dtype lossless path (crate output read)
        gi = np.array([[2**53, 0, 1, 1], [2**53 + 1, 1, 0, 1]], dtype=np.uint64)
        r = glicko2_rating(gi, 2, init=(2200.0, 300.0, 0.15), tau=0.0)
        assert r.lag.tolist() == [0, 0]
        assert r.games.tolist() == [2, 2]


class TestStephenson:
    """stephenson_rating vs the EXECUTED PlayerRatings-1.1.0 oracle
    (files/stephenson_oracle.py port of ratings.R steph() + ratings.c
    stephenson_c). Every assert reads crate outputs returned through the
    binding; anchors are float pins from the oracle run."""

    def test_anchor_s1_heterogeneous_init(self):
        # Kills: opponent-g swap (MU5), (lag+1) inflation drop (MU4) --
        # heterogeneous init deviations make own-g vs opponent-g diverge.
        import numpy as np

        from fast_mlsirm import stephenson_rating

        res = stephenson_rating(
            [[1, 0, 1, 1.0]],
            3,
            init=(
                np.array([2200.0, 2300.0, 2100.0]),
                np.array([300.0, 80.0, 150.0]),
            ),
        )
        assert res.ratings[0] == pytest.approx(2395.927345489556, rel=1e-12)
        assert res.deviations[0] == pytest.approx(233.84458992614066, rel=1e-12)
        assert res.ratings[1] == pytest.approx(2281.863290035741, rel=1e-12)
        assert res.deviations[1] == pytest.approx(80.14766179107052, rel=1e-12)
        # Non-participant keeps init state exactly (identity step-4 path).
        assert res.ratings[2] == 2100.0
        assert res.deviations[2] == 150.0
        assert list(res.games) == [1, 1, 0]
        assert list(res.wins) == [1, 0, 0]
        assert list(res.losses) == [0, 1, 0]
        assert list(res.lag) == [0, 0, 0]

    def test_full_knobs_s3(self):
        # Kills: bval drop (MU1), lambda sign flip (MU2), per-game hval
        # scaling drop (MU3) -- all knobs nonzero and gamma asymmetric.
        from fast_mlsirm import stephenson_rating

        res = stephenson_rating(
            [[1, 0, 1, 0.5], [2, 1, 2, 1.0]],
            3,
            gamma=[30.0, 0.0],
            cval=8.0,
            hval=15.0,
            bval=5.0,
            lambda_=5.0,
        )
        assert res.ratings[0] == pytest.approx(2205.081933917611, rel=1e-12)
        assert res.deviations[0] == pytest.approx(254.8041343216954, rel=1e-12)
        assert res.ratings[1] == pytest.approx(2332.590625814148, rel=1e-12)
        assert res.deviations[1] == pytest.approx(225.48194679176544, rel=1e-12)
        assert res.ratings[2] == pytest.approx(2082.6278256782653, rel=1e-12)
        assert res.deviations[2] == pytest.approx(249.45302895967546, rel=1e-12)
        assert list(res.lag) == [1, 0, 0]
        assert list(res.draws) == [1, 1, 0]

    def test_prior_run_continuation_s4(self):
        # init_games/init_lag path + rdmax clamp: (lag+1)*cval^2 would blow
        # past rdmax=350 without the clamp; games accumulate 5+1.
        from fast_mlsirm import stephenson_rating

        res = stephenson_rating(
            [[1, 0, 1, 1.0]],
            2,
            init=(2200.0, 340.0),
            cval=60.0,
            init_games=[5, 5],
            init_lag=[4, 4],
        )
        assert res.ratings[0] == pytest.approx(2362.303032952076, rel=1e-12)
        assert res.deviations[0] == pytest.approx(290.31193063897507, rel=1e-12)
        assert list(res.games) == [6, 6]
        # Current-run tallies only (init run's W/D/L are not represented).
        assert list(res.wins) == [1, 0]

    def test_lambda_zero_contrast(self):
        # Same schedule as oracle S2: lambda=0 must reproduce the S2-lam0
        # pins and differ from S2 defaults only on participants of the
        # final period's drift (p1 identical).
        from fast_mlsirm import stephenson_rating

        sched = [[1, 0, 1, 1.0], [1, 1, 2, 0.5], [2, 0, 2, 0.0]]
        base = stephenson_rating(sched, 3)
        lam0 = stephenson_rating(sched, 3, lambda_=0.0)
        assert base.ratings[0] == pytest.approx(2187.33261103678, rel=1e-12)
        assert lam0.ratings[0] == pytest.approx(2190.0339057939964, rel=1e-12)
        assert lam0.ratings[2] == pytest.approx(2345.0308320668346, rel=1e-12)
        assert lam0.ratings[1] == base.ratings[1]
        assert lam0.deviations[0] == base.deviations[0]

    def test_error_contract(self):
        import numpy as np

        from fast_mlsirm import stephenson_rating

        good = [[1, 0, 1, 1.0]]
        with pytest.raises(ValueError):
            stephenson_rating([], 2)
        with pytest.raises(ValueError):
            stephenson_rating([[1, 0, 1]], 2)
        with pytest.raises(ValueError):
            stephenson_rating(np.array(good, dtype=complex), 2)
        with pytest.raises(ValueError):
            stephenson_rating([[1, 0, 1, np.nan]], 2)
        with pytest.raises(ValueError):
            stephenson_rating([[1.5, 0, 1, 1.0]], 2)
        with pytest.raises(ValueError):
            stephenson_rating([[-1, 0, 1, 1.0]], 2)
        with pytest.raises(ValueError):
            stephenson_rating(good, 2.5)
        with pytest.raises(ValueError):
            stephenson_rating(good, None)
        with pytest.raises(ValueError):
            stephenson_rating(good, 1)
        with pytest.raises(ValueError):
            stephenson_rating(good, 10001)
        with pytest.raises(ValueError):
            stephenson_rating([[1, 0, 2, 1.0]], 2)  # index out of range
        with pytest.raises(ValueError):
            stephenson_rating([[1, 0, 0, 1.0]], 2)  # self-play
        with pytest.raises(ValueError):
            stephenson_rating([[1, 0, 1, 1.5]], 2)  # score out of [0,1]
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, gamma=[1.0, 2.0])
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, gamma=np.array([1j]))
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, init=(2200.0,))
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, init=(np.array([2200.0]), np.array([300.0])))
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, init=(2200.0, 400.0), rdmax=350.0)
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, init_games=[1, 2, 3])
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, init_games=[-1, 0])
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, init_lag=[0.5, 0.0])
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, cval=np.nan)
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, cval=-1.0)
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, rdmax=0.0)
        with pytest.raises(ValueError):
            stephenson_rating(good, 2, lambda_=None)
        with pytest.raises(ValueError):
            stephenson_rating(np.array([["a", "b", "c", "d"]], dtype=object), 2)
        # Float player ids at/above 2**53 lose integer fidelity pre-cast.
        with pytest.raises(ValueError):
            stephenson_rating([[1, 2.0**53, 1, 1.0]], 2)
        with pytest.raises(ValueError):
            stephenson_rating([[1, 0, 2.0**53, 1.0]], 2)
        # n_players cap rejects BEFORE any length-n allocation.
        with pytest.raises(ValueError):
            stephenson_rating([[1, 0, 1, 1.0]], 10**18)


class TestElom:
    def test_e1_scalar_single_event(self):
        import numpy as np

        from fast_mlsirm import elom_rating

        r = elom_rating(
            periods=np.array([1], dtype=np.uint64),
            players=np.array([[0, 1, 2]], dtype=np.int64),
            scores=np.array([[3.0, 2.0, 1.0]]),
            n_players=3,
            base=(20.0, 0.0, -20.0),
            kfac=0.5,
        )
        # Reads crate outputs (oracle E1). Killed by K-scaling or base-rank
        # mutants: ratings would leave [1510, 1500, 1490].
        assert list(r.ratings) == [1510.0, 1500.0, 1490.0]
        assert list(r.games) == [1, 1, 1]
        assert r.places.shape == (3, 3)
        assert list(r.places[0]) == [1, 0, 0]
        assert list(r.lag) == [0, 0, 0]

    def test_e8_once_shrunk_base(self):
        import numpy as np

        from fast_mlsirm import elom_rating

        # nn=4, two empty seats: PlayerRatings shrinks the ORIGINAL base
        # exactly once (not once per empty seat). Oracle E8.
        r = elom_rating(
            periods=np.array([1], dtype=np.uint64),
            players=np.array([[0, 1, -1, -1]], dtype=np.int64),
            scores=np.array([[2.0, 1.0, np.nan, np.nan]]),
            n_players=3,
            base=(30.0, 10.0, -10.0, -30.0),
            kfac=1.0,
        )
        # Once-shrunk base (30, 0, -30); ranks 1, 2 -> +30, +0 minus the
        # expected term 0 (equal ratings). Cumulative-shrink mutant gives
        # a different base and fails this crate-value pin.
        assert list(r.ratings) == [1530.0, 1500.0, 1500.0]
        assert list(r.games) == [1, 1, 0]

    def test_kriichi_default_and_validation(self):
        import numpy as np
        import pytest

        from fast_mlsirm import elom_rating

        periods = np.array([1], dtype=np.uint64)
        players = np.array([[0, 1, 2, 3]], dtype=np.int64)
        scores = np.array([[4.0, 3.0, 2.0, 1.0]])
        # Default kriichi(gv=400, kv=0.2) with zero games: K = 1 exactly.
        r = elom_rating(periods, players, scores, n_players=4)
        assert list(r.ratings) == [1530.0, 1510.0, 1490.0, 1470.0]
        with pytest.raises(ValueError):
            elom_rating(periods, players, scores, n_players=1)
        with pytest.raises(ValueError):
            elom_rating(periods, players + 1j, scores, n_players=4)
        with pytest.raises(ValueError):
            elom_rating(periods, players, scores, n_players=4, kfac=("bogus", 1, 1))
        with pytest.raises(ValueError):
            # NaN score in an occupied seat must be rejected by the core.
            bad = scores.copy()
            bad[0, 0] = np.nan
            elom_rating(periods, players, bad, n_players=4)
        with pytest.raises(ValueError):
            # uint64::MAX must not wrap to the -1 empty-seat sentinel
            # through the int64 narrowing cast (impl-review finding 1).
            up = np.array([[0, 1, 2, np.iinfo(np.uint64).max]], dtype=np.uint64)
            us = np.array([[4.0, 3.0, 2.0, np.nan]])
            elom_rating(periods, up, us, n_players=4)


class TestMetrics:
    def test_anchor_m1_and_cap_quirk(self):
        import numpy as np
        from fast_mlsirm import metrics_rating

        out = metrics_rating(
            [1.0, 0.0, 1.0, 0.0], [0.75, 0.25, 0.5, 0.5], scale=False
        )
        assert out.shape == (1, 3)
        assert abs(out[0, 0] - 49.041462650586311) < 1e-12
        assert abs(out[0, 1] - 39.528470752104745) < 1e-12
        assert out[0, 2] == 37.5
        # cap quirk: bdev capped, mse/mae raw
        q = metrics_rating([1.0, 0.0], [0.999, 0.001], scale=False)
        assert abs(q[0, 0] - 1.0050335853501451) < 1e-12
        assert abs(q[0, 1] - 0.1) < 1e-12
        assert abs(q[0, 2] - 0.1) < 1e-12

    def test_scaled_two_columns(self):
        import numpy as np
        from fast_mlsirm import metrics_rating

        pred = np.array(
            [[0.75, 0.50], [0.25, 0.50], [0.50, 0.25], [0.50, 0.75]]
        )
        out = metrics_rating([1.0, 0.0, 1.0, 0.0], pred, scale=False)
        assert out.shape == (2, 3)
        assert abs(out[0, 0] - 49.041462650586311) < 1e-12
        assert abs(out[1, 0] - 103.97207708399179) < 1e-12
        assert abs(out[1, 1] - 63.737743919909803) < 1e-12
        assert out[1, 2] == 62.5

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import metrics_rating

        with pytest.raises(ValueError, match="real-valued"):
            metrics_rating(np.array([1j, 0.0]), [0.6, 0.4])
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating(np.array(["a", "b"], dtype=object), [0.6, 0.4])
        with pytest.raises(ValueError, match="length"):
            metrics_rating([1.0], [0.6, 0.4])
        with pytest.raises(ValueError, match="cap"):
            metrics_rating([1.0, 0.0], [0.6, 0.4], cap=(0.5,))
        with pytest.raises(ValueError, match="infinite"):
            metrics_rating([1.0, np.inf], [0.6, 0.4])
        with pytest.raises(ValueError, match="0.5"):
            metrics_rating([0.5, 0.5], [0.6, 0.4], scale=True)
        # regressions: string/bool inputs and 0-D scalars are rejected
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating(["1", "0"], [0.6, 0.4])
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating(
                np.array(["1", "0"], dtype=object), [0.6, 0.4]
            )
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating([1.0, 0.0], [0.6, 0.4], cap=("0.01", "0.99"))
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating(np.array([True, False]), [0.6, 0.4])
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating(
                np.array([np.bool_(True), np.bool_(False)], dtype=object),
                [0.6, 0.4],
            )
        with pytest.raises(ValueError, match="1-D"):
            metrics_rating(np.array(1.0), np.array(0.6))
        # regression: object-dtype None must not silently become NaN
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating(np.array([1.0, None], dtype=object), [0.6, 0.4])
        with pytest.raises(ValueError, match="numeric"):
            metrics_rating([1.0, 0.0], np.array([0.6, None], dtype=object))
        # regression: masked arrays must not silently drop their mask
        with pytest.raises(ValueError, match="masked"):
            metrics_rating(
                np.ma.masked_array([1.0, 0.0], mask=[False, True]),
                [0.6, 0.4],
            )
        with pytest.raises(ValueError, match="masked"):
            metrics_rating(
                [1.0, 0.0],
                np.ma.masked_array([0.6, 0.4], mask=[False, True]),
            )


class TestFide:
    def test_anchor_f1_exact(self):
        import numpy as np
        from fast_mlsirm import fide_rating

        res = fide_rating([[1, 0, 1, 1.0]], 2)
        assert res.ratings.tolist() == [2215.0, 2185.0]
        # opponent averages read POST-update ratings (R quirk)
        assert res.opponent.tolist() == [2185.0, 2215.0]
        assert res.games.tolist() == [1, 1]
        assert res.wins.tolist() == [1, 0]
        assert res.losses.tolist() == [0, 1]
        assert res.elite.tolist() == [0, 0]

    def test_k_switch_f2(self):
        import numpy as np
        from fast_mlsirm import fide_rating

        rows = [[1, 0, i, 1.0] for i in range(1, 30)]
        rows.append([2, 0, 1, 1.0])
        res = fide_rating(np.array(rows), 31, init=1500.0)
        # 29 period-start games in period 2 -> K stays kv[2]=30
        assert abs(res.ratings[0] - 1937.0927486207672) < 1e-12
        assert abs(res.ratings[1] - 1482.9072513792328) < 1e-12
        assert res.ratings[2] == 1485.0
        assert res.ratings[30] == 1500.0
        assert abs(res.opponent[0] - 1484.9302417126412) < 1e-12
        assert res.games.tolist()[:2] == [30, 2]
        assert res.elite.tolist() == [0] * 31

    def test_elite_sticky_f3(self):
        import numpy as np
        from fast_mlsirm import fide_rating

        res = fide_rating(
            [[1, 0, 1, 1.0], [2, 0, 1, 0.0], [3, 0, 1, 0.0], [4, 0, 1, 1.0]],
            2,
            init=2395.0,
        )
        assert res.elite.tolist() == [1, 1]
        assert abs(res.ratings[0] - 2404.6257234642635) < 1e-12
        assert abs(res.ratings[1] - 2406.4738023175405) < 1e-12
        assert abs(res.opponent[0] - 2398.603771437728) < 1e-12

    def test_opponent_weights_f5(self):
        import numpy as np
        from fast_mlsirm import fide_rating

        res = fide_rating(
            [[1, 0, 1, 1.0], [2, 0, 2, 1.0], [3, 0, 3, 0.5]], 4
        )
        assert abs(res.ratings[0] - 2228.088544238428) < 1e-12
        assert abs(res.opponent[0] - 2190.6371519205236) < 1e-12
        assert res.opponent[1] == 2215.0
        assert abs(res.opponent[3] - 2228.088544238428) < 1e-12
        assert res.draws.tolist() == [1, 0, 0, 1]
        assert res.lag.tolist() == [0, 2, 1, 0]

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import fide_rating

        with pytest.raises(ValueError, match="complex"):
            fide_rating(np.array([[1, 0, 1, 1.0 + 0j]]), 2)
        with pytest.raises(ValueError, match="numeric"):
            fide_rating(np.array([[1, 0, 1, "x"]], dtype=object), 2)
        with pytest.raises(ValueError, match="masked"):
            fide_rating(
                np.ma.masked_array(
                    [[1, 0, 1, 1.0]], mask=[[False, False, False, True]]
                ),
                2,
            )
        with pytest.raises(ValueError, match=r"\(g, 4\)"):
            fide_rating([[1, 0, 1]], 2)
        with pytest.raises(ValueError, match="self-play"):
            fide_rating([[1, 0, 0, 1.0]], 2)
        with pytest.raises(ValueError, match="score"):
            fide_rating([[1, 0, 1, 1.5]], 2)
        with pytest.raises(ValueError, match="two players"):
            fide_rating([[1, 0, 1, 1.0]], 1)
        with pytest.raises(ValueError, match="triple"):
            fide_rating([[1, 0, 1, 1.0]], 2, kv=(10.0, 15.0))
        with pytest.raises(ValueError, match="finite"):
            fide_rating([[1, 0, 1, 1.0]], 2, kv=(10.0, 15.0, np.inf))
        with pytest.raises(ValueError, match="integral"):
            fide_rating([[1.5, 0, 1, 1.0]], 2)
        with pytest.raises(ValueError, match="gamma"):
            fide_rating([[1, 0, 1, 1.0]], 2, gamma=[0.0, 0.0])
        # regressions: bool/datetime games silently coerced to numbers
        with pytest.raises(ValueError, match="numeric"):
            fide_rating([[1, 0, 1, True]], 2)
        with pytest.raises(ValueError, match="numeric"):
            fide_rating([[True, 0, 1, 1.0]], 2)
        with pytest.raises(ValueError, match="numeric"):
            fide_rating(np.array([[True, False, True, True]]), 2)
        with pytest.raises(ValueError, match="numeric"):
            fide_rating(
                np.array(
                    [[np.datetime64("2020-01-01"), 0, 1, 1.0]], dtype=object
                ),
                2,
            )
        # regressions: complex kv/init raise ValueError, not TypeError
        with pytest.raises(ValueError, match="complex"):
            fide_rating([[1, 0, 1, 1.0]], 2, kv=(10 + 1j, 15, 30))
        with pytest.raises(ValueError, match="complex"):
            fide_rating([[1, 0, 1, 1.0]], 2, init=2200 + 0j)
        # regressions: bool kv/init rejected (round-2 finding)
        with pytest.raises(ValueError, match="bool"):
            fide_rating([[1, 0, 1, 1.0]], 2, kv=(True, 15.0, 30.0))
        with pytest.raises(ValueError, match="bool"):
            fide_rating([[1, 0, 1, 1.0]], 2, kv=np.array([True, True, True]))
        with pytest.raises(ValueError, match="bool"):
            fide_rating([[1, 0, 1, 1.0]], 2, init=True)
        with pytest.raises(ValueError, match="bool"):
            fide_rating([[1, 0, 1, 1.0]], 2, init=np.array(True))
        with pytest.raises(ValueError, match="bool"):
            fide_rating([[1, 0, 1, 1.0]], 2, init=np.array(True, dtype=object))

    def test_elo_reduction(self):
        import numpy as np
        from fast_mlsirm import elo_rating, fide_rating

        sched = [[1, 0, 1, 1.0], [1, 2, 3, 0.5], [2, 0, 2, 0.0]]
        f = fide_rating(sched, 4, init=1500.0, kv=(27.0, 27.0, 27.0))
        e = elo_rating(sched, 4, init=1500.0, kfac=27.0)
        assert f.ratings.tolist() == e.ratings.tolist()
        assert f.games.tolist() == e.games.tolist()
        assert f.lag.tolist() == e.lag.tolist()


class TestPredict:
    """predict_rating / predict_rating_multi (PlayerRatings predict.rating,
    R lines 1056-1133). Pins from the executed oracle
    (files/predict_oracle.py); every assert reads wrapper/crate outputs."""

    def test_elo_branch_pins(self):
        import numpy as np
        from fast_mlsirm import predict_rating

        p1 = predict_rating([2200.0, 2200.0], [20, 20], [0], [1], gamma=0.0)
        assert p1.shape == (1,)
        assert p1[0] == 0.5
        p2 = predict_rating([2200.0, 2000.0], [20, 20], [0], [1], gamma=30.0)
        assert abs(p2[0] - 0.7898441797581306) < 1e-14
        # per-game gamma array and scalar broadcast agree (crate values)
        p2b = predict_rating(
            [2200.0, 2000.0], [20, 20], [0, 0], [1, 1], gamma=np.array([30.0, 30.0])
        )
        assert p2b[0] == p2[0] and p2b[1] == p2[0]

    def test_deviation_branch(self):
        from fast_mlsirm import predict_rating

        p3 = predict_rating(
            [2200.0, 2000.0], [20, 20], [0], [1],
            deviations=[50.0, 100.0], gamma=30.0,
        )
        assert abs(p3[0] - 0.776912664201114) < 1e-14
        p3z = predict_rating(
            [2200.0, 2000.0], [20, 20], [0], [1],
            deviations=[0.0, 0.0], gamma=30.0,
        )
        p2 = predict_rating([2200.0, 2000.0], [20, 20], [0], [1], gamma=30.0)
        assert p3z[0] == p2[0]

    def test_tng_trat_unmatched(self):
        import math
        from fast_mlsirm import predict_rating

        kept = predict_rating([2200.0, 2000.0], [15, 15], [0], [1], gamma=0.0)
        assert abs(kept[0] - 0.7597469266479578) < 1e-14
        dropped = predict_rating([2200.0, 2000.0], [14, 15], [0], [1], gamma=0.0)
        assert math.isnan(dropped[0])
        replaced = predict_rating(
            [2200.0, 2000.0], [14, 15], [0], [1], gamma=0.0, trat=2000.0
        )
        assert replaced[0] == 0.5
        un = predict_rating([2200.0, 2000.0], [20, 20], [-1], [1], gamma=0.0)
        assert math.isnan(un[0])
        un_t = predict_rating(
            [2200.0, 2000.0], [20, 20], [-1], [1], gamma=0.0, trat=2000.0
        )
        assert un_t[0] == 0.5
        # stored-NaN + pair trat with deviations
        p9c = predict_rating(
            [2200.0, 2000.0], [20, 20], [0], [1],
            deviations=[float("nan"), 50.0], gamma=30.0, trat=(2200.0, 50.0),
        )
        assert abs(p9c[0] - 0.7844611342833985) < 1e-14

    def test_thresh(self):
        import math
        from fast_mlsirm import predict_rating

        p = predict_rating(
            [2200.0, 2200.0], [20, 20], [0, -1], [1, 1], gamma=0.0, thresh=0.5
        )
        assert p[0] == 1.0  # exact equality maps to 1 (>=)
        assert math.isnan(p[1])

    def test_multi_and_placing(self):
        import math
        import numpy as np
        from fast_mlsirm import predict_rating_multi

        p7 = predict_rating_multi(
            [2300.0, 2200.0, 2100.0, 2000.0], [20] * 4,
            np.array([[0, 1, 2], [0, 3, -1]]),
        )
        assert p7.shape == (2, 3)
        assert p7[0].tolist() == [2.5, 0.0, -2.5]
        assert p7[1][0] == 3.75 and p7[1][1] == -3.75 and math.isnan(p7[1][2])
        p8 = predict_rating_multi(
            [2300.0, 2300.0, 2100.0, 2000.0], [20] * 4,
            np.array([[0, 1, 2, -1]]), placing=True,
        )
        assert p8[0][0] == 1.0 and p8[0][1] == 1.0 and p8[0][2] == 3.0
        assert math.isnan(p8[0][3])

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import predict_rating, predict_rating_multi

        with pytest.raises(ValueError, match="2..=10000"):
            predict_rating([2200.0], [20], [0], [0], gamma=0.0)
        with pytest.raises(ValueError, match="complex"):
            predict_rating([2200.0 + 1j, 2000.0], [20, 20], [0], [1])
        with pytest.raises(ValueError, match="complex|bool"):
            predict_rating(np.array([True, False]), [20, 20], [0], [1])
        with pytest.raises(ValueError, match="non-numeric"):
            predict_rating([2200.0, "x"], [20, 20], [0], [1])
        with pytest.raises(ValueError, match="masked"):
            predict_rating(
                np.ma.masked_array([2200.0, 2000.0], mask=[False, True]),
                [20, 20], [0], [1],
            )
        with pytest.raises(ValueError, match="infinit"):
            predict_rating([np.inf, 2000.0], [20, 20], [0], [1])
        with pytest.raises(ValueError, match="NaN"):
            predict_rating([2200.0, 2000.0], [np.nan, 20], [0], [1])
        with pytest.raises(ValueError, match=">= -1"):
            predict_rating([2200.0, 2000.0], [20, 20], [-2], [1])
        with pytest.raises(ValueError, match="self-play"):
            predict_rating([2200.0, 2000.0], [20, 20], [1], [1])
        with pytest.raises(ValueError, match="pair"):
            predict_rating(
                [2200.0, 2000.0], [20, 20], [0], [1],
                deviations=[50.0, 50.0], trat=2000.0,
            )
        with pytest.raises(ValueError, match="bool"):
            predict_rating([2200.0, 2000.0], [20, 20], [0], [1], thresh=True)
        with pytest.raises(ValueError, match="2-D"):
            predict_rating_multi([2200.0, 2000.0], [20, 20], [0, 1])
        with pytest.raises(ValueError, match="bool"):
            predict_rating_multi(
                [2200.0, 2000.0], [20, 20], np.array([[0, 1]]), trat=True
            )
        # NaN ratings are allowed (R NA), not an error
        import math
        out = predict_rating([float("nan"), 2000.0], [20, 20], [0], [1], gamma=0.0)
        assert math.isnan(out[0])

    def test_u64_fidelity_and_overflow(self):
        import math
        import numpy as np
        import pytest
        from fast_mlsirm import predict_rating, predict_rating_multi

        big = 2**53
        # int-dtype games keep exact counts above 2**53: games == 2**53 is
        # strictly below tng == 2**53 + 1, so both players are unrated -> NaN.
        out = predict_rating(
            [2200.0, 2000.0], [big, big], [0], [1], gamma=0.0, tng=big + 1
        )
        assert math.isnan(out[0])
        outm = predict_rating_multi(
            [2300.0, 2200.0], [big, big], np.array([[0, 1]]), tng=big + 1
        )
        assert math.isnan(outm[0][0]) and math.isnan(outm[0][1])
        # float games at/above the dtype exact-integer bound are rejected
        with pytest.raises(ValueError, match='integer array'):
            predict_rating(
                [2200.0, 2000.0], np.array([float(big), 20.0]), [0], [1]
            )
        with pytest.raises(ValueError, match='integer array'):
            predict_rating_multi(
                [2200.0, 2000.0], np.array([float(big), 20.0]), np.array([[0, 1]])
            )
        # float tng at/above 2**53 rejected; huge int tng still exact
        with pytest.raises(ValueError, match='pass tng as an int'):
            predict_rating([2200.0, 2000.0], [20, 20], [0], [1], tng=float(big))
        kept = predict_rating(
            [2200.0, 2000.0], [big + 2, big + 2], [0], [1], gamma=0.0, tng=big + 1
        )
        assert abs(kept[0] - 0.7597469266479578) < 1e-14
        # core-level nr*np overflow is a checked ValueError, not a panic
        import fast_mlsirm._core as core
        with pytest.raises(ValueError, match='overflows'):
            core.predict_rating_multi(
                np.array([1.0, 2.0]),
                np.array([1, 1], dtype=np.uint64),
                np.array([], dtype=np.int64),
                1 << 61,
                8,
                15,
                None,
                False,
            )


class TestBratt:
    """bratt_mm: Bradley-Terry with additive-alpha0 ties (VGAM bratt).

    Oracle: exact-Fraction MM iterates (files/bratt_oracle.py, session
    evidence). Every assert reads values returned by the crate through the
    binding.
    """

    @staticmethod
    def _fixture():
        import numpy as np

        y = np.array([[0, 3, 1], [1, 0, 2], [2, 1, 0]], dtype=float)
        t = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], dtype=float)
        return y, t

    def test_b1_iter1_exact(self):
        import numpy as np
        from fast_mlsirm import bratt_mm

        y, t = self._fixture()
        r = bratt_mm(y, t, ref_index=0, ref_value=1.0, max_iter=100, tol=0.5)
        assert r.iterations == 1
        np.testing.assert_allclose(r.alpha, [1.0, 27.0 / 40.0, 3.0 / 4.0], rtol=1e-15)
        np.testing.assert_allclose(r.alpha0, 9.0 / 14.0, rtol=1e-15)

    def test_b2_converged(self):
        import numpy as np
        from fast_mlsirm import bratt_mm

        y, t = self._fixture()
        r = bratt_mm(y, t, tol=1e-13)
        np.testing.assert_allclose(
            r.alpha, [1.0, 0.6150318884241122, 0.686344995662376], rtol=1e-10
        )
        np.testing.assert_allclose(r.alpha0, 0.6038293879270596, rtol=1e-10)
        np.testing.assert_allclose(r.log_likelihood, -15.12765635227613, rtol=1e-12)

    def test_b4_reference_rescale(self):
        import numpy as np
        from fast_mlsirm import bratt_mm

        y, t = self._fixture()
        r = bratt_mm(y, t, tol=1e-13)
        r2 = bratt_mm(y, t, ref_index=1, ref_value=2.0, tol=1e-13)
        c = 2.0 / r.alpha[1]
        np.testing.assert_allclose(r2.alpha, r.alpha * c, rtol=1e-12)
        np.testing.assert_allclose(r2.alpha0, r.alpha0 * c, rtol=1e-12)
        np.testing.assert_allclose(r2.log_likelihood, r.log_likelihood, rtol=1e-12)

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import bratt_mm

        y, t = self._fixture()
        with pytest.raises(ValueError, match="masked"):
            bratt_mm(np.ma.masked_array(y), t)
        with pytest.raises(ValueError, match="real-valued"):
            bratt_mm(y.astype(complex), t)
        with pytest.raises(ValueError, match="numeric"):
            bratt_mm(y.astype(bool), t)
        with pytest.raises(ValueError, match="numeric"):
            bratt_mm(np.array([["a"] * 3] * 3, dtype=object), t)
        with pytest.raises(ValueError, match="square"):
            bratt_mm(y[:2], t)
        with pytest.raises(ValueError, match="same shape"):
            bratt_mm(y, np.zeros((4, 4)))
        with pytest.raises(ValueError, match="ref_index must be an integer"):
            bratt_mm(y, t, ref_index=0.5)
        with pytest.raises(ValueError, match="ref_index must be an integer"):
            bratt_mm(y, t, ref_index=True)
        with pytest.raises(ValueError, match="nonnegative"):
            bratt_mm(y, t, ref_index=-1)
        # Crate error contract surfaces through the binding:
        with pytest.raises(ValueError, match="use bradley_terry_mm"):
            bratt_mm(y, np.zeros((3, 3)))
        with pytest.raises(ValueError, match="symmetric"):
            tt = t.copy()
            tt[0, 1] = 5.0
            bratt_mm(y, tt)
        with pytest.raises(ValueError, match="no wins"):
            y0 = y.copy()
            y0[1] = 0.0
            bratt_mm(y0, t)
        with pytest.raises(ValueError, match="did not converge"):
            bratt_mm(y, t, max_iter=1, tol=1e-15)
        # Impl-review regressions:
        with pytest.raises(ValueError, match="not boolean"):
            yb = np.full((3, 3), True, dtype=object)
            np.fill_diagonal(yb, False)
            tb = np.full((3, 3), True, dtype=object)
            np.fill_diagonal(tb, False)
            bratt_mm(yb, tb)
        with pytest.raises(ValueError, match="overflow"):
            h = 9e307
            yh = np.array([[0, 1, 1], [1, 0, h], [1, h, 0]], dtype=float)
            bratt_mm(yh, t, max_iter=3, tol=2.0)


class TestFleiss:
    """fleiss_kappa binding + wrapper (irr 0.85 kappam.fleiss oracle anchors)."""

    @staticmethod
    def _fk():
        import numpy as np

        return np.array(
            [[0, 0, 0, 1], [0, 1, 1, 1], [2, 2, 2, 2], [0, 0, 2, 2], [1, 1, 1, 0]]
        )

    def test_classic_anchor(self):
        import numpy as np

        from fast_mlsirm import fleiss_kappa

        res = fleiss_kappa(self._fk())
        assert abs(res.kappa - 139.0 / 399.0) < 1e-14
        assert res.subjects_used == 5
        assert abs(res.z - 2.694739854085488) < 1e-10
        assert abs(res.p_value - 0.007044360582468963) < 5e-7
        np.testing.assert_allclose(
            res.category_kappa, [1 / 21, 31 / 91, 43 / 63], rtol=1e-14
        )

    def test_exact_and_missing(self):
        import numpy as np

        from fast_mlsirm import fleiss_kappa

        res = fleiss_kappa(self._fk(), exact=True)
        assert abs(res.kappa - 37.0 / 102.0) < 1e-14
        assert np.isnan(res.z) and np.isnan(res.p_value)
        assert res.category_kappa.size == 0
        # NaN row drops listwise and reproduces the classic anchor.
        withnan = np.vstack([[0.0, np.nan, 1.0, 2.0], self._fk().astype(float)])
        res2 = fleiss_kappa(withnan)
        assert res2.subjects_used == 5
        assert abs(res2.kappa - 139.0 / 399.0) < 1e-14
        # Explicit k adds an empty category with NaN detail.
        res3 = fleiss_kappa(self._fk(), k=4)
        assert np.isnan(res3.category_kappa[3])
        assert abs(res3.kappa - 139.0 / 399.0) < 1e-14

    def test_validation(self):
        import numpy as np
        import pytest

        from fast_mlsirm import fleiss_kappa

        fk = self._fk()
        with pytest.raises(ValueError, match="2-D"):
            fleiss_kappa(fk.reshape(-1))
        with pytest.raises(ValueError, match="masked"):
            fleiss_kappa(np.ma.masked_array(fk, mask=False))
        with pytest.raises(ValueError, match="complex|real"):
            fleiss_kappa(fk.astype(complex))
        with pytest.raises(ValueError, match="boolean"):
            fleiss_kappa(np.array([[True, False], [False, True]], dtype=object))
        with pytest.raises(ValueError, match="boolean"):
            fleiss_kappa(np.array([[True, False], [False, True]]))
        with pytest.raises(ValueError, match="integer"):
            fleiss_kappa(fk + 0.5)
        with pytest.raises(ValueError, match="infinit"):
            fleiss_kappa(np.array([[np.inf, 0.0], [1.0, 0.0]]))
        with pytest.raises(ValueError, match="raters"):
            fleiss_kappa(fk[:, :1])
        with pytest.raises(ValueError, match="infer"):
            fleiss_kappa(np.full((2, 2), np.nan))
        with pytest.raises(ValueError, match="dropped"):
            fleiss_kappa(np.full((2, 2), np.nan), k=2)
        with pytest.raises(ValueError, match="degenerate"):
            fleiss_kappa(np.ones((3, 2), dtype=int), k=2)
        # uint64 above i64::MAX must be rejected, not wrapped negative
        # (silent listwise drop). Reads the wrapper's guard, killed by
        # removing the unsigned-range check before astype(int64).
        big = np.array([[2**63, 0], [0, 1], [1, 1]], dtype=np.uint64)
        with pytest.raises(ValueError, match="int64"):
            fleiss_kappa(big, k=2)
        # Explicit k must be a true integer, not lossy-coerced.
        for bad_k in (3.9, "3", np.float64(3.0), True):
            with pytest.raises(ValueError, match="k must be an integer"):
                fleiss_kappa(fk, k=bad_k)
        # np.integer k still accepted.
        assert fleiss_kappa(fk, k=np.int64(3)).subjects_used == 5


class TestIcc:
    """icc (irr icc.R; Shrout-Fleiss taxonomy). Pins from the exact-Fraction
    oracle executed on Shrout-Fleiss (1979) Table 2. Every assert reads
    IccResult fields returned by the crate through the wrapper."""

    SF = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]

    def test_anchor_all_six(self):
        import numpy as np
        from fast_mlsirm import icc

        want = {
            ("oneway", "consistency", "single"): 448 / 2703,
            ("twoway", "consistency", "single"): 920 / 1287,
            ("twoway", "agreement", "single"): 184 / 635,
            ("oneway", "consistency", "average"): 1792 / 4047,
            ("twoway", "consistency", "average"): 3680 / 4047,
            ("twoway", "agreement", "average"): 736 / 1187,
        }
        for (m, t, u), v in want.items():
            r = icc(np.array(self.SF, dtype=float), model=m, type=t, unit=u)
            assert abs(r.value - v) < 1e-12
            assert r.subjects == 6 and r.raters == 4
        r = icc(np.array(self.SF, dtype=float), model="twoway", type="agreement", unit="single")
        assert abs(r.fvalue - 11.02724795640327) < 1e-9
        assert abs(r.df2 - 15.0) < 1e-9
        assert abs(r.p_value - 0.00013456651648433688) < 1e-9
        assert abs(r.lbound - 0.018786513374712013) < 1e-9
        assert abs(r.ubound - 0.7610843696489528) < 1e-9

    def test_r0_and_conf(self):
        import numpy as np
        from fast_mlsirm import icc

        x = np.array(self.SF, dtype=float)
        r = icc(x, model="twoway", type="agreement", unit="single", r0=0.3)
        assert abs(r.fvalue - 0.9561240676364373) < 1e-12
        assert abs(r.df2 - 4.7463353743354775) < 1e-9
        c = icc(x, model="twoway", type="consistency", unit="single", conf_level=0.80)
        assert abs(c.lbound - 0.4905340604697688) < 1e-9
        assert abs(c.ubound - 0.8966577783576283) < 1e-9

    def test_nan_listwise(self):
        import numpy as np
        from fast_mlsirm import icc

        x = np.array(self.SF, dtype=float)
        x[3, 2] = np.nan
        r = icc(x, model="twoway", type="consistency", unit="single")
        assert r.subjects == 5
        assert abs(r.value - 329 / 459) < 1e-12
        assert abs(r.fvalue - 11.123076923076923) < 1e-12

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import icc

        x = np.array(self.SF, dtype=float)
        with pytest.raises(ValueError):
            icc(x, model="threeway")
        with pytest.raises(ValueError):
            icc(x, type="absolute")
        with pytest.raises(ValueError):
            icc(x, unit="median")
        with pytest.raises(ValueError):
            icc(x, r0=1.0)
        with pytest.raises(ValueError):
            icc(x, conf_level=1.0)
        with pytest.raises(ValueError):
            icc(np.array([1.0, 2.0, 3.0]))  # 1-D
        with pytest.raises(ValueError):
            icc(np.array([[1.0, np.inf], [2.0, 3.0]]))
        with pytest.raises(ValueError):
            icc(np.ma.masked_array(x, mask=False))
        with pytest.raises(ValueError):
            icc(x.astype(complex))
        with pytest.raises(ValueError):
            icc(np.array([[True, False], [False, True]]))
        with pytest.raises(ValueError):
            icc(np.array([[True, False], [False, True]], dtype=object))
        with pytest.raises(ValueError):
            icc(np.array([[np.array(True), 2.0], [3.0, 4.0]], dtype=object))
        with pytest.raises(ValueError):
            icc(np.array([[np.array(True, dtype=object), 2.0], [3.0, 4.0]], dtype=object))
        with pytest.raises(ValueError):
            icc(
                np.array(
                    [[np.array(np.bool_(True), dtype=object), 2.0], [3.0, 4.0]], dtype=object
                )
            )
        with pytest.raises((ValueError, TypeError)):
            icc(np.array([["a", "b"], ["c", "d"]], dtype=object))
        void_bool = np.zeros((), dtype=[("f", "?")])[()]  # np.void w/ bool field
        with pytest.raises(ValueError):
            icc(np.array([[void_bool, 2.0], [3.0, 4.0]], dtype=object))
        with pytest.raises(ValueError):
            icc(x, r0=True)
        with pytest.raises(ValueError):
            icc(np.full((4, 3), 7.0))  # constant matrix -> degenerate
class TestKripp:
    """kripp_alpha (irr kripp.alpha.R). Pins are exact Fractions from the
    executed oracle; K1 matches the irr man page's published values. Every
    assert reads KrippResult fields returned by the crate via the wrapper."""

    @staticmethod
    def k1():
        import numpy as np

        nan = float("nan")
        cols = [
            [1, 1, nan, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3],
            [2, 2, 2, 2], [1, 2, 3, 4], [4, 4, 4, 4], [1, 1, 2, 1],
            [2, 2, 2, 2], [nan, 5, 5, 5], [nan, nan, 1, 1], [nan, nan, 3, nan],
        ]
        return np.array(cols, dtype=float).T  # 4 raters x 12 subjects

    def test_anchor_k1(self):
        from fast_mlsirm import kripp_alpha

        x = self.k1()
        want = {
            "nominal": 113 / 152,
            "ordinal": 108577 / 133160,
            "interval": 951 / 1120,
            "ratio": 18222619 / 22852465,
        }
        for m, v in want.items():
            r = kripp_alpha(x, method=m)
            assert abs(r.value - v) < 1e-12, m
            assert r.nmatchval == 40.0
            assert r.subjects == 12 and r.raters == 4 and r.levels == 5

    def test_no_na_quirk(self):
        import numpy as np
        from fast_mlsirm import kripp_alpha

        x = np.array([[1, 2, 3, 3, 2], [1, 2, 3, 3, 1], [2, 2, 3, 3, 2]], dtype=float)
        r = kripp_alpha(x, method="nominal")
        assert abs(r.value - 43 / 72) < 1e-12
        assert r.nmatchval == 30.0  # mc = 1 on complete data (irr quirk)

    def test_single_level(self):
        import numpy as np
        from fast_mlsirm import kripp_alpha

        r = kripp_alpha(np.full((2, 2), 2.0))
        assert r.value == 1.0 and r.levels == 1

    def test_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import kripp_alpha

        x = self.k1()
        with pytest.raises(ValueError):
            kripp_alpha(x, method="euclid")
        with pytest.raises(ValueError):
            kripp_alpha(np.array([1.0, 2.0]))  # 1-D
        with pytest.raises(ValueError):
            kripp_alpha(np.array([[1.0, np.inf], [2.0, 3.0]]))
        with pytest.raises(ValueError):
            kripp_alpha(np.full((2, 2), np.nan))  # all missing
        with pytest.raises(ValueError):
            kripp_alpha(np.array([[-1.0, 1.0], [1.0, -1.0]]), method="ratio")
        with pytest.raises(ValueError):
            kripp_alpha(np.ma.masked_array(x, mask=False))
        with pytest.raises(ValueError):
            kripp_alpha(x.astype(complex))
        with pytest.raises(ValueError):
            kripp_alpha(np.array([[True, False], [False, True]]))
        with pytest.raises(ValueError):
            kripp_alpha(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=object))

    def test_large_integer_levels_rejected(self):
        # Review finding: int64 levels beyond 2**53 collapse in the float64
        # cast (2**53 and 2**53+1 map to the same f64), silently turning
        # complete disagreement into alpha=1 with a single level. Reads the
        # wrapper's rejection; the accepted-boundary asserts read crate output.
        import numpy as np
        import pytest
        from fast_mlsirm import kripp_alpha

        big = 2**53
        x = np.array([[big, big + 1], [big + 1, big]], dtype=np.int64)
        with pytest.raises(ValueError, match="2\\*\\*53"):
            kripp_alpha(x)
        with pytest.raises(ValueError, match="2\\*\\*53"):
            kripp_alpha(np.array([[2**63, 2**63 + 2]], dtype=np.uint64).reshape(2, 1).T)
        with pytest.raises(ValueError, match="2\\*\\*53"):
            kripp_alpha(-x)  # negative side
        # Boundary: values at exactly +/-2**53 are exactly representable.
        ok = np.array([[big, -big], [-big, big]], dtype=np.int64)
        r = kripp_alpha(ok)
        assert r.levels == 2 and abs(r.value - (-0.5)) < 1e-12


class TestLight:
    """Light's kappa (irr 0.85 kappam.light + unweighted kappa2, both READ).

    Pins from the exact-Fraction oracle; every assert reads outputs of the
    crate-backed light_kappa wrapper.
    """

    def _l1(self):
        import numpy as np

        return np.array(
            [
                [1, 1, 2], [2, 2, 2], [3, 3, 3], [1, 2, 1], [2, 2, 2],
                [1, 1, 1], [3, 3, 3], [2, 2, 3], [1, 1, 1], [2, 3, 2],
            ],
            dtype=np.int64,
        )

    def test_anchor_l1(self):
        import numpy as np

        from fast_mlsirm import light_kappa

        r = light_kappa(self._l1())
        assert r.subjects_used == 10 and r.raters == 3
        assert np.allclose(
            r.kappas, [23 / 33, 23 / 33, 13 / 33], rtol=1e-15, atol=0
        )
        assert abs(r.value - 59 / 99) < 1e-15
        assert abs(r.z - 4.719794049843912) < 1e-12
        # crate erfc is a rational approximation; rel tolerance per spec.
        assert abs(r.p_value - 2.3608354551285515e-06) < 1e-12

    def test_nan_listwise_missing(self):
        import numpy as np

        from fast_mlsirm import light_kappa

        # Oracle L5 with the missing code expressed as NaN (float payload).
        m = np.array(
            [
                [1, 1, 2], [2, 2, 2], [np.nan, 1, 1], [1, 2, 1],
                [2, 2, 2], [1, 1, 1], [2, 1, 2],
            ]
        )
        r = light_kappa(m)
        assert r.subjects_used == 6
        assert abs(r.value - 1 / 3) < 1e-15
        assert abs(r.z - 0.6324555320336758) < 1e-12
        # Same data with integer -1 as the missing sentinel: identical
        # crate outputs.
        mi = np.where(np.isnan(m), -1.0, m).astype(np.int64)
        ri = light_kappa(mi)
        assert ri.value == r.value and ri.z == r.z
        assert ri.kappas.tolist() == r.kappas.tolist()

    def test_input_rejections(self):
        import numpy as np
        import pytest

        from fast_mlsirm import light_kappa

        ok = self._l1()
        with pytest.raises(ValueError):
            light_kappa(ok.astype(np.float64) + 0.5)  # 2.5 non-integer
        neg = ok.astype(np.float64)
        neg[0, 0] = -1.0
        with pytest.raises(ValueError):
            light_kappa(neg)  # negative FLOAT rejected (not missing)
        with pytest.raises(ValueError):
            light_kappa(np.ma.masked_array(ok))
        with pytest.raises(ValueError):
            light_kappa(ok.astype(object))
        with pytest.raises(ValueError):
            light_kappa(ok.astype(np.complex128))
        with pytest.raises(ValueError):
            light_kappa(ok.astype(bool))
        with pytest.raises(ValueError):
            light_kappa(ok.reshape(-1))  # 1-D
        inf = ok.astype(np.float64)
        inf[0, 0] = np.inf
        with pytest.raises(ValueError):
            light_kappa(inf)
        with pytest.raises(ValueError):
            light_kappa(np.array([[2**32 + 1, 1], [2, 2]], dtype=np.int64))
        with pytest.raises(ValueError):
            # chanceP <= 0 on valid data: disjoint rater level sets.
            light_kappa(np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int64))

    def test_two_raters_l2(self):
        import numpy as np

        from fast_mlsirm import light_kappa

        m = np.array([[1, 1], [1, 2], [2, 2], [2, 2], [1, 1], [2, 1]])
        r = light_kappa(m)
        assert r.kappas.shape == (1,)
        assert abs(r.value - 1 / 3) < 1e-15
        assert r.value == float(r.kappas[0])
        assert abs(r.z - 0.816496580927726) < 1e-12
class TestFinn:
    """Finn (1970) coefficient: anchors from the exact-Fraction oracle
    (session files/finn_oracle.py, EXECUTED; transcribed from CRAN irr
    0.85 finn.R, read in full). Every assert reads FinnResult fields
    returned by the crate."""

    FN1 = [
        [2, 3, 2, 3],
        [4, 4, 5, 4],
        [1, 2, 1, 1],
        [3, 3, 3, 4],
        [5, 4, 5, 5],
        [2, 2, 3, 2],
    ]

    def test_oneway_anchor(self):
        import numpy as np
        from fast_mlsirm import finn_coefficient

        r = finn_coefficient(np.array(self.FN1, dtype=float), s_levels=5)
        assert abs(r.value - 125.0 / 144.0) < 1e-15
        assert abs(r.statistic - 144.0 / 19.0) < 1e-13
        assert r.df2 == 18.0
        # crate chi2_sf is a rational approximation; abs tol 1e-12.
        assert abs(r.p_value - 4.47127350746514e-06) < 1e-12
        assert r.subjects == 6 and r.raters == 4

    def test_twoway_anchor_and_listwise_drop(self):
        import numpy as np
        from fast_mlsirm import finn_coefficient

        t = finn_coefficient(np.array(self.FN1, dtype=float), 5, model="twoway")
        assert abs(t.value - 617.0 / 720.0) < 1e-15
        assert abs(t.statistic - 720.0 / 103.0) < 1e-13
        assert t.df2 == 18.0  # R quirk: twoway also uses ns*(nr-1)
        assert abs(t.p_value - 8.469265956566033e-06) < 1e-12
        # NaN row -> oracle FN3 5-row pins.
        m = np.array(self.FN1, dtype=float)
        m[2, 1] = np.nan
        d = finn_coefficient(m, 5)
        assert d.subjects == 5
        assert abs(d.value - 13.0 / 15.0) < 1e-15
        assert abs(d.p_value - 2.9654977282546142e-05) < 1e-12

    def test_perfect_agreement(self):
        import numpy as np
        from fast_mlsirm import finn_coefficient

        r = finn_coefficient(np.array([[1, 1], [3, 3], [5, 5], [2, 2]], dtype=float), 5)
        assert r.value == 1.0
        assert r.statistic == float("inf")
        assert r.p_value == 0.0

    def test_input_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import finn_coefficient

        good = np.array([[1, 2], [2, 2], [3, 1]], dtype=float)
        with pytest.raises(ValueError, match="masked"):
            finn_coefficient(np.ma.masked_array(good), 3)
        with pytest.raises(ValueError, match="object"):
            finn_coefficient(np.array([[1, "a"], [2, 3]], dtype=object), 3)
        with pytest.raises(ValueError, match="real"):
            finn_coefficient(good.astype(complex), 3)
        with pytest.raises(ValueError, match="boolean"):
            finn_coefficient(good.astype(bool), 3)
        with pytest.raises(ValueError, match="2-D"):
            finn_coefficient(good.ravel(), 3)
        # bool s_levels: True is an int subclass and would become 1.
        with pytest.raises(ValueError, match="s_levels must be an integer"):
            finn_coefficient(good, True)
        with pytest.raises(ValueError, match="at least 2"):
            finn_coefficient(good, 1)
        with pytest.raises(ValueError, match="model"):
            finn_coefficient(good, 3, model="both")
        with pytest.raises(ValueError, match="infinit"):
            finn_coefficient(np.array([[1.0, np.inf], [2.0, 3.0]]), 3)
class TestMaxwell:
    """Maxwell's RE (irr maxwell.R): anchors from the executed
    exact-Fraction oracle. Every assert reads crate outputs through the
    Python wrapper."""

    def test_anchor_m2b(self):
        import numpy as np
        from fast_mlsirm import maxwell_re

        nan = float("nan")
        r = np.array(
            [
                [0, 0], [1, 1], [0, 1], [1, 1], [0, 0],
                [1, 0], [1, 0], [1, 1], [nan, 0], [0, nan],
            ],
            dtype=float,
        )
        out = maxwell_re(r)
        assert out.value == 0.25
        assert out.subjects == 8
        assert out.raters == 2

    def test_anchor_m1_and_labels(self):
        import numpy as np
        from fast_mlsirm import maxwell_re

        m1 = np.array(
            [
                [0, 0], [1, 1], [0, 1], [1, 1], [0, 0],
                [1, 0], [1, 1], [0, 0], [1, 1], [0, 1],
            ],
            dtype=float,
        )
        assert abs(maxwell_re(m1).value - 0.4) < 1e-15
        m4 = np.array(
            [[2.5, 2.5], [7.0, 7.0], [2.5, 7.0], [7.0, 7.0], [2.5, 2.5], [7.0, 2.5]]
        )
        assert abs(maxwell_re(m4).value - 1.0 / 3.0) < 1e-15

    def test_not_binary_union(self):
        import numpy as np
        import pytest
        from fast_mlsirm import maxwell_re

        # Each column binary, union of 3 levels -> rejected (E1).
        e1 = np.array([[0, 0], [1, 2], [0, 0], [1, 2]], dtype=float)
        with pytest.raises(ValueError):
            maxwell_re(e1)

    def test_input_validation(self):
        import numpy as np
        import pytest
        from fast_mlsirm import maxwell_re

        ok = np.array([[0, 0], [1, 1]], dtype=float)
        with pytest.raises(ValueError):
            maxwell_re(np.ma.masked_array(ok))
        with pytest.raises(ValueError):
            maxwell_re(ok.astype(object))
        with pytest.raises(ValueError):
            maxwell_re(ok.astype(complex))
        with pytest.raises(ValueError):
            maxwell_re(ok.astype(bool))
        with pytest.raises(ValueError):
            maxwell_re(np.array([0.0, 1.0]))  # 1-D
        with pytest.raises(ValueError):
            maxwell_re(np.array([[0, 0, 0], [1, 1, 1]], dtype=float))  # nr=3
        with pytest.raises(ValueError):
            maxwell_re(np.array([[0, 2**53 + 2], [1, 1]], dtype=np.int64))
        # Regression: np.abs overflows on int64 min, which previously let
        # -2**63 slip past the 2**53 fidelity guard (impl-review round 1).
        with pytest.raises(ValueError):
            maxwell_re(np.array([[-2**63, -2**63], [0, 0]], dtype=np.int64))
        with pytest.raises(ValueError):
            maxwell_re(np.array([[0, 2**63 - 1], [1, 1]], dtype=np.uint64))
class TestRobinson:
    """Robinson's A (irr 0.84.1 robinson.R; exact-Fraction oracle pins)."""

    def test_anchor_r1(self):
        import numpy as np

        from fast_mlsirm import robinson_a

        r = robinson_a(np.array([[1, 2, 5], [3, 3, 4], [2, 5, 5], [1, 1, 3]], dtype=float))
        # Oracle pin: A = 107/171 (exact-Fraction oracle, EXECUTED).
        assert abs(r.value - 107.0 / 171.0) < 1e-15
        assert r.subjects == 4
        assert r.raters == 3

    def test_nan_row_dropped(self):
        import numpy as np

        from fast_mlsirm import robinson_a

        x = np.array(
            [[1, 2, 5], [3, 3, 4], [2, np.nan, 4], [2, 5, 5], [1, 1, 3]], dtype=float
        )
        r = robinson_a(x)
        assert abs(r.value - 107.0 / 171.0) < 1e-15
        assert r.subjects == 4

    def test_degenerate_and_input_rejects(self):
        import numpy as np
        import pytest

        from fast_mlsirm import robinson_a

        with pytest.raises(ValueError, match="degenerate"):
            robinson_a(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float))
        with pytest.raises(ValueError):
            robinson_a(np.ma.masked_array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError):
            robinson_a(np.array([[1, 2], [3, "a"]], dtype=object))
        with pytest.raises(ValueError):
            robinson_a(np.array([[1 + 1j, 2], [3, 4]]))
        with pytest.raises(ValueError):
            robinson_a(np.array([[True, False], [False, True]]))
        with pytest.raises(ValueError):
            robinson_a(np.array([1.0, 2.0, 3.0]))

    def test_integer_dtype_accepted(self):
        import numpy as np

        from fast_mlsirm import robinson_a

        r = robinson_a(np.array([[1, 2], [2, 4], [3, 3], [4, 1], [5, 5]]))
        # Oracle pin: R4 A = 13/20.
        assert abs(r.value - 0.65) < 1e-15
        assert r.raters == 2


class TestMeancor:
    """Mean pairwise correlation (irr 0.84.1 meancor.R; oracle pins)."""

    # C1 fixture: columns are permutations of 1..4 so every pairwise r
    # is exactly rational (oracle pins EXECUTED): r = [4/5, 3/5, 0].
    C1 = [[1, 1, 2], [2, 3, 1], [3, 2, 4], [4, 4, 3]]

    def test_anchor_c1_fisher(self):
        import numpy as np

        from fast_mlsirm import mean_pairwise_cor

        r = mean_pairwise_cor(np.array(self.C1, dtype=float))
        assert abs(r.value - 0.5350920914541507) < 1e-12
        # 2e-7 tolerance: the Rust core's erfc rational approximation
        # has documented |error| < 1.2e-7 vs the oracle's math.erfc.
        assert abs(r.p_value - 0.592586178359586) < 2e-7
        assert r.dropped == 0
        assert r.subjects == 4
        assert r.raters == 3

    def test_plain_mean_and_none_shape(self):
        import numpy as np

        from fast_mlsirm import mean_pairwise_cor

        r = mean_pairwise_cor(np.array(self.C1), fisher=False)
        # Oracle pin: mean(4/5, 3/5, 0) = 7/15 exactly.
        assert abs(r.value - 7.0 / 15.0) < 1e-15
        assert r.statistic is None and r.p_value is None
        assert r.dropped == 0

    def test_perfect_pair_dropped(self):
        import numpy as np

        from fast_mlsirm import mean_pairwise_cor

        x = np.array([row + [row[0]] for row in self.C1], dtype=float)
        r = mean_pairwise_cor(x)
        assert r.dropped == 1  # the duplicated column's r == 1 pair
        assert abs(r.value - 0.6148634005909716) < 1e-12
        rp = mean_pairwise_cor(x, fisher=False)
        assert abs(rp.value - 19.0 / 30.0) < 1e-15  # perfect pair kept

    def test_rejects_and_bool_policy(self):
        import numpy as np
        import pytest

        from fast_mlsirm import mean_pairwise_cor

        x = np.array(self.C1, dtype=float)
        # fisher must be a real bool (deliberate stricter-than-legacy
        # policy; spec-verify mandatory change 4).
        with pytest.raises(TypeError):
            mean_pairwise_cor(x, fisher=1)
        with pytest.raises(TypeError):
            mean_pairwise_cor(x, fisher="false")
        assert mean_pairwise_cor(x, fisher=np.bool_(True)).p_value is not None
        with pytest.raises(ValueError):
            mean_pairwise_cor(np.ma.masked_array(x))
        # Guard-order regression: masked reject fires BEFORE the fisher
        # bool check (impl-review round-1 finding).
        with pytest.raises(ValueError):
            mean_pairwise_cor(np.ma.masked_array(x), fisher=1)
        with pytest.raises(ValueError):
            mean_pairwise_cor(np.array([[1, "a"], [2, 3]], dtype=object))
        with pytest.raises(ValueError):
            mean_pairwise_cor(x.astype(complex))
        with pytest.raises(ValueError):
            mean_pairwise_cor(np.array([[True, False], [False, True]]))
        with pytest.raises(ValueError):
            mean_pairwise_cor(np.array([1.0, 2.0]))
        # Constant column and fisher m < 4 are hard errors.
        with pytest.raises(ValueError):
            mean_pairwise_cor(np.array([[1, 5], [2, 5], [3, 5], [4, 5]], dtype=float))
        with pytest.raises(ValueError):
            mean_pairwise_cor(np.array([[1, 1], [2, 3], [3, 2]], dtype=float))


class TestMeanrho:
    """Mean pairwise Spearman rho (irr 0.84.1 meanrho.R; oracle pins)."""

    # S5 fixture: columns are permutations of 1..6 with three distinct
    # pairwise rhos (oracle pins EXECUTED): rho = [29/35, 27/35, 13/35].
    S5 = [[1, 2, 1], [2, 1, 3], [3, 4, 2], [4, 3, 6], [5, 6, 4], [6, 5, 5]]

    def test_anchor_s5_fisher(self):
        import numpy as np

        from fast_mlsirm import mean_pairwise_rho

        r = mean_pairwise_rho(np.array(self.S5, dtype=float))
        assert abs(r.value - 0.6992488340329346) < 1e-12
        assert abs(r.statistic - 1.2111345076783402) < 1e-12
        # 2e-7 tolerance: the Rust core's erfc rational approximation
        # has documented |error| < 1.2e-7 vs the oracle's math.erfc.
        assert abs(r.p_value - 0.22584385801024365) < 2e-7
        assert r.dropped == 0
        assert r.ties is False
        assert r.subjects == 6
        assert r.raters == 3

    def test_plain_mean_and_none_shape(self):
        import numpy as np

        from fast_mlsirm import mean_pairwise_rho

        r = mean_pairwise_rho(np.array(self.S5), fisher=False)
        # Oracle pin: mean(29/35, 27/35, 13/35) = 23/35 exactly.
        assert abs(r.value - 23.0 / 35.0) < 1e-15
        assert r.statistic is None and r.p_value is None
        assert r.dropped == 0

    def test_perfect_pair_dropped_and_ties(self):
        import numpy as np

        from fast_mlsirm import mean_pairwise_rho

        # S6 = S5 plus a duplicate of column 0: one rho == 1 pair.
        x = np.array([[row[0]] + row for row in self.S5], dtype=float)
        r = mean_pairwise_rho(x)
        assert r.dropped == 1
        assert abs(r.value - 0.7447132997063424) < 1e-12
        rp = mean_pairwise_rho(x, fisher=False)
        assert abs(rp.value - 16.0 / 21.0) < 1e-15  # perfect pair kept
        # Ties flag: midranks engage (S3-style column with a duplicate).
        t = mean_pairwise_rho(
            np.array([[1, 1], [1, 2], [2, 2], [3, 3], [3, 4]], dtype=float),
            fisher=False,
        )
        assert t.ties is True
        assert abs(t.value - 0.8922178162191939) < 1e-12

    def test_rejects_and_bool_policy(self):
        import numpy as np
        import pytest

        from fast_mlsirm import mean_pairwise_rho

        x = np.array(self.S5, dtype=float)
        with pytest.raises(TypeError):
            mean_pairwise_rho(x, fisher=1)
        with pytest.raises(TypeError):
            mean_pairwise_rho(x, fisher="false")
        assert mean_pairwise_rho(x, fisher=np.bool_(True)).p_value is not None
        with pytest.raises(ValueError):
            mean_pairwise_rho(np.ma.masked_array(x))
        # Guard-order regression: masked reject fires BEFORE the fisher
        # bool check (meancor impl-review round-1 precedent).
        with pytest.raises(ValueError):
            mean_pairwise_rho(np.ma.masked_array(x), fisher=1)
        with pytest.raises(ValueError):
            mean_pairwise_rho(np.array([[1, "a"], [2, 3]], dtype=object))
        with pytest.raises(ValueError):
            mean_pairwise_rho(x.astype(complex))
        with pytest.raises(ValueError):
            mean_pairwise_rho(np.array([[True, False], [False, True]]))
        with pytest.raises(ValueError):
            mean_pairwise_rho(np.array([1.0, 2.0]))
        # Constant column and fisher m < 4 are hard errors.
        with pytest.raises(ValueError):
            mean_pairwise_rho(np.array([[1, 5], [2, 5], [3, 5], [4, 5]], dtype=float))
        with pytest.raises(ValueError):
            mean_pairwise_rho(np.array([[1, 1], [2, 3], [3, 2]], dtype=float))


class TestStuartMaxwell:
    """Stuart-Maxwell MH test (irr 0.84.1 stuart.maxwell.R; oracle pins)."""

    # Oracle fixtures (files/stuartmaxwell_oracle.py, EXECUTED ALL PASS).
    M1 = [[20, 10, 5], [3, 30, 15], [2, 4, 40]]
    M2 = [[25, 5, 3, 3], [4, 30, 6, 1], [2, 3, 28, 4], [1, 2, 5, 22]]

    def test_anchor_m1(self):
        import numpy as np

        from fast_mlsirm import stuart_maxwell_mh

        r = stuart_maxwell_mh(np.array(self.M1, dtype=float))
        # Oracle pin: stat = 1520/157 exactly.
        assert abs(r.value - 1520.0 / 157.0) < 1e-13
        assert r.df == 2
        assert abs(r.p_value - 0.007901012752471986) < 1e-12
        assert r.dropped == 0
        assert r.subjects == 129
        assert r.categories == 3

    def test_drop_path_m2(self):
        import numpy as np

        from fast_mlsirm import stuart_maxwell_mh

        r = stuart_maxwell_mh(np.array(self.M2))
        assert r.dropped == 1
        assert r.df == 2
        assert r.subjects == 106
        # Oracle pin: stat = 200/171 exactly.
        assert abs(r.value - 200.0 / 171.0) < 1e-13
        assert abs(r.p_value - 0.5572199009980283) < 1e-12
        # McNemar reduction (oracle M3): (6-2)^2/8 == 2 exactly.
        m3 = stuart_maxwell_mh(np.array([[10, 6], [2, 12]], dtype=float))
        assert m3.value == 2.0
        assert m3.df == 1
        assert abs(m3.p_value - 0.1572992070502851) < 1e-12

    def test_rejects(self):
        import numpy as np
        import pytest

        from fast_mlsirm import stuart_maxwell_mh

        x = np.array(self.M1, dtype=float)
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.ma.masked_array(x))
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([[1, "a"], [2, 3]], dtype=object))
        with pytest.raises(ValueError):
            stuart_maxwell_mh(x.astype(complex))
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([[True, False], [False, True]]))
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([1.0, 2.0]))  # not 2-D
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([[10, -6], [2, 12]], dtype=float))
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([[10, 6.5], [2, 12]], dtype=float))
        # All marginals equal -> everything dropped -> error (oracle M6).
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([[5, 3], [3, 7]], dtype=float))
        # Integer fidelity guard fires before the f64 cast.
        with pytest.raises(ValueError):
            stuart_maxwell_mh(np.array([[2**60, 6], [2, 12]], dtype=np.int64))
class TestBhapkar:
    """Bhapkar marginal homogeneity (irr bhapkar.r; exact-oracle pins)."""

    def test_anchor_b1(self):
        import numpy as np

        from fast_mlsirm import bhapkar_mh

        b1 = np.array([[20, 10, 5], [3, 30, 15], [2, 4, 40]], dtype=float)
        r = bhapkar_mh(b1)
        want = 196080 / 18733
        assert abs(r.value - want) <= 1e-12 * want
        assert r.df == 2
        assert abs(r.p_value - 0.00533458022326506) <= 1e-12
        assert r.subjects == 129
        assert r.categories == 3

    def test_sm_identity_b1(self):
        # Oracle-proven exact identity bhapkar = SM / (1 - SM/n) with the
        # NO-DROP Stuart-Maxwell statistic; B1 has no equal marginals so
        # the crate stuart_maxwell_mh performs no drop (dropped == 0).
        import numpy as np

        from fast_mlsirm import bhapkar_mh, stuart_maxwell_mh

        b1 = np.array([[20, 10, 5], [3, 30, 15], [2, 4, 40]], dtype=float)
        bh = bhapkar_mh(b1)
        sm = stuart_maxwell_mh(b1)
        assert sm.dropped == 0
        want = sm.value / (1.0 - sm.value / bh.subjects)
        assert abs(bh.value - want) <= 1e-12 * want

    def test_errors(self):
        import numpy as np
        import pytest

        from fast_mlsirm import bhapkar_mh

        b1 = np.array([[20, 10, 5], [3, 30, 15], [2, 4, 40]], dtype=float)
        with pytest.raises(ValueError):
            bhapkar_mh(np.ma.masked_array(b1))
        with pytest.raises(ValueError):
            bhapkar_mh(b1.astype(object))
        with pytest.raises(ValueError):
            bhapkar_mh(b1 + 0j)
        with pytest.raises(ValueError):
            bhapkar_mh(b1.astype(bool))
        with pytest.raises(ValueError):
            bhapkar_mh(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        with pytest.raises(ValueError):
            bhapkar_mh(np.array([[10.0, -6.0], [2.0, 12.0]]))
        # Perfect agreement -> W = 0 -> singular.
        with pytest.raises(ValueError):
            bhapkar_mh(np.array([[5.0, 0.0], [0.0, 7.0]]))
        # int64 min must be rejected (np.abs-overflow guard regression).
        bad = np.array([[np.iinfo(np.int64).min, 6], [2, 12]], dtype=np.int64)
        with pytest.raises(ValueError):
            bhapkar_mh(bad)

class TestRaterBias:
    """Rater bias chi-square (irr rater.bias.R; exact-oracle pins)."""

    def test_anchor_b1(self):
        import numpy as np

        from fast_mlsirm import rater_bias

        b1 = np.array([[20, 10, 5], [3, 30, 15], [2, 4, 40]], dtype=float)
        r = rater_bias(b1)
        assert abs(r.value - 10 / 13) <= 1e-15 * (10 / 13)
        assert abs(r.statistic - 147 / 13) <= 1e-15 * (147 / 13)
        assert r.df == 1
        assert abs(r.p_value - 0.0007718664488286608) <= 1e-12
        assert r.subjects == 129

    def test_dyadic_and_balanced(self):
        import numpy as np

        from fast_mlsirm import rater_bias

        r = rater_bias(np.array([[10, 6], [2, 12]], dtype=float))
        assert r.value == 0.75 and r.statistic == 2.0 and r.subjects == 30
        r = rater_bias(np.array([[5, 3], [3, 7]], dtype=float))
        assert r.statistic == 0.0 and r.p_value == 1.0 and r.value == 0.5

    def test_errors(self):
        import numpy as np
        import pytest

        from fast_mlsirm import rater_bias

        b1 = np.array([[20, 10, 5], [3, 30, 15], [2, 4, 40]], dtype=float)
        with pytest.raises(ValueError):
            rater_bias(np.ma.masked_array(b1))
        with pytest.raises(ValueError):
            rater_bias(b1.astype(object))
        with pytest.raises(ValueError):
            rater_bias(b1 + 0j)
        with pytest.raises(ValueError):
            rater_bias(b1.astype(bool))
        with pytest.raises(ValueError):
            rater_bias(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        with pytest.raises(ValueError):
            rater_bias(np.array([[10.0, -6.0], [2.0, 12.0]]))
        # Diagonal-only table: rbb + rbc == 0.
        with pytest.raises(ValueError):
            rater_bias(np.array([[5.0, 0.0], [0.0, 7.0]]))
        # int64 min must be rejected (np.abs-overflow guard regression).
        bad = np.array([[np.iinfo(np.int64).min, 6], [2, 12]], dtype=np.int64)
        with pytest.raises(ValueError):
            rater_bias(bad)
class TestNCohenKappa:
    """Sample size for Cohen's kappa vs the executed exact-Fraction
    oracle (files/ncohen_oracle.py, session). Every assert reads
    values returned by the crate through the wrapper."""

    def test_anchor_pins(self):
        from fast_mlsirm import n_cohen_kappa

        r = n_cohen_kappa(0.3, 0.3, 0.7, 0.4)
        assert r.n == 59
        assert abs(r.q1 - 307 / 500) <= 1e-12 * (307 / 500)
        assert abs(r.q0 - 863 / 875) <= 1e-12 * (863 / 875)
        assert abs(r.pre_ceil - 58.421285144581596) <= 1e-9 * 58.421285144581596
        r2 = n_cohen_kappa(0.25, 0.3, 0.6, 0.3, twosided=True)
        assert r2.n == 87
        assert abs(r2.q0 - 34377 / 32000) <= 1e-12 * (34377 / 32000)
        r3 = n_cohen_kappa(0.5, 0.5, 0.8, 0.5, alpha=0.01, power=0.9)
        assert r3.n == 87
        assert abs(r3.q1 - 9 / 25) <= 1e-12 * (9 / 25)

    def test_error_contract(self):
        import pytest

        from fast_mlsirm import n_cohen_kappa

        with pytest.raises(ValueError):
            n_cohen_kappa(0.0, 0.3, 0.7, 0.4)
        with pytest.raises(ValueError):
            n_cohen_kappa(0.3, 0.3, 0.4, 0.4)
        with pytest.raises(ValueError):
            n_cohen_kappa(0.9, 0.1, 0.9, 0.1)  # infeasible cells
        with pytest.raises(ValueError):
            n_cohen_kappa(0.3, 0.3, 0.7, 0.4, alpha=1.0)
        with pytest.raises(ValueError):
            n_cohen_kappa(True, 0.3, 0.7, 0.4)
        with pytest.raises(ValueError):
            n_cohen_kappa("0.3", 0.3, 0.7, 0.4)  # impl-review: str parsed by float()
        with pytest.raises(ValueError):
            n_cohen_kappa(0.3, 0.3, 0.700000000000001, 0.7)  # n > 2^53
        with pytest.raises(ValueError):
            n_cohen_kappa(0.3, 0.3, 0.7, 0.4, twosided=1)

    def test_result_type(self):
        from fast_mlsirm import NCohenKappaResult, n_cohen_kappa

        r = n_cohen_kappa(0.2, 0.2, 0.5, 0.2)
        assert isinstance(r, NCohenKappaResult)
        assert isinstance(r.n, int) and r.n == 86
        assert r.pre_ceil < r.n <= r.pre_ceil + 1


def test_ksirt_analysis_alias_is_deprecated_and_matches_analyze_ksirt():
    """The renamed alias (ADR-0028): ``ksirt_analysis`` still works, warns,
    and returns the exact same result as ``analyze_ksirt``."""
    import numpy as np
    import pytest
    from fast_mlsirm import analyze_ksirt, ksirt_analysis
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "ksirt_occ"):
        pytest.skip("compiled core built without ksirt_occ")

    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    with pytest.deprecated_call():
        old = ksirt_analysis(x, nevalpoints=5)
    new = analyze_ksirt(x, nevalpoints=5)
    assert np.allclose(old.theta, new.theta)
    assert np.allclose(old.grid, new.grid)


def test_analyze_ksirt_requires_nevalpoints():
    import numpy as np
    from fast_mlsirm import analyze_ksirt

    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    with pytest.raises(TypeError):
        analyze_ksirt(x)
