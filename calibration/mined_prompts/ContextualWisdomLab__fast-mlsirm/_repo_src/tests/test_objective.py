import math

import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLSIRMParams
from fast_mlsirm.config import PenaltyConfig as _PenaltyConfig
from fast_mlsirm.objective import get_model_flags, neg_loglik_and_grad, validate_factor_id


_ZERO_PENALTY = _PenaltyConfig(
    lambda_theta=0.0,
    lambda_xi=0.0,
    lambda_zeta=0.0,
    lambda_b=0.0,
    lambda_alpha=0.0,
    lambda_tau=0.0,
    mu_alpha=0.0,
    mu_tau=0.0,
)


def _softplus(x: float) -> float:
    # Numerically stable log(1 + exp(x)), matches math.py softplus.
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _reference_neg_loglik_and_grad(y, factors, params, config):
    """Independent closed-form reference derived directly from the MLS2PLM
    canonical equations (docs/papers/mls2plm-canonical-equations.md).

    Pure-Python triple loop with a different structure than the vectorized
    production path, so it pins the objective, every gradient block, and every
    summation axis to the published formula rather than to prior code.
    """
    free_alpha, uses_space = get_model_flags(config.normalized_model())
    penalty = config.penalty
    eps = config.eps_distance

    n_persons, n_items = y.shape
    n_traits = params.theta.shape[1]
    latent_dim = params.xi.shape[1]

    a = [math.exp(params.alpha[j]) if free_alpha else 1.0 for j in range(n_items)]
    gamma = math.exp(params.tau) if uses_space else 0.0

    nll = 0.0
    g_theta = [[0.0] * n_traits for _ in range(n_persons)]
    g_alpha = [0.0] * n_items
    g_b = [0.0] * n_items
    g_xi = [[0.0] * latent_dim for _ in range(n_persons)]
    g_zeta = [[0.0] * latent_dim for _ in range(n_items)]
    g_tau = 0.0

    for p in range(n_persons):
        for j in range(n_items):
            yy = float(y[p, j])
            if yy == -1.0 or not math.isfinite(yy):  # missing sentinel
                continue
            d = int(factors[j])
            r = 0.0
            if uses_space:
                dist2 = eps
                for k in range(latent_dim):
                    diff = params.xi[p, k] - params.zeta[j, k]
                    dist2 += diff * diff
                r = math.sqrt(dist2)
            eta = a[j] * params.theta[p, d] + params.b[j] - gamma * r
            nll += _softplus(eta) - yy * eta
            e = _sigmoid(eta) - yy

            g_b[j] += e
            if free_alpha:
                g_alpha[j] += e * a[j] * params.theta[p, d]
            g_theta[p][d] += e * a[j]
            if uses_space:
                g_tau += e * (-gamma * r)
                for k in range(latent_dim):
                    common = gamma * (params.xi[p, k] - params.zeta[j, k]) / r
                    g_xi[p][k] += e * (-common)
                    g_zeta[j][k] += e * common

    # Penalty block (Molenaar & Jeon, 2026 regularized JML).
    for p in range(n_persons):
        for k in range(n_traits):
            nll += 0.5 * penalty.lambda_theta * params.theta[p, k] ** 2
            g_theta[p][k] += penalty.lambda_theta * params.theta[p, k]
    for j in range(n_items):
        nll += 0.5 * penalty.lambda_b * params.b[j] ** 2
        g_b[j] += penalty.lambda_b * params.b[j]
    if free_alpha:
        for j in range(n_items):
            delta = params.alpha[j] - penalty.mu_alpha
            nll += 0.5 * penalty.lambda_alpha * delta * delta
            g_alpha[j] += penalty.lambda_alpha * delta
    if uses_space:
        for p in range(n_persons):
            for k in range(latent_dim):
                nll += 0.5 * penalty.lambda_xi * params.xi[p, k] ** 2
                g_xi[p][k] += penalty.lambda_xi * params.xi[p, k]
        for j in range(n_items):
            for k in range(latent_dim):
                nll += 0.5 * penalty.lambda_zeta * params.zeta[j, k] ** 2
                g_zeta[j][k] += penalty.lambda_zeta * params.zeta[j, k]
        tau_delta = params.tau - penalty.mu_tau
        nll += 0.5 * penalty.lambda_tau * tau_delta * tau_delta
        g_tau += penalty.lambda_tau * tau_delta

    return nll, {
        "theta": np.asarray(g_theta),
        "alpha": np.asarray(g_alpha),
        "b": np.asarray(g_b),
        "xi": np.asarray(g_xi),
        "zeta": np.asarray(g_zeta),
        "tau": g_tau,
    }


def test_neg_loglik_matches_closed_form_single_entry():
    # Single person, single item, 2-D latent space: hand-computable pin of the
    # canonical MLS2PLM equation eta = exp(alpha)*theta + b - exp(tau)*r.
    params = MLSIRMParams(
        theta=np.array([[0.5]]),
        alpha=np.array([0.2]),
        b=np.array([0.1]),
        xi=np.array([[0.3, -0.1]]),
        zeta=np.array([[-0.2, 0.4]]),
        tau=0.1,
    )
    y = np.array([[1.0]])
    factors = np.array([0])
    config = FitConfig(max_iter=1, penalty=_ZERO_PENALTY, eps_distance=1e-8)

    a = math.exp(0.2)
    gamma = math.exp(0.1)
    r = math.sqrt((0.3 - (-0.2)) ** 2 + (-0.1 - 0.4) ** 2 + 1e-8)
    eta = a * 0.5 + 0.1 - gamma * r
    expected_nll = _softplus(eta) - 1.0 * eta
    e = _sigmoid(eta) - 1.0

    nll, grad, loglik = neg_loglik_and_grad(y, factors, params, config)

    assert math.isclose(nll, expected_nll, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(loglik, -expected_nll, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(grad.b[0]), e, abs_tol=1e-12)
    assert math.isclose(float(grad.alpha[0]), e * a * 0.5, abs_tol=1e-12)
    assert math.isclose(float(grad.theta[0, 0]), e * a, abs_tol=1e-12)
    assert math.isclose(float(grad.tau), e * (-gamma * r), abs_tol=1e-12)
    # d r / d xi = (xi - zeta)/r ; eta has -gamma*r, so grad_xi = -gamma*e*(xi-zeta)/r
    assert math.isclose(float(grad.xi[0, 0]), -gamma * e * (0.3 - (-0.2)) / r, abs_tol=1e-12)
    assert math.isclose(float(grad.xi[0, 1]), -gamma * e * (-0.1 - 0.4) / r, abs_tol=1e-12)
    assert math.isclose(float(grad.zeta[0, 0]), gamma * e * (0.3 - (-0.2)) / r, abs_tol=1e-12)
    assert math.isclose(float(grad.zeta[0, 1]), gamma * e * (-0.1 - 0.4) / r, abs_tol=1e-12)


def test_neg_loglik_and_grad_matches_independent_reference():
    # Multi-person / multi-item / multi-trait with a non-trivial factor_id and
    # the DEFAULT (non-zero) penalty. Pins the objective and every gradient
    # block/axis to a pure-Python reference derived from the canonical equation.
    params = MLSIRMParams(
        theta=np.array([[0.2, -0.3], [0.5, 0.1], [-0.4, 0.25]]),
        alpha=np.array([0.1, -0.2, 0.05, 0.3]),
        b=np.array([0.3, -0.1, 0.2, -0.25]),
        xi=np.array([[0.1, 0.2], [-0.2, 0.4], [0.35, -0.15]]),
        zeta=np.array([[0.0, -0.1], [0.3, -0.4], [-0.25, 0.2], [0.15, 0.05]]),
        tau=0.2,
    )
    y = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    factors = np.array([0, 1, 0, 1])
    config = FitConfig(max_iter=1)  # default penalty is non-zero

    nll, grad, loglik = neg_loglik_and_grad(y, factors, params, config)
    ref_nll, ref_grad = _reference_neg_loglik_and_grad(y, factors, params, config)

    assert math.isclose(nll, ref_nll, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(loglik, -(ref_nll - _reference_penalty_only(params, config)), abs_tol=1e-10)
    assert np.allclose(grad.theta, ref_grad["theta"], atol=1e-12)
    assert np.allclose(grad.alpha, ref_grad["alpha"], atol=1e-12)
    assert np.allclose(grad.b, ref_grad["b"], atol=1e-12)
    assert np.allclose(grad.xi, ref_grad["xi"], atol=1e-12)
    assert np.allclose(grad.zeta, ref_grad["zeta"], atol=1e-12)
    assert math.isclose(float(grad.tau), ref_grad["tau"], abs_tol=1e-12)


def _reference_penalty_only(params, config) -> float:
    free_alpha, uses_space = get_model_flags(config.normalized_model())
    penalty = config.penalty
    value = 0.5 * penalty.lambda_theta * float(np.vdot(params.theta, params.theta))
    value += 0.5 * penalty.lambda_b * float(np.vdot(params.b, params.b))
    if free_alpha:
        delta = params.alpha - penalty.mu_alpha
        value += 0.5 * penalty.lambda_alpha * float(np.vdot(delta, delta))
    if uses_space:
        value += 0.5 * penalty.lambda_xi * float(np.vdot(params.xi, params.xi))
        value += 0.5 * penalty.lambda_zeta * float(np.vdot(params.zeta, params.zeta))
        value += 0.5 * penalty.lambda_tau * float((params.tau - penalty.mu_tau) ** 2)
    return value


def test_missing_entries_are_excluded():
    params = MLSIRMParams(
        theta=np.array([[0.2], [-0.1]]),
        alpha=np.array([0.0, 0.1]),
        b=np.array([0.0, 0.2]),
        xi=np.array([[0.1, 0.2], [-0.2, 0.3]]),
        zeta=np.array([[0.0, 0.0], [0.2, -0.1]]),
        tau=0.0,
    )
    y = np.array([[1.0, -1.0], [0.0, 1.0]])
    full_obj, _, _ = neg_loglik_and_grad(y, np.array([0, 0]), params, FitConfig(max_iter=1))

    y2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[True, False], [True, True]])
    mask_obj, _, _ = neg_loglik_and_grad(y2, np.array([0, 0]), params, FitConfig(max_iter=1), mask=mask)
    assert np.isclose(full_obj, mask_obj)


def test_gradient_matches_finite_difference():
    params = MLSIRMParams(
        theta=np.array([[0.2], [-0.3]], dtype=float),
        alpha=np.array([0.1, -0.2], dtype=float),
        b=np.array([0.3, -0.1], dtype=float),
        xi=np.array([[0.1, 0.2], [-0.2, 0.4]], dtype=float),
        zeta=np.array([[0.0, -0.1], [0.3, -0.4]], dtype=float),
        tau=0.2,
    )
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    config = FitConfig(max_iter=1)
    base, grad, _ = neg_loglik_and_grad(y, np.array([0, 0]), params, config)

    checks = [
        ("theta", (0, 0), grad.theta[0, 0]),
        ("alpha", (0,), grad.alpha[0]),
        ("b", (1,), grad.b[1]),
        ("xi", (0, 1), grad.xi[0, 1]),
        ("zeta", (1, 0), grad.zeta[1, 0]),
    ]
    h = 1e-6
    for name, idx, analytic in checks:
        trial = params.copy()
        arr = getattr(trial, name)
        arr[idx] += h
        got, _, _ = neg_loglik_and_grad(y, np.array([0, 0]), trial, config)
        assert np.isclose((got - base) / h, analytic, atol=2e-5)

    trial = params.copy()
    trial.tau += h
    got, _, _ = neg_loglik_and_grad(y, np.array([0, 0]), trial, config)
    assert np.isclose((got - base) / h, grad.tau, atol=2e-5)


def test_rust_backend_matches_numpy_objective():
    pytest.importorskip("fast_mlsirm._core")
    params = MLSIRMParams(
        theta=np.array([[0.2], [-0.3]], dtype=float),
        alpha=np.array([0.1, -0.2], dtype=float),
        b=np.array([0.3, -0.1], dtype=float),
        xi=np.array([[0.1, 0.2], [-0.2, 0.4]], dtype=float),
        zeta=np.array([[0.0, -0.1], [0.3, -0.4]], dtype=float),
        tau=0.2,
    )
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    mask = np.array([[True, True], [True, False]])
    factors = np.array([0, 0])
    config = FitConfig(max_iter=1)

    numpy_obj, numpy_grad, numpy_loglik = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="numpy")
    rust_obj, rust_grad, rust_loglik = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="rust")

    assert np.isclose(rust_obj, numpy_obj)
    assert np.isclose(rust_loglik, numpy_loglik)
    assert np.allclose(rust_grad.theta, numpy_grad.theta)
    assert np.allclose(rust_grad.alpha, numpy_grad.alpha)
    assert np.allclose(rust_grad.b, numpy_grad.b)
    assert np.allclose(rust_grad.xi, numpy_grad.xi)
    assert np.allclose(rust_grad.zeta, numpy_grad.zeta)
    assert np.isclose(rust_grad.tau, numpy_grad.tau)


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_rust_gpgpu_device_matches_numpy(device):
    """The rust backend's device paths (including the GPGPU code) must agree
    with the numpy reference. On CI without a GPU, ``gpu``/``auto`` fall back to
    the f64 CPU implementation; on a GPU they run the f32 wgpu kernels, so the
    tolerance is chosen to cover single-precision agreement as well."""
    pytest.importorskip("fast_mlsirm._core")
    rng = np.random.default_rng(7)
    n_persons, n_items, latent_dim = 12, 6, 2
    params = MLSIRMParams(
        theta=rng.normal(size=(n_persons, 1)),
        alpha=rng.normal(scale=0.2, size=n_items),
        b=rng.normal(scale=0.3, size=n_items),
        xi=rng.normal(scale=0.5, size=(n_persons, latent_dim)),
        zeta=rng.normal(scale=0.5, size=(n_items, latent_dim)),
        tau=0.15,
    )
    y = (rng.random((n_persons, n_items)) < 0.5).astype(float)
    mask = np.ones((n_persons, n_items), dtype=bool)
    mask[0, 0] = False
    factors = np.zeros(n_items, dtype=np.int64)
    config = FitConfig(max_iter=1)

    numpy_obj, numpy_grad, numpy_loglik = neg_loglik_and_grad(
        y, factors, params, config, mask=mask, backend="numpy"
    )
    rust_obj, rust_grad, rust_loglik = neg_loglik_and_grad(
        y, factors, params, config, mask=mask, backend="rust", device=device
    )

    assert np.isclose(rust_obj, numpy_obj, rtol=1e-4, atol=1e-4)
    assert np.isclose(rust_loglik, numpy_loglik, rtol=1e-4, atol=1e-4)
    assert np.allclose(rust_grad.theta, numpy_grad.theta, rtol=1e-4, atol=1e-4)
    assert np.allclose(rust_grad.alpha, numpy_grad.alpha, rtol=1e-4, atol=1e-4)
    assert np.allclose(rust_grad.b, numpy_grad.b, rtol=1e-4, atol=1e-4)
    assert np.allclose(rust_grad.xi, numpy_grad.xi, rtol=1e-4, atol=1e-4)
    assert np.allclose(rust_grad.zeta, numpy_grad.zeta, rtol=1e-4, atol=1e-4)
    assert np.isclose(rust_grad.tau, numpy_grad.tau, rtol=1e-4, atol=1e-4)


def test_rust_core_rejects_shape_mismatch():
    pytest.importorskip("fast_mlsirm._core")
    from fast_mlsirm import _core

    with pytest.raises(ValueError, match="factor_id length must match number of items"):
        _core.neg_loglik_and_grad(
            np.zeros((2, 2), dtype=float),
            np.ones((2, 2), dtype=bool),
            np.array([0], dtype=np.int64),
            np.zeros((2, 1), dtype=float),
            np.zeros(2, dtype=float),
            np.zeros(2, dtype=float),
            np.zeros((2, 1), dtype=float),
            np.zeros((2, 1), dtype=float),
            0.0,
            "MLS2PLM",
            1e-8,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
            0.0,
            0.0,
        )


def test_validate_factor_id():
    res = validate_factor_id([0, 1, 0], n_items=3, n_dims=2)
    assert np.array_equal(res, np.array([0, 1, 0]))

    with pytest.raises(ValueError, match="factor_id length must match number of items"):
        validate_factor_id([0, 1], n_items=3, n_dims=2)

    with pytest.raises(ValueError, match="factor_id values must be in 0..n_dims-1"):
        validate_factor_id([-1, 0, 1], n_items=3, n_dims=2)

    with pytest.raises(ValueError, match="factor_id values must be in 0..n_dims-1"):
        validate_factor_id([0, 2, 0], n_items=3, n_dims=2)


import pytest
from fast_mlsirm.objective import prepare_response, _add_penalty
from fast_mlsirm.config import PenaltyConfig


def test_prepare_response_errors():
    with pytest.raises(ValueError, match="responses must be a 2D matrix"):
        prepare_response(np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="responses must be a 2D matrix"):
        prepare_response(np.array([[[1.0]]]))

    with pytest.raises(ValueError, match="mask shape must match responses"):
        prepare_response(np.array([[1.0, 0.0]]), mask=np.array([True]))

    with pytest.raises(ValueError, match="responses contain no observed entries"):
        prepare_response(np.array([[-1.0, np.nan], [np.inf, -1.0]]))

    with pytest.raises(ValueError, match="observed responses must be 0 or 1"):
        prepare_response(np.array([[2.0, 0.0], [1.0, -1.0]]))

    clean, observed = prepare_response(np.array([[1.0, -1.0], [0.0, -1.0], [-1.0, np.nan]]))
    assert observed.sum(axis=0).tolist() == [2, 0]
    assert observed.sum(axis=1).tolist() == [1, 1, 0]
    assert np.array_equal(clean[2], np.array([0.0, 0.0]))


def test_objective_check_responses_errors():
    with pytest.raises(ValueError, match="responses must be a 2D matrix"):
        prepare_response(np.array([1, 0]))

    with pytest.raises(ValueError, match="mask shape must match responses"):
        prepare_response(np.zeros((2, 2)), mask=np.zeros((3, 2)))

    with pytest.raises(ValueError, match="responses contain no observed entries"):
        prepare_response(np.full((2, 2), np.nan))

    with pytest.raises(ValueError, match="observed responses must be 0 or 1"):
        prepare_response(np.full((2, 2), 2.0))

    clean, observed = prepare_response(np.array([[np.nan, np.nan], [1, 0]]))
    assert observed.sum(axis=0).tolist() == [1, 1]
    assert observed.sum(axis=1).tolist() == [0, 2]
    assert np.array_equal(clean[0], np.array([0.0, 0.0]))


def test_objective_model_requires_one_trait():
    from fast_mlsirm.objective import neg_loglik_and_grad
    from fast_mlsirm.config import FitConfig
    params = MLSIRMParams(theta=np.zeros((2, 2)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 2)), zeta=np.zeros((2, 2)), tau=1.0)

    with pytest.raises(ValueError, match="ULS2PLM requires one trait dimension"):
        neg_loglik_and_grad(np.zeros((2, 2)), np.zeros(2, dtype=int), params, config=FitConfig(model="ULS2PLM"))
    for backend in ("numpy", "rust"):
        with pytest.raises(ValueError, match="marginal estimator only"):
            neg_loglik_and_grad(
                np.zeros((2, 2)),
                np.zeros(2, dtype=int),
                params,
                config=FitConfig(model="BIFAC2PLM", estimator="mmle"),
                backend=backend,
            )


def test_objective_add_penalty_uses_space():
    from fast_mlsirm.types import MLSIRMParams
    params = MLSIRMParams(theta=np.zeros((2, 2)), alpha=np.zeros(2), b=np.zeros(2), xi=np.zeros((2, 2)), zeta=np.zeros((2, 2)), tau=1.0)
    penalty = PenaltyConfig(
        lambda_theta=1.0, lambda_b=1.0, lambda_alpha=1.0, lambda_xi=1.0, lambda_zeta=1.0, lambda_tau=1.0,
        mu_alpha=0.0, mu_tau=0.0
    )
    val = _add_penalty(params, penalty, free_alpha=True, uses_space=True)
    assert val > 0.0


def test_get_model_flags_matches_old_model_flags_and_warns():
    from fast_mlsirm.objective import model_flags

    with pytest.deprecated_call():
        old = model_flags("MLS2PLM")
    assert old == get_model_flags("MLS2PLM")


def test_compute_linear_predictor_requires_model_and_eps_distance():
    from fast_mlsirm.objective import compute_linear_predictor

    params = MLSIRMParams(
        theta=np.zeros((1, 1)), alpha=np.zeros(1), b=np.zeros(1),
        xi=np.zeros((1, 2)), zeta=np.zeros((1, 2)), tau=0.0,
    )
    with pytest.raises(TypeError):
        compute_linear_predictor(params, np.array([0]))
    with pytest.raises(TypeError):
        compute_linear_predictor(params, np.array([0]), model="MLS2PLM")
    with pytest.raises(TypeError):
        compute_linear_predictor(params, np.array([0]), eps_distance=1e-8)


def test_linear_predictor_alias_warns_and_matches_new_name():
    from fast_mlsirm.objective import compute_linear_predictor, linear_predictor

    params = MLSIRMParams(
        theta=np.array([[0.5]]), alpha=np.array([0.2]), b=np.array([0.1]),
        xi=np.array([[0.3, -0.1]]), zeta=np.array([[-0.2, 0.4]]), tau=0.1,
    )
    with pytest.deprecated_call():
        old_eta, old_dist = linear_predictor(params, np.array([0]))
    new_eta, new_dist = compute_linear_predictor(
        params, np.array([0]), model="MLS2PLM", eps_distance=1e-8
    )
    assert np.allclose(old_eta, new_eta)
    assert np.allclose(old_dist, new_dist)
