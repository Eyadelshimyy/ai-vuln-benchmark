"""Rust <-> numpy numerical parity gate.

Rust is the *primary* numeric path for fast-mlsirm; the pure-numpy
implementation in :mod:`fast_mlsirm.objective` is the reference kept for
parity testing and as a fallback when the compiled ``fast_mlsirm._core``
extension is unavailable.

These tests assert that the Rust core reproduces the reference LSIRM/MLS2PLM
neg-loglik + gradients to a tight ``1e-6`` absolute tolerance over
representative fixtures (all five model variants, several problem sizes,
with and without missing-data masks). This is the acceptance gate for the
"move numeric computation to Rust" change: the observable outputs must not
change when the Rust backend takes over.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import FitConfig, MLSIRMParams
from fast_mlsirm.objective import neg_loglik_and_grad

# Tight parity tolerance required by the port: Rust f64 must match the numpy
# reference to within 1e-6 (in practice the difference is ~1e-13). This is an
# f64 contract, so the tight tests pin rust_device="cpu": on GPU hosts the
# default "auto" routes through the f32 wgpu kernels (~7e-6 error). The f32
# GPU/auto path has a separate 1e-4 model matrix below.
PARITY_ATOL = 1e-6
GPU_ATOL = 1e-4

MODELS = ["MLS2PLM", "MLSRM", "MIRT", "ULS2PLM", "ULSRM"]

# (n_persons, n_items, n_dims, latent_dim) fixture shapes. n_dims is clamped to
# 1 for the unidimensional (ULS*) models inside the builder.
SHAPES = [
    (8, 4, 2, 2),
    (40, 12, 3, 2),
    (25, 6, 1, 1),
    (60, 15, 5, 3),
]


def _make_params(
    rng: np.random.Generator, n_persons: int, n_items: int, n_dims: int, latent_dim: int
) -> MLSIRMParams:
    return MLSIRMParams(
        theta=rng.normal(size=(n_persons, n_dims)),
        alpha=rng.normal(size=n_items) * 0.3,
        b=rng.normal(size=n_items),
        xi=rng.normal(size=(n_persons, latent_dim)),
        zeta=rng.normal(size=(n_items, latent_dim)),
        tau=0.35,
    )


def _fixture(model: str, shape: tuple[int, int, int, int], seed: int):
    n_persons, n_items, n_dims, latent_dim = shape
    if model in {"ULS2PLM", "ULSRM"}:
        n_dims = 1
    rng = np.random.default_rng(seed)
    params = _make_params(rng, n_persons, n_items, n_dims, latent_dim)
    if n_dims > 1:
        factors = (np.arange(n_items) % n_dims).astype(np.int64)
    else:
        factors = np.zeros(n_items, dtype=np.int64)
    y = (rng.random((n_persons, n_items)) < 0.5).astype(np.float64)
    return params, factors, y, n_persons, n_items


def _max_grad_diff(grad_a: MLSIRMParams, grad_b: MLSIRMParams) -> float:
    return max(
        float(np.max(np.abs(grad_a.theta - grad_b.theta))),
        float(np.max(np.abs(grad_a.alpha - grad_b.alpha))),
        float(np.max(np.abs(grad_a.b - grad_b.b))),
        float(np.max(np.abs(grad_a.xi - grad_b.xi))),
        float(np.max(np.abs(grad_a.zeta - grad_b.zeta))),
        float(abs(grad_a.tau - grad_b.tau)),
    )


@pytest.fixture(autouse=True)
def _require_rust_core():
    pytest.importorskip("fast_mlsirm._core")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("shape", SHAPES)
def test_rust_matches_numpy_dense(model: str, shape: tuple[int, int, int, int]) -> None:
    params, factors, y, _, _ = _fixture(model, shape, seed=hash((model, shape)) % 2**32)
    config = FitConfig(model=model, max_iter=1, rust_device="cpu")

    n_obj, n_grad, n_ll = neg_loglik_and_grad(y, factors, params, config, backend="numpy")
    r_obj, r_grad, r_ll = neg_loglik_and_grad(y, factors, params, config, backend="rust")

    assert abs(r_obj - n_obj) < PARITY_ATOL
    assert abs(r_ll - n_ll) < PARITY_ATOL
    assert _max_grad_diff(r_grad, n_grad) < PARITY_ATOL


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("shape", SHAPES)
def test_rust_matches_numpy_with_mask(model: str, shape: tuple[int, int, int, int]) -> None:
    params, factors, y, n_persons, n_items = _fixture(
        model, shape, seed=(hash((model, shape, "mask")) % 2**32)
    )
    rng = np.random.default_rng(7)
    mask = rng.random((n_persons, n_items)) < 0.85
    # Guarantee no all-missing row/column (rejected by prepare_response).
    mask[:, 0] = True
    mask[0, :] = True
    config = FitConfig(model=model, max_iter=1, rust_device="cpu")

    n_obj, n_grad, n_ll = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="numpy")
    r_obj, r_grad, r_ll = neg_loglik_and_grad(y, factors, params, config, mask=mask, backend="rust")

    assert abs(r_obj - n_obj) < PARITY_ATOL
    assert abs(r_ll - n_ll) < PARITY_ATOL
    assert _max_grad_diff(r_grad, n_grad) < PARITY_ATOL


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("device", ["gpu", "auto"])
@pytest.mark.parametrize("masked", [False, True])
def test_rust_gpu_or_auto_matches_numpy_model_matrix(
    model: str, device: str, masked: bool
) -> None:
    params, factors, y, n_persons, n_items = _fixture(
        model, (18, 9, 3, 2), seed=hash((model, device, masked, "gpu")) % 2**32
    )
    mask = None
    if masked:
        rng = np.random.default_rng(17)
        mask = rng.random((n_persons, n_items)) < 0.8
        mask[:, 0] = True
        mask[0, :] = True
    config = FitConfig(model=model, max_iter=1, rust_device=device)

    n_obj, n_grad, n_ll = neg_loglik_and_grad(
        y, factors, params, config, mask=mask, backend="numpy"
    )
    r_obj, r_grad, r_ll = neg_loglik_and_grad(
        y, factors, params, config, mask=mask, backend="rust"
    )

    assert abs(r_obj - n_obj) < GPU_ATOL
    assert abs(r_ll - n_ll) < GPU_ATOL
    assert _max_grad_diff(r_grad, n_grad) < GPU_ATOL


def test_rust_is_default_resolved_backend() -> None:
    """With the extension built, the default ("auto") backend resolves to Rust."""
    from fast_mlsirm.backend import resolve_backend

    assert resolve_backend(FitConfig().backend) == "rust"


# ---------------------------------------------------------------------------
# MMLE-EM parity: Rust `_core.fit_mmle_2pl` vs the NumPy reference.
#
# Both paths share the identical 41-node Gauss-Hermite table (the Rust
# constants are the shortest-roundtrip output of hermegauss(41)), but the
# NumPy reference jitters its initial `a` with seeded noise while Rust starts
# at exactly 1.0 — so the contract is agreement at the shared EM optimum
# (tight run: small problem, tol=1e-10), not bitwise identity.
# ---------------------------------------------------------------------------


def _rust_mmle():
    _core = pytest.importorskip("fast_mlsirm._core")
    fn = getattr(_core, "fit_mmle_2pl", None)
    if fn is None:
        pytest.skip("fast_mlsirm._core lacks fit_mmle_2pl")
    return fn


def _mmle_fixture(seed=0, n_persons=400, n_items=12, missing=0.25):
    rng = np.random.default_rng(seed)
    a = 0.7 + 1.3 * rng.random(n_items)
    b = -1.5 + 3.0 * rng.random(n_items)
    theta = rng.standard_normal(n_persons)
    prob = 1.0 / (1.0 + np.exp(-(a[None, :] * theta[:, None] + b[None, :])))
    y = (rng.random((n_persons, n_items)) < prob).astype(np.float64)
    observed = rng.random((n_persons, n_items)) >= missing
    observed[:, 0] = True
    observed[0, :] = True
    return np.where(observed, y, 0.0), observed


def test_mmle_rust_matches_numpy_reference() -> None:
    rust_mmle = _rust_mmle()
    from fast_mlsirm.estimators.mmle import fit_mmle_2pl as numpy_mmle

    y_filled, observed = _mmle_fixture()
    n_persons, n_items = y_filled.shape
    max_iter, tol = 2000, 1e-10

    r_a, r_b, r_theta, r_trace, r_converged = rust_mmle(
        y_filled.ravel(), observed.ravel(), n_persons, n_items, max_iter, tol
    )
    ref = numpy_mmle(y_filled, observed, n_nodes=41, max_iter=max_iter, tol=tol, seed=1)

    assert r_converged
    assert ref["status"] == "converged"
    # Measured agreement is ~1e-8; 1e-4 leaves headroom for other BLAS/platforms.
    np.testing.assert_allclose(r_a, ref["a"], atol=1e-4)
    np.testing.assert_allclose(r_b, ref["b"], atol=1e-4)
    np.testing.assert_allclose(r_theta, ref["theta"], atol=1e-4)
    assert abs(r_trace[-1] - ref["loglik_trace"][-1]) < 1e-4
    assert np.corrcoef(r_a, ref["a"])[0, 1] > 0.9999
    assert np.corrcoef(r_b, ref["b"])[0, 1] > 0.9999
    assert np.corrcoef(r_theta, ref["theta"])[0, 1] > 0.9999


def test_mmle_rust_rejects_length_mismatch() -> None:
    rust_mmle = _rust_mmle()
    with pytest.raises(ValueError):
        rust_mmle(np.zeros(5), np.ones(5, dtype=bool), 2, 3, 10, 1e-6)


def test_fit_mmle_dispatches_to_rust() -> None:
    """fit(estimator="mmle") must run on the Rust core when it is available."""
    _rust_mmle()
    from fast_mlsirm.fit import fit

    y_filled, observed = _mmle_fixture(n_persons=200, n_items=8)
    factors = np.zeros(y_filled.shape[1], dtype=np.int64)
    result = fit(
        y_filled,
        factors,
        FitConfig(model="ULS2PLM", estimator="mmle", max_iter=300),
        mask=observed,
    )
    assert result.optimizer == "mmle_em/rust"
    assert result.convergence_status == "converged"
