from __future__ import annotations

import warnings

import numpy as np

from .backend import (
    load_rust_core,
    normalize_backend,
    normalize_device,
    resolve_backend,
    resolve_reference_backend,
)
from .config import VALID_MODELS, FitConfig, PenaltyConfig
from .math import sigmoid, softplus
from .types import MLSIRMParams


def prepare_response(
    responses: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a binary response matrix and derive its observed-entry mask.

    Entries that are non-finite or ``-1`` are treated as missing; an optional
    ``mask`` is intersected with that rule. Observed entries must be 0/1.
    Returns ``(clean, observed)`` where missing cells in ``clean`` are zeroed.
    """
    y = np.asarray(responses, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2D matrix")
    if mask is None:
        observed = np.isfinite(y) & (y != -1)
    else:
        observed = np.asarray(mask, dtype=bool)
        if observed.shape != y.shape:
            raise ValueError("mask shape must match responses")
        observed &= np.isfinite(y) & (y != -1)

    if not np.any(observed):
        raise ValueError("responses contain no observed entries")

    invalid = (y != 0) & (y != 1)
    if np.any(observed & invalid):
        raise ValueError("observed responses must be 0 or 1")

    clean = np.where(observed, y, 0.0)
    return clean, observed


def validate_factor_id(factor_id: np.ndarray, n_items: int, n_dims: int) -> np.ndarray:
    """Validate the item-to-trait assignment and return it as ``int64``.

    ``factor_id`` must have one integer entry per item, each in
    ``0..n_dims-1`` (the simple-structure trait each item loads on).
    """
    raw = np.asarray(factor_id)
    if raw.shape != (n_items,):
        raise ValueError("factor_id length must match number of items")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("factor_id must contain integer values")
    if raw.size and (np.any(raw < 0) or int(raw.max()) >= n_dims):
        raise ValueError("factor_id values must be in 0..n_dims-1")
    return raw.astype(np.int64, copy=False)


def get_model_flags(model: str) -> tuple[bool, bool]:
    """Return ``(free_alpha, uses_space)`` capability flags for a model variant.

    ``free_alpha`` is false for the Rasch-type variants (``MLSRM``/``ULSRM``)
    that fix discriminations to 1; ``uses_space`` is false only for plain
    ``MIRT``, which has no latent-space distance term.
    """
    name = model.upper()
    if name not in VALID_MODELS:
        raise ValueError(f"model must be one of {sorted(VALID_MODELS)}")
    free_alpha = name not in {"MLSRM", "ULSRM"}
    uses_space = name != "MIRT"
    return free_alpha, uses_space


def model_flags(model: str) -> tuple[bool, bool]:
    """Deprecated alias for :func:`get_model_flags`.

    .. deprecated:: (ADR-0028) use :func:`get_model_flags` instead.
    """
    warnings.warn(
        "model_flags is deprecated, use get_model_flags instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_model_flags(model)


def compute_linear_predictor(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    *,
    model: str,
    eps_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the model linear predictor and any distance matrix.

    ``BIFAC2PLM`` uses a general-factor inner product as in Gibbons and
    Hedeker (1992). LSIRM models instead use this repository's negative
    distance interaction.

    References (APA 7th ed.):
        Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item
            bi-factor analysis. *Psychometrika, 57*(3), 423–436.
            https://doi.org/10.1007/BF02295430
    """
    name = model.upper()
    free_alpha, uses_space = get_model_flags(name)
    a = params.a if free_alpha else np.ones_like(params.alpha)
    theta_factor = params.theta[:, factor_id]

    if name == "BIFAC2PLM":
        # The bifactor's general-factor contribution is bilinear, not a
        # latent-space distance penalty (Gibbons & Hedeker, 1992).
        distance = np.zeros((params.theta.shape[0], len(factor_id)), dtype=np.float64)
        interaction = np.dot(params.xi, params.zeta.T)
    elif uses_space:
        # Optimized distance computation: replace O(N*J*D) 3D broadcast with O(N*J) 2D dot product
        dist_sq = -2.0 * np.dot(params.xi, params.zeta.T)
        dist_sq += np.einsum("ij,ij->i", params.xi, params.xi)[:, None]
        dist_sq += np.einsum("ij,ij->i", params.zeta, params.zeta)[None, :]
        np.maximum(dist_sq, 0.0, out=dist_sq)
        dist_sq += eps_distance
        distance = np.sqrt(dist_sq, out=dist_sq)
        interaction = -params.gamma * distance
    else:
        distance = np.zeros((params.theta.shape[0], len(factor_id)), dtype=np.float64)
        interaction = 0.0

    eta = a[None, :] * theta_factor + params.b[None, :] + interaction
    return eta, distance


def linear_predictor(
    params: MLSIRMParams,
    factor_id: np.ndarray,
    model: str = "MLS2PLM",
    eps_distance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated alias for :func:`compute_linear_predictor`.

    .. deprecated:: (ADR-0028) use :func:`compute_linear_predictor` instead.
    """
    warnings.warn(
        "linear_predictor is deprecated, use compute_linear_predictor instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_linear_predictor(params, factor_id, model=model, eps_distance=eps_distance)


def neg_loglik_and_grad(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    backend: str = "numpy",
    device: str | None = None,
) -> tuple[float, MLSIRMParams, float]:
    """Compute the penalized negative log-likelihood and its gradient.

    Evaluates the JMLE objective for the current ``params`` over the observed
    responses and returns ``(nll, gradients, loglik)``, where ``gradients`` is
    an :class:`MLSIRMParams` of partial derivatives and ``loglik`` is the
    unpenalized log-likelihood. Dispatches to the Rust core when ``backend``
    resolves to ``rust``; the NumPy path below is the numerically identical
    reference. ``mask``/``-1``/NaN mark missing responses.
    """
    config = config or FitConfig()
    model = config.normalized_model()
    if model == "BIFAC2PLM":
        raise ValueError("BIFAC2PLM is supported by the marginal estimator only")

    requested_backend = normalize_backend(backend)
    normalized_backend = (
        resolve_backend(requested_backend)
        if requested_backend in {"auto", "rust"}
        else resolve_reference_backend(requested_backend)
    )
    if normalized_backend == "rust":
        resolved_device = normalize_device(
            device if device is not None else config.rust_device
        )
        return _neg_loglik_and_grad_rust(
            responses, factor_id, params, config, mask, resolved_device
        )

    penalty = config.penalty
    y, observed = prepare_response(responses, mask)
    factors = validate_factor_id(factor_id, y.shape[1], params.theta.shape[1])

    if model in {"ULS2PLM", "ULSRM"} and params.theta.shape[1] != 1:
        raise ValueError(f"{model} requires one trait dimension")

    free_alpha, uses_space = get_model_flags(model)
    a = params.a if free_alpha else np.ones_like(params.alpha)
    eta, distance = compute_linear_predictor(
        params, factors, model=model, eps_distance=config.eps_distance
    )
    pi = sigmoid(eta)
    entry_loss = (softplus(eta) - y * eta) * observed
    nll = float(entry_loss.sum())
    loglik = -nll

    e = (pi - y) * observed
    grad_b = e.sum(axis=0)
    grad_alpha = np.zeros_like(params.alpha)
    if free_alpha:
        # Optimized gradient computation: avoid intermediate N x J array allocation in element-wise multiplication
        grad_alpha = (e.T @ params.theta)[np.arange(e.shape[1]), factors] * a

    # Optimized gradient computation: replace loop over dimensions with matrix multiplication
    # We embed 'a' directly into the projection matrix to avoid a JxD intermediate array allocation during multiplication
    idx = np.zeros((e.shape[1], params.theta.shape[1]), dtype=e.dtype)
    idx[np.arange(e.shape[1]), factors] = a
    grad_theta = e @ idx

    grad_xi = np.zeros_like(params.xi)
    grad_zeta = np.zeros_like(params.zeta)
    grad_tau = 0.0
    if uses_space:
        gamma = params.gamma

        # Optimized gradient computation: avoid 3D array creation, use 2D matrix multiplication instead
        e_over_d = e / distance
        sum_e_over_d = e_over_d.sum(axis=1, keepdims=True)
        grad_xi = -gamma * (params.xi * sum_e_over_d - np.dot(e_over_d, params.zeta))

        sum_e_over_d_j = e_over_d.sum(axis=0, keepdims=True).T
        grad_zeta = gamma * (
            np.dot(e_over_d.T, params.xi) - params.zeta * sum_e_over_d_j
        )

        # Optimized gradient computation: avoid intermediate array allocation by using vdot
        grad_tau = float(-gamma * np.vdot(e, distance))

    nll += _add_penalty(params, penalty, free_alpha=free_alpha, uses_space=uses_space)
    grad_theta += penalty.lambda_theta * params.theta
    grad_b += penalty.lambda_b * params.b
    if free_alpha:
        grad_alpha += penalty.lambda_alpha * (params.alpha - penalty.mu_alpha)
    if uses_space:
        grad_xi += penalty.lambda_xi * params.xi
        grad_zeta += penalty.lambda_zeta * params.zeta
        grad_tau += penalty.lambda_tau * (params.tau - penalty.mu_tau)

    grads = MLSIRMParams(
        theta=grad_theta,
        alpha=grad_alpha,
        b=grad_b,
        xi=grad_xi,
        zeta=grad_zeta,
        tau=float(grad_tau),
    )
    return float(nll), grads, loglik


def _neg_loglik_and_grad_rust(
    responses: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    config: FitConfig,
    mask: np.ndarray | None,
    device: str = "cpu",
) -> tuple[float, MLSIRMParams, float]:
    """Evaluate the penalized objective and gradient via the Rust core.

    Marshals validated arrays into ``fast_mlsirm._core.neg_loglik_and_grad`` on
    the requested ``device`` and reshapes the returned gradient blocks back into
    an :class:`MLSIRMParams`. Kept numerically identical to the NumPy path.
    """
    model = config.normalized_model()
    penalty = config.penalty
    y, observed = prepare_response(responses, mask)
    factors = validate_factor_id(factor_id, y.shape[1], params.theta.shape[1])

    if model in {"ULS2PLM", "ULSRM"} and params.theta.shape[1] != 1:
        raise ValueError(f"{model} requires one trait dimension")

    core = load_rust_core()
    objective, gradients, loglik = core.neg_loglik_and_grad(
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(observed, dtype=np.bool_),
        np.ascontiguousarray(factors, dtype=np.int64),
        np.ascontiguousarray(params.theta, dtype=np.float64),
        np.ascontiguousarray(params.alpha, dtype=np.float64),
        np.ascontiguousarray(params.b, dtype=np.float64),
        np.ascontiguousarray(params.xi, dtype=np.float64),
        np.ascontiguousarray(params.zeta, dtype=np.float64),
        float(params.tau),
        model,
        float(config.eps_distance),
        float(penalty.lambda_theta),
        float(penalty.lambda_xi),
        float(penalty.lambda_zeta),
        float(penalty.lambda_b),
        float(penalty.lambda_alpha),
        float(penalty.lambda_tau),
        float(penalty.mu_alpha),
        float(penalty.mu_tau),
        device,
    )
    grads = MLSIRMParams(
        theta=np.asarray(gradients["theta"], dtype=np.float64).reshape(
            params.theta.shape
        ),
        alpha=np.asarray(gradients["alpha"], dtype=np.float64),
        b=np.asarray(gradients["b"], dtype=np.float64),
        xi=np.asarray(gradients["xi"], dtype=np.float64).reshape(params.xi.shape),
        zeta=np.asarray(gradients["zeta"], dtype=np.float64).reshape(params.zeta.shape),
        tau=float(np.asarray(gradients["tau"], dtype=np.float64)[0]),
    )
    return float(objective), grads, float(loglik)


def _add_penalty(
    params: MLSIRMParams, penalty: PenaltyConfig, free_alpha: bool, uses_space: bool
) -> float:
    """Return the ridge-penalty contribution to the objective for ``params``.

    Sums the half-weighted squared deviations of each active parameter block
    from their respective centers (discriminations and latent-space terms are
    included only when the model variant uses them).
    """
    # Optimized penalty calculation: replace np.sum(x * x) with np.vdot(x, x) to avoid intermediate array allocation
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
