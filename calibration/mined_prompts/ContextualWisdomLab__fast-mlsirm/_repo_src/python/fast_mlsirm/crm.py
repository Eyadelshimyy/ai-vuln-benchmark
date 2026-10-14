"""Continuous Response Model (Samejima, 1973): item response theory for a
continuous bounded response, estimated by marginal-ML EM in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER


_MAX_CRM_RESPONSE_CELLS = 20_000_000
_MAX_CRM_RESPONSE_STRUCTURAL_NODES = 2 * _MAX_CRM_RESPONSE_CELLS
_NUMPY_INTEGER_TYPES = tuple(
    np.dtype(name).type
    for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
)
_NUMPY_FLOAT_TYPES = tuple(
    np.dtype(name).type for name in ("float16", "float32", "float64", "longdouble")
)
_NUMPY_COMPLEX_TYPES = tuple(
    np.dtype(name).type for name in ("complex64", "complex128", "clongdouble")
)
_TRUSTED_RESPONSE_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    np.bool_,
    *_NUMPY_INTEGER_TYPES,
    *_NUMPY_FLOAT_TYPES,
    *_NUMPY_COMPLEX_TYPES,
)
_TRUSTED_RESPONSE_ARRAY_KINDS = ("b", "i", "u", "f", "c")


def _is_exact_type(value_type: type, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value_type`` is one trusted scalar type without callbacks."""

    return any(value_type is trusted_type for trusted_type in trusted_types)


def _trusted_integer(value: object, name: str) -> int:
    """Normalize an exact built-in or genuine NumPy integer scalar."""

    value_type = type(value)
    if value_type is int:
        return value
    if _is_exact_type(value_type, _NUMPY_INTEGER_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _positive_real(value: object, name: str) -> float:
    """Normalize a trusted finite positive numeric scalar without subclass hooks."""

    value_type = type(value)
    trusted = (
        value_type is int
        or value_type is float
        or _is_exact_type(value_type, _NUMPY_INTEGER_TYPES)
        or _is_exact_type(value_type, _NUMPY_FLOAT_TYPES)
    )
    if not trusted:
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _trusted_response_array(responses: object) -> np.ndarray:
    """Materialize CRM evidence only after callback-free trust/resource preflight."""

    resource_error = (
        f"responses must contain at most {_MAX_CRM_RESPONSE_CELLS:,} logical cells"
    )
    structural_error = "responses exceed structural traversal budget"
    if type(responses) is np.ndarray:
        if responses.size > _MAX_CRM_RESPONSE_CELLS:
            raise ValueError(resource_error)
        return responses

    error = "responses must be a trusted NumPy array or built-in response matrix"
    if type(responses) is not list and type(responses) is not tuple:
        raise ValueError(error)

    # Every valid non-empty two-dimensional built-in matrix with N scalar cells
    # visits at most N row entries plus N scalar entries. Keep a separate budget
    # so malformed zero-cell/deep fan-out cannot evade the logical-cell ceiling.
    structural_nodes = 0

    # Each frame is [exact built-in container, next child index, logical-cell subtotal].
    # Memoized subtotals let repeated/shared acyclic subtrees count by logical
    # occurrence without re-traversing their descendants exponentially.
    frames: list[list[object]] = [[responses, 0, 0]]
    active_container_ids: set[int] = {id(responses)}
    subtree_cells: dict[int, int] = {}

    while frames:
        frame = frames[-1]
        item = frame[0]
        child_index = int(frame[1])

        if child_index >= len(item):
            subtotal = int(frame[2])
            item_id = id(item)
            active_container_ids.remove(item_id)
            subtree_cells[item_id] = subtotal
            frames.pop()
            if frames:
                parent_total = int(frames[-1][2]) + subtotal
                if parent_total > _MAX_CRM_RESPONSE_CELLS:
                    raise ValueError(resource_error)
                frames[-1][2] = parent_total
            continue

        frame[1] = child_index + 1
        child = item[child_index]
        structural_nodes += 1
        if structural_nodes > _MAX_CRM_RESPONSE_STRUCTURAL_NODES:
            raise ValueError(structural_error)
        child_type = type(child)

        if _is_exact_type(child_type, _TRUSTED_RESPONSE_SCALAR_TYPES):
            subtotal = int(frame[2]) + 1
            if subtotal > _MAX_CRM_RESPONSE_CELLS:
                raise ValueError(resource_error)
            frame[2] = subtotal
            continue

        if child_type is np.ndarray:
            if child.dtype.kind not in _TRUSTED_RESPONSE_ARRAY_KINDS:
                raise ValueError(error)
            subtotal = int(frame[2]) + int(child.size)
            if subtotal > _MAX_CRM_RESPONSE_CELLS:
                raise ValueError(resource_error)
            frame[2] = subtotal
            continue

        if child_type is not list and child_type is not tuple:
            raise ValueError(error)

        child_id = id(child)
        if child_id in active_container_ids:
            raise ValueError(error)
        cached_cells = subtree_cells.get(child_id)
        if cached_cells is not None:
            subtotal = int(frame[2]) + cached_cells
            if subtotal > _MAX_CRM_RESPONSE_CELLS:
                raise ValueError(resource_error)
            frame[2] = subtotal
            continue

        active_container_ids.add(child_id)
        frames.append([child, 0, 0])

    return np.asarray(responses)


@dataclass
class CrmFit:
    """Fitted continuous response model (Samejima, 1973).

    The logit of the response is conditionally normal and linear in the trait:
    ``logit(Z_ij) | theta_j ~ N(slope_i * theta_j + intercept_i, resid_sd_i^2)`` with
    ``theta ~ N(0, 1)``. ``slope``/``intercept``/``resid_sd`` are the working item
    parameters; ``discrimination = slope / resid_sd`` and
    ``difficulty = -intercept / slope`` are the classic Samejima ``(alpha, b)``.
    ``theta`` is the per-person EAP trait score."""

    slope: np.ndarray
    intercept: np.ndarray
    resid_sd: np.ndarray
    discrimination: np.ndarray
    difficulty: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int
    termination_reason: str = "unknown"
    final_delta: float = float("nan")
    stopping_tolerance: float = float("nan")


def fit_crm(
    responses: np.ndarray,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> CrmFit:
    """Fit the continuous response model (compute in Rust; Samejima, 1973).

    Samejima's CRM is the limit of the graded response model as the number of ordered
    categories grows without bound, for an item scored on a *continuous* bounded scale.
    Operationally (Wang & Zeng, 1998), the logit of a response ``Z in (0, 1)`` is
    conditionally normal and linear in the latent trait:
    ``logit(Z_ij) | theta_j ~ N(a_i theta_j + d_i, sigma_i^2)``, ``theta ~ N(0, 1)``.
    The item slope ``a_i``, intercept ``d_i``, and residual sd ``sigma_i`` map to the
    classic ``(discrimination alpha_i = a_i/sigma_i, difficulty b_i = -d_i/a_i,
    scale gamma_i = a_i)``. Estimated by marginal-ML EM with a Gauss-Hermite
    quadrature over the trait and a closed-form weighted-least-squares item M-step.

    ``responses`` is a persons x items array of values in the open interval ``(0, 1)``
    (values are clamped to ``[eps, 1-eps]`` before the logit transform; ``NaN`` marks a
    missing cell, dropped under a missing-at-random assumption). The trait is
    identified up to a global sign, resolved so the mean slope is non-negative.
    Convergence requires a finite, non-decreasing observed-data log-likelihood and
    a signed final increment no larger than ``tol * (1 + abs(previous_loglik))``;
    the returned fit records the termination reason and effective stopping metric
    (Dempster et al., 1977; Wu, 1983).

    References (APA 7th ed.):
        Samejima, F. (1973). Homogeneous case of the continuous response model.
            *Psychometrika, 38*(2), 203-219. https://doi.org/10.1007/BF02291114
        Wang, T., & Zeng, L. (1998). Item parameter estimation for a continuous
            response model using an EM algorithm. *Applied Psychological Measurement,
            22*(4), 333-344. https://doi.org/10.1177/014662169802200402
        Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood
            from incomplete data via the EM algorithm. *Journal of the Royal
            Statistical Society: Series B (Methodological), 39*(1), 1-22.
            https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
        Wu, C. F. J. (1983). On the convergence properties of the EM algorithm.
            *The Annals of Statistics, 11*(1), 95-103.
            https://doi.org/10.1214/aos/1176346060
    """
    from .fitstats import _core_module

    # #1929: no node-count cap; the Rust core generates any n >= 1 rule on
    # demand (Golub & Welsch, 1969) and guards allocation overflow.
    q_theta_value = _trusted_integer(q_theta, "q_theta")
    if q_theta_value < 1:
        raise ValueError("q_theta must be >= 1")
    max_iter_value = _trusted_integer(max_iter, "max_iter")
    if not 1 <= max_iter_value <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be in 1..={MAX_MAX_ITER}")
    tol_value = _positive_real(tol, "tol")

    raw = _trusted_response_array(responses)
    if np.iscomplexobj(raw) or raw.dtype == object:
        raise ValueError("responses must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a real numeric array")
    y = raw.astype(np.float64, copy=False)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    if np.any(np.isinf(y)):
        raise ValueError("responses may only use NaN for missing values")

    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.5).reshape(-1)

    core = _core_module()
    if core is None or not hasattr(core, "fit_crm"):
        raise RuntimeError("fit_crm requires the compiled Rust core")

    res = core.fit_crm(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        q_theta_value,
        max_iter_value,
        tol_value,
    )
    return CrmFit(
        slope=np.asarray(res["slope"], dtype=np.float64),
        intercept=np.asarray(res["intercept"], dtype=np.float64),
        resid_sd=np.asarray(res["resid_sd"], dtype=np.float64),
        discrimination=np.asarray(res["discrimination"], dtype=np.float64),
        difficulty=np.asarray(res["difficulty"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
        termination_reason=str(res["termination_reason"]),
        final_delta=float(res["final_delta"]),
        stopping_tolerance=float(res["stopping_tolerance"]),
    )
