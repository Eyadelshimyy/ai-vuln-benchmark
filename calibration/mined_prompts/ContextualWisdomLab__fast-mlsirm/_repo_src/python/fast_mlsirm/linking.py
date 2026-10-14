from __future__ import annotations

import numpy as np

from .types import MLSIRMParams


_IRT_LINK_METHOD_ALIASES = frozenset(
    {"meanmean", "mm", "meansigma", "ms", "haebara", "hb", "stockinglord", "sl"}
)
_NUMPY_INTEGER_SCALAR_TYPES = frozenset(
    {
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.intp,
        np.longlong,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.uintp,
        np.ulonglong,
    }
)
_TRUSTED_LINKING_REAL_SCALAR_TYPES = (
    bool,
    int,
    float,
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_TRUSTED_LINKING_COMPLEX_SCALAR_TYPES = (
    complex,
    np.complex64,
    np.complex128,
    np.clongdouble,
)


def _require_irt_link_method(method, *, name: str = "method") -> str:
    """Return a trusted IRT-link method without caller-controlled callbacks.

    The Rust parser accepts case-insensitive method names and ignores ``-`` and
    ``_`` separators. Accept only exact built-in strings before applying that
    normalization so hostile string subclasses or arbitrary objects cannot run
    representation or normalization hooks at the Python-to-Rust boundary.
    """
    if type(method) is not str:
        raise ValueError(f"{name} must be a str method identity")
    normalized = method.lower().replace("-", "").replace("_", "")
    if normalized not in _IRT_LINK_METHOD_ALIASES:
        raise ValueError(
            f"{name} must be one of mean_mean, mean_sigma, haebara, or stocking_lord"
        )
    return method


def _require_irt_link_quadrature_size(value, *, name: str = "q_theta") -> int:
    """Return a trusted quadrature-size integer without caller coercion hooks.

    Plain Python integers and genuine NumPy integer scalar classes retain the
    established public contract. Integer subclasses are rejected before
    ``int()`` so caller-defined ``__int__``, type hashing, equality, or
    representation hooks cannot run. Range/support validation remains owned by
    the established quadrature helper.
    """
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer quadrature size")


def _is_trusted_linking_real_scalar(value) -> bool:
    """Return whether ``value`` has a package-trusted inert real scalar identity."""
    value_type = type(value)
    return any(
        value_type is scalar_type for scalar_type in _TRUSTED_LINKING_REAL_SCALAR_TYPES
    )


def _is_trusted_linking_complex_scalar(value) -> bool:
    """Return whether ``value`` has a package-trusted inert complex scalar identity."""
    value_type = type(value)
    return any(
        value_type is scalar_type
        for scalar_type in _TRUSTED_LINKING_COMPLEX_SCALAR_TYPES
    )


def _real_numeric_array(value, *, name: str) -> np.ndarray:
    """Marshal trusted real numeric storage without caller conversion hooks.

    Only exact NumPy arrays or exact built-in list/tuple trees containing
    package-trusted concrete numeric scalars/arrays are materialized. This
    prevents caller-defined array, container, and numeric protocols from
    executing while linking evidence is still being admitted.
    """
    value_type = type(value)
    if value_type is np.ndarray:
        raw = value
    elif value_type is list or value_type is tuple:
        stack: list[tuple[object, bool]] = [(value, False)]
        active_container_ids: set[int] = set()
        while stack:
            current, leaving = stack.pop()
            current_type = type(current)
            if current_type is list or current_type is tuple:
                current_id = id(current)
                if leaving:
                    active_container_ids.remove(current_id)
                    continue
                if current_id in active_container_ids:
                    raise ValueError(f"{name} must be a numeric array")
                active_container_ids.add(current_id)
                stack.append((current, True))
                stack.extend(
                    (current[index], False)
                    for index in range(len(current) - 1, -1, -1)
                )
                continue
            if current_type is np.ndarray:
                if current.dtype.kind == "c":
                    raise ValueError(f"{name} must be real-valued")
                if current.dtype.kind not in ("b", "i", "u", "f"):
                    raise ValueError(f"{name} must be a numeric array")
                continue
            if _is_trusted_linking_complex_scalar(current):
                raise ValueError(f"{name} must be real-valued")
            if not _is_trusted_linking_real_scalar(current):
                raise ValueError(f"{name} must be a numeric array")
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a numeric array") from exc
    else:
        raise ValueError(f"{name} must be a numeric array")

    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    try:
        return np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc


def _real_numeric_scalar(value, *, name: str) -> float:
    """Normalize an inert real scalar without invoking caller conversion hooks."""
    if _is_trusted_linking_complex_scalar(value) or not _is_trusted_linking_real_scalar(value):
        raise ValueError(f"{name} must be a real number")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real number") from exc


def link_fixed_item_parameters(
    source: MLSIRMParams,
    target: MLSIRMParams,
    anchor_items: np.ndarray,
    factor_id: np.ndarray | None = None,
) -> tuple[MLSIRMParams, dict[str, np.ndarray]]:
    """Put source parameters on the target metric using fixed anchor items.

    Affine scale/shift estimation and parameter transformation are owned by the
    compiled Rust core (``link_fixed_item_parameters``); Python validates public
    shapes and reconstructs the parameter object plus evidence map.
    """
    source_theta = _real_numeric_array(source.theta, name="source.theta")
    source_alpha = _real_numeric_array(source.alpha, name="source.alpha")
    source_b = _real_numeric_array(source.b, name="source.b")
    target_theta = _real_numeric_array(target.theta, name="target.theta")
    target_alpha = _real_numeric_array(target.alpha, name="target.alpha")
    target_b = _real_numeric_array(target.b, name="target.b")
    for arr, nm in (
        (source_theta, "source.theta"),
        (source_alpha, "source.alpha"),
        (source_b, "source.b"),
        (target_alpha, "target.alpha"),
        (target_b, "target.b"),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{nm} must be finite")

    if source_alpha.shape != target_alpha.shape or source_b.shape != target_b.shape:
        raise ValueError("source and target item parameters must have matching shapes")
    if source_theta.ndim != 2 or target_theta.ndim != 2:
        raise ValueError("source and target theta must be 2-D (items x dimensions)")
    if source_theta.shape[1] != target_theta.shape[1]:
        raise ValueError("source and target theta must have the same dimensionality")

    n_items = source_alpha.size
    n_dims = source_theta.shape[1]

    a_fl = _real_numeric_array(anchor_items, name="anchor_items")
    if a_fl.ndim != 1 or a_fl.size == 0:
        raise ValueError("anchor_items must be a non-empty 1D array")
    if not np.all(np.isfinite(a_fl)) or np.any(a_fl < 0) or np.any(a_fl != np.floor(a_fl)):
        raise ValueError("anchor_items must be finite non-negative integers")
    # Range-check on the float BEFORE narrowing: uint64 max casts to -1 and
    # would slip past an upper-bound-only int64 check as a valid last-item index.
    if np.any(a_fl >= n_items):
        raise ValueError("anchor_items must reference existing items")
    anchors = a_fl.astype(np.int64)
    if anchors.size != np.unique(anchors).size:
        raise ValueError("anchor_items must be unique")

    if factor_id is None:
        factors = np.zeros(n_items, dtype=np.int64)
    else:
        f_fl = _real_numeric_array(factor_id, name="factor_id")
        if (
            f_fl.ndim != 1
            or not np.all(np.isfinite(f_fl))
            or np.any(f_fl < 0)
            or np.any(f_fl != np.floor(f_fl))
            or np.any(f_fl >= n_items)
        ):
            raise ValueError("factor_id must be a 1-D array of finite non-negative integers")
        factors = f_fl.astype(np.int64)
    if factors.shape != (n_items,):
        raise ValueError("factor_id length must match number of items")
    if np.any(factors >= n_dims):
        raise ValueError("factor_id values must be in 0..n_dims-1")

    # These fields are not transformed by linking, but they are part of the
    # returned MLSIRMParams record. Admit them before Rust so reconstructing the
    # linked result never invokes caller-controlled source.copy()/NumPy hooks.
    source_xi = _real_numeric_array(source.xi, name="source.xi")
    source_zeta = _real_numeric_array(source.zeta, name="source.zeta")
    source_tau = _real_numeric_scalar(source.tau, name="source.tau")

    # Affine coefficients and transformed parameters are Rust-owned.
    from . import _core as core

    res = core.link_fixed_item_parameters(
        source_theta,
        source_alpha,
        source_b,
        target_alpha,
        target_b,
        np.ascontiguousarray(anchors, dtype=np.int64),
        np.ascontiguousarray(factors, dtype=np.int64),
    )
    linked = MLSIRMParams(
        theta=np.asarray(res["theta"], dtype=np.float64),
        alpha=np.asarray(res["alpha"], dtype=np.float64),
        b=np.asarray(res["b"], dtype=np.float64),
        xi=np.array(source_xi, copy=True),
        zeta=np.array(source_zeta, copy=True),
        tau=source_tau,
    )
    scale = np.asarray(res["scale"], dtype=np.float64)
    shift = np.asarray(res["shift"], dtype=np.float64)
    return linked, {"scale": scale, "shift": shift, "anchor_items": anchors.copy()}


# --------------------------------------------------------------------------
# Characteristic-curve / moment IRT scale linking for separately-calibrated
# common-item designs (Kolen & Brennan 2014; Haebara 1980; Stocking & Lord
# 1983). Rust core is the compute path.
# --------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class IrtLinkResult:
    """IRT linking coefficients (theta_old = slope*theta_new + intercept) with
    the characteristic-curve criterion and explicit termination evidence."""

    slope: float       # theta_old = slope * theta_new + intercept
    intercept: float
    criterion: float   # characteristic-curve loss (0 for moment methods)
    n_iter: int
    method: str
    converged: bool = True
    termination_reason: str = "closed_form"
    max_iter: int = 0
    final_objective_span: float = 0.0
    objective_tolerance: float = 0.0
    final_parameter_span: float = 0.0
    parameter_tolerance: float = 0.0


def irt_link(
    a_old: np.ndarray,
    b_old: np.ndarray,
    a_new: np.ndarray,
    b_new: np.ndarray,
    method: str = "stocking_lord",
    q_theta: int = 41,
) -> IrtLinkResult:
    """Link a separately-calibrated *new* form onto the *old* (reference) scale
    from common items, returning ``theta_old = slope * theta_new + intercept``.

    ``a_*`` are slopes (``exp(alpha)`` in the engine's parameterization) and
    ``b_*`` the intercepts of the common items in the ``eta = a*theta + b``
    form. ``method`` is one of ``mean_mean``, ``mean_sigma``, ``haebara``,
    ``stocking_lord``; the characteristic-curve methods integrate over a
    standard-normal Gauss-Hermite grid of ``q_theta`` nodes. Mean/sigma
    linking requires non-zero common-item difficulty spread on both scales.
    Characteristic-curve results expose both the objective and parameter
    simplex stopping criteria; inspect ``converged`` before using a result.

    References
    ----------
    Haebara, T. (1980). Equating logistic ability scales by a weighted least
    squares method. *Japanese Psychological Research, 22*(3), 144–149.
    https://doi.org/10.4992/psycholres1954.22.144

    Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and
    linking: Methods and practices* (3rd ed.). Springer.
    https://doi.org/10.1007/978-1-4939-0317-7

    Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item
    response theory. *Applied Psychological Measurement, 7*(2), 201–210.
    https://doi.org/10.1177/014662168300700208
    """
    method = _require_irt_link_method(method)
    q_theta = _require_irt_link_quadrature_size(q_theta)

    ao = _real_numeric_array(a_old, name="a_old")
    bo = _real_numeric_array(b_old, name="b_old")
    an = _real_numeric_array(a_new, name="a_new")
    bn = _real_numeric_array(b_new, name="b_new")
    for _arr, _nm in ((ao, "a_old"), (bo, "b_old"), (an, "a_new"), (bn, "b_new")):
        if _arr.ndim != 1 or not np.all(np.isfinite(_arr)):
            raise ValueError(f"{_nm} must be a 1-D array of finite numbers")
    if ao.shape != bo.shape or an.shape != bn.shape or ao.shape != an.shape:
        raise ValueError("slope/intercept arrays must have matching lengths")
    if np.any(ao <= 0) or np.any(an <= 0):
        raise ValueError("slopes (a_old/a_new) must be positive")

    from .estimators.marginal import _gh
    from .fitstats import _core_module

    core = _core_module()
    if core is None:  # pragma: no cover
        raise RuntimeError("irt_link requires the compiled Rust core")
    nodes, weights = _gh(q_theta)
    res = core.irt_link(
        ao,
        bo,
        an,
        bn,
        np.asarray(nodes, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        method=method,
    )
    return IrtLinkResult(
        slope=float(res["slope"]), intercept=float(res["intercept"]),
        criterion=float(res["criterion"]), n_iter=int(res["n_iter"]),
        method=method,
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        max_iter=int(res["max_iter"]),
        final_objective_span=float(res["final_objective_span"]),
        objective_tolerance=float(res["objective_tolerance"]),
        final_parameter_span=float(res["final_parameter_span"]),
        parameter_tolerance=float(res["parameter_tolerance"]),
    )
