"""Callback-free serving trust-boundary admission.

Serving artifacts are scientific deployment records: their structural controls,
item identities, dimension identities, and item-to-dimension identities must be
established before caller callbacks, NumPy integer narrowing, or Rust-owned
scoring can observe them. Numerical scoring arithmetic remains in the existing
Rust-backed serving implementation.
"""

from __future__ import annotations

import builtins
import math
from functools import wraps
from typing import Any, Callable

import numpy as np

_MAX_SERVING_DIMS = 64
_HARDENED_ATTR = "__fast_mlsirm_serving_export_safety_hardened__"
_NUMPY_INTEGER_TYPES = (
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
)
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def _factor_scalar(value: Any) -> int:
    """Normalize one trusted integral factor identity without caller callbacks."""
    value_type = builtins.type(value)
    if value_type is bool:
        normalized = int(value)
    elif value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
        normalized = int(value)
    elif value_type is float or any(
        value_type is scalar_type for scalar_type in _NUMPY_FLOAT_TYPES
    ):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("factor_id values must be finite integers")
        normalized = int(numeric)
    else:
        raise ValueError("factor_id must contain trusted real numeric values")

    if not 0 <= normalized < _MAX_SERVING_DIMS:
        raise ValueError(
            f"factor_id values must be integers in 0..{_MAX_SERVING_DIMS - 1}"
        )
    return normalized


def _factor_id_vector(value: Any, n_items: int) -> np.ndarray:
    """Return a callback-free contiguous signed-64 factor identity vector."""
    value_type = builtins.type(value)
    if value_type is list or value_type is tuple:
        if len(value) != n_items:
            raise ValueError("factor_id length must match the fitted item count")
        normalized = [_factor_scalar(cell) for cell in value]
        return np.ascontiguousarray(normalized, dtype=np.int64)

    if value_type is not np.ndarray:
        raise ValueError("factor_id must be a 1-D real numeric array or sequence")
    if value.shape != (n_items,):
        raise ValueError("factor_id length must match the fitted item count")
    if value.dtype.kind not in "biuf":
        raise ValueError("factor_id must contain trusted real numeric values")

    if value.dtype.kind == "f":
        if not np.all(np.isfinite(value)) or not np.all(value == np.floor(value)):
            raise ValueError("factor_id values must be finite integers")
    if not np.all(value >= 0) or not np.all(value < _MAX_SERVING_DIMS):
        raise ValueError(
            f"factor_id values must be integers in 0..{_MAX_SERVING_DIMS - 1}"
        )
    return np.ascontiguousarray(value, dtype=np.int64)


def _trusted_real_scalar(value: Any) -> bool:
    """Return whether one scalar has package-trusted real numeric identity."""
    value_type = builtins.type(value)
    return (
        value_type is bool
        or value_type is int
        or value_type is float
        or value_type is np.bool_
        or any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES)
        or any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_TYPES)
    )


def _latent_dimension(value: Any, n_items: int) -> int:
    """Return trusted zeta width without arbitrary array/container callbacks."""
    value_type = builtins.type(value)
    if value_type is np.ndarray:
        if value.ndim != 2 or value.shape[0] != n_items:
            raise ValueError("zeta must be a 2-D item-by-latent-position matrix")
        if value.dtype.kind not in "biuf":
            raise ValueError("zeta must contain trusted real numeric values")
        return int(value.shape[1])

    if value_type is not list and value_type is not tuple:
        raise ValueError("zeta must be a 2-D real numeric array or sequence")
    if len(value) != n_items:
        raise ValueError("zeta item count must match the fitted item count")

    latent_dim: int | None = None
    for row in value:
        if builtins.type(row) is not list and builtins.type(row) is not tuple:
            raise ValueError("zeta must be a rectangular 2-D real numeric matrix")
        row_width = len(row)
        if latent_dim is None:
            latent_dim = row_width
        elif row_width != latent_dim:
            raise ValueError("zeta must be a rectangular 2-D real numeric matrix")
        if not all(_trusted_real_scalar(cell) for cell in row):
            raise ValueError("zeta must contain trusted real numeric values")

    return 0 if latent_dim is None else latent_dim


def _item_code_list(value: Any, n_items: int) -> list[str]:
    """Return inert built-in item identities without caller container callbacks."""
    value_type = builtins.type(value)
    if value_type is not list and value_type is not tuple:
        raise ValueError("item_codes must be a list or tuple of built-in strings")
    if len(value) != n_items:
        raise ValueError("item_codes length must match the fitted item count")

    normalized: list[str] = []
    for code in value:
        if builtins.type(code) is not str:
            raise ValueError("item_codes must contain built-in strings")
        normalized.append(code)
    return normalized


def _dimension_name_list(value: Any, n_dims: int) -> list[str] | None:
    """Return inert built-in dimension labels without caller callbacks."""
    if value is None:
        return None
    if builtins.type(value) is not list:
        raise ValueError("dim_names must be null or a list of built-in strings")
    if len(value) != n_dims:
        raise ValueError("dim_names length must match the fitted dimension count")

    normalized: list[str] = []
    for name in value:
        if builtins.type(name) is not str:
            raise ValueError("dim_names must contain built-in strings")
        normalized.append(name)
    return normalized


def _quadrature_integer(value: Any, *, label: str) -> int:
    """Normalize one trusted supported quadrature order to a JSON-safe int."""
    value_type = builtins.type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
        normalized = int(value)
    else:
        raise ValueError(f"{label} must be an integer")
    # #1929: no node-count cap; the Rust core generates any n >= 1 rule on
    # demand (Golub & Welsch, 1969) and guards allocation overflow.
    if normalized < 1:
        raise ValueError(f"{label} must be >= 1")
    return normalized


def _validate_quadrature_resources(
    *,
    model: Any,
    n_items: int,
    n_dims: int,
    latent_dim: int,
    q_theta: int,
    q_xi: int,
) -> None:
    """Replay serving-grid allocation ceilings before compiled-core discovery."""
    if builtins.type(model) is not str:
        raise ValueError("result.model must be a built-in string")
    # #1929: no node-count cap; bound q_xi ** latent_dim with an overflow-safe
    # repeated multiply-with-division-check instead of computing the raw
    # power (a hostile q_xi could otherwise force an astronomical bigint).
    limit = 1_000_000
    n_xi = 1
    if model != "MIRT":
        for _ in range(latent_dim):
            if n_xi > limit // max(q_xi, 1):
                n_xi = limit + 1
                break
            n_xi *= q_xi
    if model != "MIRT" and n_xi > limit:
        raise ValueError("bundle q_xi ** latent_dim exceeds the serving grid limit")
    if max(n_items, n_dims) * q_theta * n_xi > 50_000_000:
        raise ValueError(
            "bundle scoring-table size (items x q_theta x n_xi) exceeds the serving limit"
        )


def _eps_distance(value: Any, *, maximum: float) -> float:
    """Normalize one trusted distance control to a JSON-safe finite float."""
    value_type = builtins.type(value)
    if value_type is bool or value_type is np.bool_:
        raise ValueError(
            f"eps_distance must be in the safe numeric range (0, {maximum}]"
        )
    if value_type is int or value_type is float:
        normalized = float(value)
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
        normalized = float(value)
    elif any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_TYPES):
        normalized = float(value)
    else:
        raise ValueError("eps_distance must be a real numeric scalar")

    if not math.isfinite(normalized) or not 0 < normalized <= maximum:
        raise ValueError(
            f"eps_distance must be in the safe numeric range (0, {maximum}]"
        )
    return normalized


def _exact_string_keys(mapping: dict[Any, Any]) -> bool:
    """Return whether an exact built-in mapping contains only inert text keys."""
    return all(builtins.type(key) is str for key in mapping)


def _is_hardened(callback: Callable[..., Any]) -> bool:
    """Return whether ``callback`` is one of this module's installed wrappers."""
    return getattr(callback, _HARDENED_ATTR, False) is True


def _unwrap_hardened(callback: Callable[..., Any]) -> Callable[..., Any]:
    """Return the package delegate captured by one installed wrapper."""
    if _is_hardened(callback):
        return callback.__wrapped__
    return callback


def install(serving_module: Any) -> None:
    """Install callback-free serving validation and export identity boundaries once."""
    current_validate: Callable[..., Any] = serving_module._validate_bundle
    current_export: Callable[..., Any] = serving_module.export_serving_bundle
    if _is_hardened(current_validate) and _is_hardened(current_export):
        return

    # Recover deterministically from a partially installed pair without stacking
    # another copy of a package-owned wrapper around the already-hardened sibling.
    original_validate = _unwrap_hardened(current_validate)
    original_export = _unwrap_hardened(current_export)

    @wraps(original_validate)
    def safe_validate_bundle(bundle: Any) -> Any:
        # Mirror the original validation order far enough to inspect
        # population.kind only after the preceding top-level contracts are
        # already known to be inert and valid. Otherwise defer untouched to
        # the original package-owned error path.
        if builtins.type(bundle) is dict and _exact_string_keys(bundle):
            schema_version = bundle.get("schema_version")
            if (
                builtins.type(schema_version) is int
                and schema_version == serving_module.SCHEMA_VERSION
            ):
                population = bundle.get("population")
                if (
                    builtins.type(population) is dict
                    and _exact_string_keys(population)
                ):
                    kind = population.get("kind")
                    if kind is not None and builtins.type(kind) is not str:
                        raise ValueError("bundle population kind must be a string")

        # Preserve every existing package-owned validation/error precedence.
        # `dim_names` is not consumed by the historical validator, so replay its
        # identity contract only after the rest of the bundle has been proven
        # valid and before any public scorer can discover the compiled core.
        result = original_validate(bundle)
        _dimension_name_list(bundle.get("dim_names"), bundle["n_dims"])
        return result

    @wraps(original_export)
    def safe_export_serving_bundle(
        result: Any,
        item_codes: Any,
        factor_id: Any,
        path: Any = None,
        q_theta: Any = 21,
        q_xi: Any = 11,
        eps_distance: Any = 1e-8,
        screening_audit: Any = None,
        dim_names: Any = None,
    ) -> Any:
        # Establish the convergence-status scalar before delegating to the
        # historical exporter: its ``str(...)`` conversion would otherwise run
        # caller-defined string-subclass callbacks on a public FitResult.
        status = getattr(result, "convergence_status", None)
        if builtins.type(status) is not str:
            raise RuntimeError(
                "export_serving_bundle requires convergence_status to be a built-in string"
            )
        if status.strip().lower() != "converged":
            return original_export(
                result,
                item_codes,
                factor_id,
                path,
                q_theta,
                q_xi,
                eps_distance,
                screening_audit,
                dim_names,
            )

        params = result.params
        n_items = len(params.b)
        item_codes_value = _item_code_list(item_codes, n_items)
        factor_value = _factor_id_vector(factor_id, n_items)
        n_dims = int(factor_value.max()) + 1
        dim_names_value = _dimension_name_list(dim_names, n_dims)
        q_theta_value = _quadrature_integer(q_theta, label="q_theta")
        q_xi_value = _quadrature_integer(q_xi, label="q_xi")
        latent_dim = _latent_dimension(params.zeta, n_items)
        _validate_quadrature_resources(
            model=result.model,
            n_items=n_items,
            n_dims=n_dims,
            latent_dim=latent_dim,
            q_theta=q_theta_value,
            q_xi=q_xi_value,
        )
        eps_distance_value = _eps_distance(
            eps_distance, maximum=serving_module.MAX_ABS_ITEM_PARAMETER
        )
        return original_export(
            result,
            item_codes_value,
            factor_value,
            path,
            q_theta_value,
            q_xi_value,
            eps_distance_value,
            screening_audit,
            dim_names_value,
        )

    setattr(safe_validate_bundle, _HARDENED_ATTR, True)
    setattr(safe_export_serving_bundle, _HARDENED_ATTR, True)
    serving_module._validate_bundle = safe_validate_bundle
    serving_module.export_serving_bundle = safe_export_serving_bundle
