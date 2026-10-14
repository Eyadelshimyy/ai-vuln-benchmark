"""Trusted scalar boundary for paired-comparison scaling controls.

This module validates and normalizes caller-owned semantic controls before the
historical :mod:`fast_mlsirm.scaling` wrappers materialize data or discover
the compiled core. Bradley-Terry estimation and every result-affecting
calculation remain owned by the Rust implementation.
"""

from __future__ import annotations

from functools import wraps
import math
from types import ModuleType
from typing import Any, Callable

import numpy as np

from .config import MAX_MAX_ITER

_NUMPY_INTEGER_SCALAR_TYPES = (
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
_NUMPY_FLOATING_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _exact_integer(value: object, name: str) -> int:
    """Return one trusted integer without invoking caller conversion hooks."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _exact_real(value: object, name: str) -> float:
    """Return one trusted real scalar without invoking caller conversion hooks."""
    value_type = type(value)
    try:
        if value_type is int or value_type is float:
            return float(value)
        if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            return float(value)
        if any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES):
            return float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    raise ValueError(f"{name} must be a real number")


def _normalized_controls(alpha: object, max_iter: object, tol: object) -> tuple[float, int, float]:
    """Normalize and domain-check Bradley-Terry controls before side effects."""
    normalized_alpha = _exact_real(alpha, "alpha")
    normalized_max_iter = _exact_integer(max_iter, "max_iter")
    normalized_tol = _exact_real(tol, "tol")

    if not math.isfinite(normalized_alpha) or normalized_alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not 1 <= normalized_max_iter <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be between 1 and {MAX_MAX_ITER}")
    if not math.isfinite(normalized_tol) or normalized_tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    return normalized_alpha, normalized_max_iter, normalized_tol


def _normalized_bratt_controls(
    ref_index: object,
    ref_value: object,
    max_iter: object,
    tol: object,
) -> tuple[int, float, int, float]:
    """Normalize BRATT controls before comparison data or native discovery."""
    normalized_ref_index = _exact_integer(ref_index, "ref_index")
    normalized_ref_value = _exact_real(ref_value, "ref_value")
    normalized_max_iter = _exact_integer(max_iter, "max_iter")
    normalized_tol = _exact_real(tol, "tol")

    if normalized_ref_index < 0:
        raise ValueError("ref_index must be nonnegative")
    if not math.isfinite(normalized_ref_value) or normalized_ref_value <= 0.0:
        raise ValueError("ref_value must be finite and positive")
    if not 1 <= normalized_max_iter <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be between 1 and {MAX_MAX_ITER}")
    if not math.isfinite(normalized_tol) or normalized_tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    return (
        normalized_ref_index,
        normalized_ref_value,
        normalized_max_iter,
        normalized_tol,
    )


def install(scaling_module: ModuleType) -> None:
    """Install paired-comparison control wrappers once on ``scaling_module``."""
    bradley_original: Callable[..., Any] = scaling_module.bradley_terry_mm
    if not getattr(bradley_original, "__fast_mlsirm_control_hardened__", False):

        @wraps(bradley_original)
        def hardened_bradley_terry_mm(
            wins: object,
            alpha: object = 0.0,
            max_iter: object = 10_000,
            tol: object = 1e-8,
        ) -> Any:
            """Validate trusted controls, then delegate unchanged arithmetic to Rust."""
            normalized_alpha, normalized_max_iter, normalized_tol = _normalized_controls(
                alpha, max_iter, tol
            )
            return bradley_original(
                wins,
                alpha=normalized_alpha,
                max_iter=normalized_max_iter,
                tol=normalized_tol,
            )

        hardened_bradley_terry_mm.__fast_mlsirm_control_hardened__ = True
        scaling_module.bradley_terry_mm = hardened_bradley_terry_mm

    bratt_original: Callable[..., Any] = scaling_module.bratt_mm
    if not getattr(bratt_original, "__fast_mlsirm_control_hardened__", False):

        @wraps(bratt_original)
        def hardened_bratt_mm(
            wins: object,
            ties: object,
            ref_index: object = 0,
            ref_value: object = 1.0,
            max_iter: object = 10_000,
            tol: object = 1e-10,
        ) -> Any:
            """Validate BRATT controls, then delegate unchanged arithmetic to Rust."""
            (
                normalized_ref_index,
                normalized_ref_value,
                normalized_max_iter,
                normalized_tol,
            ) = _normalized_bratt_controls(ref_index, ref_value, max_iter, tol)
            return bratt_original(
                wins,
                ties,
                ref_index=normalized_ref_index,
                ref_value=normalized_ref_value,
                max_iter=normalized_max_iter,
                tol=normalized_tol,
            )

        hardened_bratt_mm.__fast_mlsirm_control_hardened__ = True
        scaling_module.bratt_mm = hardened_bratt_mm
