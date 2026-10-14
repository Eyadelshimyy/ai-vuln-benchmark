"""Callback-free semantic-control validation for observed-score DIF adapters.

The logistic-regression, Mantel-Haenszel, and purification statistics remain
Rust-owned. This module only establishes trusted scalar identities and native
parameter ranges before legacy adapters can discover the compiled core or
materialize caller-owned response data.
"""

from __future__ import annotations

import builtins
import math
from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

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
_USIZE_MAX = int(np.iinfo(np.uintp).max)
_DIF_HARDENED_ATTR = "__fast_mlsirm_dif_control_hardened__"


def _exact_bool(value: Any, name: str) -> bool:
    """Normalize a concrete Python/NumPy bool without caller callbacks."""
    value_type = builtins.type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a bool")


def _real(value: Any, name: str) -> float:
    """Normalize a trusted real scalar without arbitrary conversion hooks."""
    value_type = builtins.type(value)
    try:
        if value_type is int or value_type is float:
            normalized = float(value)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            normalized = float(value)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES):
            normalized = float(value)
        else:
            raise ValueError(f"{name} must be a real number")
    except OverflowError as exc:
        raise ValueError(f"{name} must be a real number") from exc
    return normalized


def _usize(value: Any, name: str, *, minimum: int) -> int:
    """Normalize one trusted integer into the native Rust ``usize`` domain."""
    value_type = builtins.type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if normalized < minimum:
        comparator = ">= 1" if minimum == 1 else ">= 0"
        raise ValueError(f"{name} must be {comparator}")
    if normalized > _USIZE_MAX:
        raise ValueError(f"{name} exceeds the native usize range")
    return normalized


def _common_controls(exclude_studied_item: Any, fdr_q: Any) -> tuple[bool, float]:
    """Normalize matching-score and FDR controls before data/native access."""
    exclude = _exact_bool(exclude_studied_item, "exclude_studied_item")
    q = _real(fdr_q, "fdr_q")
    if not math.isfinite(q) or not 0.0 < q <= 1.0:
        raise ValueError("fdr_q must be finite and in (0, 1]")
    return exclude, q


def _original(function: Callable[..., Any]) -> Callable[..., Any]:
    """Return the package-owned original behind a hardened wrapper."""
    if getattr(function, _DIF_HARDENED_ATTR, False):
        return function.__wrapped__
    return function


def install(dif_module: ModuleType) -> None:
    """Install callback-free wrappers on the observed-score DIF module."""
    current = (
        dif_module.detect_dif_logistic,
        dif_module.detect_dif_mantel_haenszel_purified,
        dif_module.detect_dif_logistic_purified,
    )
    if all(getattr(function, _DIF_HARDENED_ATTR, False) for function in current):
        return

    original_logistic = _original(current[0])
    original_mh_purified = _original(current[1])
    original_logistic_purified = _original(current[2])

    @wraps(original_logistic)
    def safe_detect_dif_logistic(
        responses: Any,
        group: Any,
        *,
        exclude_studied_item: bool = False,
        fdr_q: float,
        max_iter: int,
    ) -> Any:
        """Validate logistic DIF controls before data or native discovery."""
        exclude, q = _common_controls(exclude_studied_item, fdr_q)
        iterations = _usize(max_iter, "max_iter", minimum=1)
        return original_logistic(
            responses,
            group,
            exclude_studied_item=exclude,
            fdr_q=q,
            max_iter=iterations,
        )

    @wraps(original_mh_purified)
    def safe_detect_dif_mantel_haenszel_purified(
        responses: Any,
        group: Any,
        *,
        exclude_studied_item: bool = False,
        fdr_q: float,
        max_rounds: int,
        min_anchor_items: int,
    ) -> Any:
        """Validate purified MH controls before data or native discovery."""
        exclude, q = _common_controls(exclude_studied_item, fdr_q)
        rounds = _usize(max_rounds, "max_rounds", minimum=1)
        anchors = _usize(min_anchor_items, "min_anchor_items", minimum=0)
        return original_mh_purified(
            responses,
            group,
            exclude_studied_item=exclude,
            fdr_q=q,
            max_rounds=rounds,
            min_anchor_items=anchors,
        )

    @wraps(original_logistic_purified)
    def safe_detect_dif_logistic_purified(
        responses: Any,
        group: Any,
        *,
        exclude_studied_item: bool = False,
        fdr_q: float,
        max_iter: int,
        max_rounds: int,
        min_anchor_items: int,
    ) -> Any:
        """Validate purified logistic controls before data or native discovery."""
        exclude, q = _common_controls(exclude_studied_item, fdr_q)
        iterations = _usize(max_iter, "max_iter", minimum=1)
        rounds = _usize(max_rounds, "max_rounds", minimum=1)
        anchors = _usize(min_anchor_items, "min_anchor_items", minimum=0)
        return original_logistic_purified(
            responses,
            group,
            exclude_studied_item=exclude,
            fdr_q=q,
            max_iter=iterations,
            max_rounds=rounds,
            min_anchor_items=anchors,
        )

    wrappers = (
        safe_detect_dif_logistic,
        safe_detect_dif_mantel_haenszel_purified,
        safe_detect_dif_logistic_purified,
    )
    for function in wrappers:
        setattr(function, _DIF_HARDENED_ATTR, True)
    (
        dif_module.detect_dif_logistic,
        dif_module.detect_dif_mantel_haenszel_purified,
        dif_module.detect_dif_logistic_purified,
    ) = wrappers
