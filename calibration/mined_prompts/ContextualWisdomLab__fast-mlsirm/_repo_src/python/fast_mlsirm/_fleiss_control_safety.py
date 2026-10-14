"""Callback-safe control preflight for the public Fleiss-kappa adapter."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

MAX_FLEISS_CATEGORIES = 10_000


def _trusted_category_count(module: ModuleType, value: object | None) -> int | None:
    """Normalize an explicit category count without caller-controlled coercion."""
    if value is None:
        return None
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif module._is_exact_numpy_integer_scalar_type(value_type):
        normalized = int(value)
    else:
        raise ValueError("k must be an integer")
    if normalized < 2:
        raise ValueError("k must be >= 2")
    if normalized > MAX_FLEISS_CATEGORIES:
        raise ValueError(f"k must be <= {MAX_FLEISS_CATEGORIES}")
    return normalized


def _trusted_exact(value: object) -> bool:
    """Normalize exact-mode selection without invoking caller truthiness hooks."""
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise ValueError("exact must be a boolean")


def install(module: ModuleType) -> None:
    """Install a callback-safe Fleiss-kappa marshalling boundary once."""
    current = module.fleiss_kappa
    if getattr(current, "__fast_mlsirm_fleiss_control_hardened__", False):
        return

    @wraps(current)
    def safe_fleiss_kappa(
        ratings: np.ndarray,
        k: int | None = None,
        exact: bool = False,
    ) -> Any:
        """Validate Fleiss controls and ratings before native-core discovery."""
        category_count = _trusted_category_count(module, k)
        exact_flag = _trusted_exact(exact)

        if isinstance(ratings, np.ma.MaskedArray):
            raise ValueError("masked arrays are not supported; use NaN for missing")
        arr = np.asarray(ratings)
        if arr.ndim != 2:
            raise ValueError("ratings must be a 2-D (subjects x raters) array")
        if np.iscomplexobj(arr):
            raise ValueError("ratings must be real-valued")
        if arr.dtype == object:
            for value in arr.flat:
                if value is None or isinstance(value, (bool, np.bool_, str, bytes)):
                    raise ValueError("ratings must be numeric, not boolean/str/None")
            arr = arr.astype(np.float64)
        if arr.dtype.kind == "b":
            raise ValueError("ratings must be integer codes, not booleans")
        if arr.dtype.kind not in "fiu":
            raise ValueError(f"ratings dtype {arr.dtype} is not numeric")
        ns, nr = arr.shape
        if arr.dtype.kind == "f":
            finite = np.isfinite(arr)
            if np.any(np.isinf(arr)):
                raise ValueError("ratings must not contain infinities")
            if np.any(arr[finite] != np.floor(arr[finite])):
                raise ValueError("ratings must be integer category codes")
            if np.any(np.abs(arr[finite]) > 2.0**53):
                raise ValueError("ratings exceed exact float64 integer range")
            codes = np.where(finite, arr, -1.0).astype(np.int64)
        else:
            if arr.dtype.kind == "u" and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
                raise ValueError("ratings values must fit in int64")
            codes = arr.astype(np.int64)

        if category_count is None:
            if codes.size == 0 or int(codes.max()) < 0:
                raise ValueError("cannot infer k: no observed category codes")
            category_count = int(codes.max()) + 1
            if category_count < 2:
                raise ValueError("k must be >= 2")
            if category_count > MAX_FLEISS_CATEGORIES:
                raise ValueError(f"k must be <= {MAX_FLEISS_CATEGORIES}")

        from . import _core

        result = _core.fleiss_kappa(
            np.ascontiguousarray(codes.reshape(-1)),
            int(ns),
            int(nr),
            category_count,
            exact_flag,
        )
        return module.FleissKappaResult(
            kappa=float(result["kappa"]),
            subjects_used=int(result["subjects_used"]),
            z=float(result["z"]),
            p_value=float(result["p_value"]),
            category_kappa=np.asarray(result["category_kappa"]),
            category_z=np.asarray(result["category_z"]),
            category_p=np.asarray(result["category_p"]),
        )

    safe_fleiss_kappa.__fast_mlsirm_fleiss_control_hardened__ = True
    module.fleiss_kappa = safe_fleiss_kappa
