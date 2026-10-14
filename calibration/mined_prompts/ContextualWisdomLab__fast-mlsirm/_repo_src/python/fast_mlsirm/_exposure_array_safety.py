"""Callback-safe container admission for exposure APIs.

The historical exposure surface accepts NumPy arrays and ordinary built-in
array-like sequences. Establish a package-trusted container/scalar boundary
before any ``np.asarray`` call so caller-defined ``__array__`` or numeric
protocols cannot execute during validation. All numerical psychometric work
remains in the existing Rust core.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_REAL_NUMERIC_KINDS = frozenset({"b", "i", "u", "f"})
_NUMPY_REAL_SCALAR_TYPES = frozenset(
    {
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
    }
)
_TRUSTED_REAL_SCALAR_TYPES = frozenset({bool, int, float}) | _NUMPY_REAL_SCALAR_TYPES
_TRUSTED_BOOLEAN_SCALAR_TYPES = frozenset({bool, np.bool_})


def exact_ndarray(value: object, *, message: str) -> np.ndarray:
    """Return an exact ndarray or fail before invoking array protocols."""

    if type(value) is not np.ndarray:
        raise ValueError(message)
    return value


def _safe_array_source(
    value: object,
    *,
    message: str,
    scalar_types: frozenset[type[object]],
) -> np.ndarray:
    """Admit an exact ndarray or a plain one-dimensional built-in sequence."""

    if type(value) is np.ndarray:
        return value
    if type(value) is not list and type(value) is not tuple:
        raise ValueError(message)
    # Exact list/tuple identity makes iteration inert. Inspect every element's
    # exact scalar identity before NumPy sees the sequence, so compatibility
    # does not reopen caller-defined conversion protocols.
    if any(type(item) not in scalar_types for item in value):
        raise ValueError(message)
    return np.asarray(value)


def install(exposure_module: ModuleType) -> None:
    """Replace exposure array admission helpers and guard inline CCAT groups."""

    def real_numeric_array(name: str, value: object) -> np.ndarray:
        message = f"{name} must be a real numeric array"
        array = _safe_array_source(
            value,
            message=message,
            scalar_types=_TRUSTED_REAL_SCALAR_TYPES,
        )
        if np.iscomplexobj(array) or array.dtype.kind not in _REAL_NUMERIC_KINDS:
            raise ValueError(message)
        return np.ascontiguousarray(array, dtype=np.float64)

    def boolean_array(name: str, value: object) -> np.ndarray:
        message = f"{name} must be a boolean array"
        array = _safe_array_source(
            value,
            message=message,
            scalar_types=_TRUSTED_BOOLEAN_SCALAR_TYPES,
        )
        if array.dtype != np.bool_:
            raise ValueError(message)
        return np.ascontiguousarray(array, dtype=np.bool_)

    def binary_response_array(name: str, value: object) -> np.ndarray:
        message = f"{name} must be a real numeric array"
        array = _safe_array_source(
            value,
            message=message,
            scalar_types=_TRUSTED_REAL_SCALAR_TYPES,
        )
        if np.iscomplexobj(array) or array.dtype.kind not in _REAL_NUMERIC_KINDS:
            raise ValueError(message)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        values = np.ascontiguousarray(array, dtype=np.float64)
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain only 0 or 1")
        return np.ascontiguousarray(values, dtype=np.uint8)

    exposure_module._as_real_numeric_array = real_numeric_array
    exposure_module._as_boolean_array = boolean_array
    exposure_module._as_binary_response_array = binary_response_array

    original_ccat: Callable[..., Any] = exposure_module.ccat_select

    @wraps(original_ccat)
    def safe_ccat_select(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray | None = None,
        *,
        groups: object,
        targets: np.ndarray,
        administered: np.ndarray,
        theta0: float,
    ) -> dict:
        """Reject unsafe CCAT controls/providers before legacy validation."""

        # Preserve the public contract that semantic controls fail before any
        # caller-owned array work. The wrapped implementation validates this
        # value again, but only after it has been normalized to an inert
        # package-owned float.
        theta0_value = exposure_module._as_real_scalar("theta0", theta0)
        groups_value = _safe_array_source(
            groups,
            message="groups must be a real numeric array",
            scalar_types=_TRUSTED_REAL_SCALAR_TYPES,
        )
        return original_ccat(
            a,
            b,
            c,
            groups=groups_value,
            targets=targets,
            administered=administered,
            theta0=theta0_value,
        )

    exposure_module.ccat_select = safe_ccat_select
