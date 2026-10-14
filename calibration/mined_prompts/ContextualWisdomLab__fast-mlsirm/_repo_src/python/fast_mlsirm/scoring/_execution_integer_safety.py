"""Inert integer normalization for governed scoring execution contracts.

This composition layer exists because execution contracts are imported through the
stable scoring surface while their validation helpers remain private. It installs
exact-type integer normalizers before any public execution constructor is exposed,
so rejected caller values cannot dispatch conversion or comparison callbacks.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from ._validation import assessment_error

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


def _trusted_integer(value: Any, name: str, message: str) -> int:
    """Return an exact trusted integer without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise assessment_error(
        f"invalid_{name}",
        f"$.{name}",
        message,
    )


def _nonnegative_integer(value: Any, name: str, maximum: int) -> int:
    """Return one bounded nonnegative trusted integer control."""
    normalized = _trusted_integer(
        value,
        name,
        f"{name} must be an integer between 0 and {maximum}",
    )
    if not 0 <= normalized <= maximum:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be between 0 and {maximum}",
        )
    return normalized


def _score_integer(value: Any, name: str = "score_category") -> int:
    """Return one trusted integer score category without protocol dispatch."""
    return _trusted_integer(
        value,
        name,
        f"{name} must be an integer",
    )


def install(execution_module: ModuleType) -> None:
    """Install inert integer normalizers on the package-owned execution module."""
    execution_module._nonnegative_integer = _nonnegative_integer
    execution_module._score_integer = _score_integer
