"""Inert integer normalization for governed essay adapter contracts.

The essay adapters accept Python and NumPy integer scalars for bounded counts and
offsets. This composition helper establishes an exact package-trusted scalar
identity before normalization so caller-defined conversion protocols or numeric
subclasses cannot execute code at the public validation boundary.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from .._validation import assessment_error

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


def _trusted_integer(value: Any, name: str, maximum: int) -> int:
    """Return one exact trusted integer without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise assessment_error(
        f"invalid_{name}",
        f"$.{name}",
        f"{name} must be an integer between 0 and {maximum}",
    )


def _nonnegative_integer(value: Any, name: str, maximum: int) -> int:
    """Return one bounded nonnegative trusted integer control."""
    normalized = _trusted_integer(value, name, maximum)
    if not 0 <= normalized <= maximum:
        raise assessment_error(
            f"invalid_{name}",
            f"$.{name}",
            f"{name} must be between 0 and {maximum}",
        )
    return normalized


def install(contracts_module: ModuleType) -> None:
    """Install inert integer normalization on the package-owned contracts module."""
    contracts_module._nonnegative_integer = _nonnegative_integer
