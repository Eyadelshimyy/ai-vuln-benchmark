"""Stable cut-score normalization for public classification adapters."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_ERROR = "cutscores entries must be finite real scalars"
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
_NUMPY_FLOAT_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


def _has_exact_type(value: object, trusted_types: tuple[type, ...]) -> bool:
    """Return whether ``value`` has one exact trusted scalar type."""

    value_type = type(value)
    return any(value_type is trusted_type for trusted_type in trusted_types)


def normalize_cutscores(cutscores: Sequence[float]) -> list[float]:
    """Normalize finite trusted cut scores without caller coercion hooks."""

    try:
        iterator = iter(cutscores)
    except TypeError as error:
        raise ValueError(_ERROR) from error

    normalized: list[float] = []
    for value in iterator:
        value_type = type(value)
        if value_type is bool or value_type is np.bool_:
            raise ValueError(_ERROR)
        if not (
            value_type is int
            or value_type is float
            or _has_exact_type(value, _NUMPY_INTEGER_SCALAR_TYPES)
            or _has_exact_type(value, _NUMPY_FLOAT_SCALAR_TYPES)
        ):
            raise ValueError(_ERROR)
        try:
            parsed = float(value)
        except OverflowError as error:
            raise ValueError(_ERROR) from error
        if not math.isfinite(parsed):
            raise ValueError(_ERROR)
        normalized.append(parsed)
    return normalized
