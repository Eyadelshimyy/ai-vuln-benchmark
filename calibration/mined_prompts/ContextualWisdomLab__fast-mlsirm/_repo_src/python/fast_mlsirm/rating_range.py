"""Paired rating-range evidence for governed automated-scoring validation.

The Python layer performs only fail-closed input validation and typed
marshalling. All descriptive arithmetic is delegated to the Rust extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rating_range_core_loader import rating_range_core

MAX_CATEGORY_COUNT = 1_000
MAX_OBSERVATIONS = 1_000_000
_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
)


@dataclass(frozen=True)
class RatingRangeEvidence:
    """Immutable paired descriptive evidence about ordinal category-range use."""

    sample_size: int
    automated_min: int
    automated_max: int
    reference_min: int
    reference_max: int
    automated_distinct_categories: int
    reference_distinct_categories: int
    automated_span: int
    reference_span: int
    automated_sd: float
    reference_sd: float
    span_ratio: float | None
    distinct_category_ratio: float
    sd_ratio: float | None
    lower_endpoint_gap: int
    upper_endpoint_gap: int
    narrower_observed_support: bool
    central_tendency_signal: bool


def _require_category_count(value: object) -> int:
    """Return a trusted built-in category count without caller callbacks.

    Only an exact Python ``int`` or an exact supported NumPy integer scalar
    identity is admitted. Subclasses are rejected before conversion, type
    hashing, equality, representation, or native-core dispatch can execute
    caller-controlled code.
    """
    value_type = type(value)
    if value_type is int:
        category_count = value
    elif any(value_type is trusted_type for trusted_type in _NUMPY_INTEGER_TYPES):
        category_count = int(value)
    else:
        raise ValueError("category_count must be an integer between 2 and 1000")
    if not 2 <= category_count <= MAX_CATEGORY_COUNT:
        raise ValueError("category_count must be between 2 and 1000")
    return category_count


def _rating_array(
    values: np.ndarray,
    name: str,
    *,
    category_count: int,
    expected_length: int | None = None,
) -> np.ndarray:
    """Validate one paired rating vector without performing scoring arithmetic."""
    if type(values) is not np.ndarray:
        raise ValueError(f"{name} must be a NumPy array")
    arr = values
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if arr.size > MAX_OBSERVATIONS:
        raise ValueError(f"{name} exceeds the {MAX_OBSERVATIONS} observation limit")
    if expected_length is not None and arr.size != expected_length:
        raise ValueError(f"{name} length must match the paired labels")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if arr.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer labels, not boolean values")
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric integer labels")
    if arr.dtype.kind == "f":
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite integer labels")
        if np.any(arr != np.floor(arr)):
            raise ValueError(f"{name} must contain integer labels")
    labels = arr.astype(np.float64, copy=False)
    if np.any(labels < 0):
        raise ValueError(f"{name} labels must be non-negative")
    if np.any(labels >= category_count):
        raise ValueError(f"{name} labels must be in 0..category_count-1")
    return np.ascontiguousarray(arr, dtype=np.uint32)


def paired_rating_range_evidence(
    automated: np.ndarray,
    reference: np.ndarray,
    *,
    category_count: int,
) -> RatingRangeEvidence:
    """Return Rust-owned descriptive range-use evidence on paired ratings.

    ``automated`` and ``reference`` must contain category labels for the same
    cases. Relative span and SD ratios are unavailable when the reference
    denominator is zero. No acceptance threshold is applied.
    """
    category_count = _require_category_count(category_count)

    automated_v = _rating_array(
        automated,
        "automated",
        category_count=category_count,
    )
    reference_v = _rating_array(
        reference,
        "reference",
        category_count=category_count,
        expected_length=automated_v.size,
    )
    raw = rating_range_core().paired_rating_range_evidence(
        automated_v,
        reference_v,
        category_count,
    )
    return RatingRangeEvidence(
        sample_size=int(raw["sample_size"]),
        automated_min=int(raw["automated_min"]),
        automated_max=int(raw["automated_max"]),
        reference_min=int(raw["reference_min"]),
        reference_max=int(raw["reference_max"]),
        automated_distinct_categories=int(raw["automated_distinct_categories"]),
        reference_distinct_categories=int(raw["reference_distinct_categories"]),
        automated_span=int(raw["automated_span"]),
        reference_span=int(raw["reference_span"]),
        automated_sd=float(raw["automated_sd"]),
        reference_sd=float(raw["reference_sd"]),
        span_ratio=None if raw["span_ratio"] is None else float(raw["span_ratio"]),
        distinct_category_ratio=float(raw["distinct_category_ratio"]),
        sd_ratio=None if raw["sd_ratio"] is None else float(raw["sd_ratio"]),
        lower_endpoint_gap=int(raw["lower_endpoint_gap"]),
        upper_endpoint_gap=int(raw["upper_endpoint_gap"]),
        narrower_observed_support=bool(raw["narrower_observed_support"]),
        central_tendency_signal=bool(raw["central_tendency_signal"]),
    )
