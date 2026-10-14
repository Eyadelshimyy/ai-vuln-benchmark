"""Rust-backed residual interaction maps for downstream measurement products."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _core

_MAX_INTERACTION_MAP_CELLS = 20_000_000
_MAX_INTERACTION_MAP_STRUCTURAL_NODES = 2 * _MAX_INTERACTION_MAP_CELLS
_MAX_INTERACTION_MAP_COORDINATE_CELLS = 20_000_000
_TRUSTED_NUMPY_INTEGER_TYPES = (
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
_TRUSTED_NUMPY_FLOAT_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


@dataclass(frozen=True)
class ResidualInteractionMap:
    """Complete-case Gabriel coordinates and auditable cell decomposition."""

    person_indices: np.ndarray
    item_indices: np.ndarray
    scored_person_count: int
    scored_item_count: int
    person_coordinates: np.ndarray
    item_coordinates: np.ndarray
    singular_values: np.ndarray
    axis_shares: np.ndarray
    residual: np.ndarray
    distance: np.ndarray
    reconstruction: np.ndarray
    unexplained: np.ndarray
    cross_share: np.ndarray


def _axis_count(value: object) -> int:
    """Normalize one inert positive integer axis request without caller coercion."""
    value_type = type(value)
    if value_type is int:
        parsed = value
    elif any(value_type is trusted for trusted in _TRUSTED_NUMPY_INTEGER_TYPES):
        parsed = int(value)
    else:
        raise TypeError("axis_count must be a positive integer")
    if parsed <= 0:
        raise ValueError("axis_count must be a positive integer")
    if parsed > _MAX_INTERACTION_MAP_COORDINATE_CELLS:
        raise ValueError(
            "interaction map coordinate request exceeds "
            f"{_MAX_INTERACTION_MAP_COORDINATE_CELLS} cells"
        )
    return parsed


def _trusted_scalar_float64(name: str, value: object, *, allow_nan: bool) -> float:
    """Return one callback-free scalar whose numeric identity survives binary64."""
    value_type = type(value)
    if value_type is bool:
        return float(value)
    if value_type is int:
        try:
            parsed = float(value)
        except OverflowError:
            raise ValueError(
                f"{name} values must be exactly representable as float64"
            ) from None
        if not math.isfinite(parsed) or int(parsed) != value:
            raise ValueError(f"{name} values must be exactly representable as float64")
        return parsed
    if value_type is float:
        if math.isinf(value):
            raise ValueError(f"{name} must not contain infinite values")
        if math.isnan(value) and not allow_nan:
            raise ValueError(f"{name} must not contain NaN values")
        return value
    if value_type is np.bool_:
        return float(value)
    if any(value_type is trusted for trusted in _TRUSTED_NUMPY_INTEGER_TYPES):
        integer = int(value)
        try:
            parsed = float(integer)
        except OverflowError:
            raise ValueError(
                f"{name} values must be exactly representable as float64"
            ) from None
        if not math.isfinite(parsed) or int(parsed) != integer:
            raise ValueError(f"{name} values must be exactly representable as float64")
        return parsed
    if any(value_type is trusted for trusted in _TRUSTED_NUMPY_FLOAT_TYPES):
        if np.isinf(value):
            raise ValueError(f"{name} must not contain infinite values")
        if np.isnan(value):
            if not allow_nan:
                raise ValueError(f"{name} must not contain NaN values")
            return float("nan")
        parsed = float(value)
        if not math.isfinite(parsed) or value_type(parsed) != value:
            raise ValueError(f"{name} values must be exactly representable as float64")
        return parsed
    raise ValueError(f"{name} must contain only real numeric values")


def _lossless_float64_array(
    name: str, value: np.ndarray, *, allow_nan: bool
) -> np.ndarray:
    """Normalize one exact numeric ndarray without silently changing evidence."""
    if value.dtype.kind not in ("b", "i", "u", "f"):
        if value.dtype.kind == "c":
            raise ValueError(f"{name} must be real-valued")
        raise ValueError(f"{name} must contain only real numeric values")
    if np.isinf(value).any():
        raise ValueError(f"{name} must not contain infinite values")
    if not allow_nan and np.isnan(value).any():
        raise ValueError(f"{name} must not contain NaN values")

    with np.errstate(invalid="ignore", over="ignore"):
        converted = np.ascontiguousarray(value, dtype=np.float64)
    if value.dtype.kind in ("i", "u"):
        with np.errstate(invalid="ignore", over="ignore"):
            round_trip = converted.astype(value.dtype)
        if not np.array_equal(round_trip, value):
            raise ValueError(f"{name} values must be exactly representable as float64")
    elif value.dtype.kind == "f":
        with np.errstate(invalid="ignore", over="ignore"):
            round_trip = converted.astype(value.dtype)
        if not np.array_equal(round_trip, value, equal_nan=True):
            raise ValueError(f"{name} values must be exactly representable as float64")
    return converted


def _trusted_matrix(name: str, value: object, *, allow_nan: bool) -> np.ndarray:
    """Admit one bounded two-dimensional real-numeric matrix without protocols."""
    if type(value) is np.ndarray:
        raw = value
        if raw.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if raw.size > _MAX_INTERACTION_MAP_CELLS:
            raise ValueError(
                f"{name} logical-cell count exceeds {_MAX_INTERACTION_MAP_CELLS}"
            )
        return _lossless_float64_array(name, raw, allow_nan=allow_nan)

    if type(value) not in (list, tuple):
        raise ValueError(
            f"{name} must be an exact NumPy array or built-in two-dimensional sequence"
        )
    if not value:
        raise ValueError(f"{name} must be two-dimensional")

    width: int | None = None
    normalized_rows: list[list[float]] = []
    logical_cells = 0
    structural_nodes = 0
    for row in value:
        structural_nodes += 1
        if structural_nodes > _MAX_INTERACTION_MAP_STRUCTURAL_NODES:
            raise ValueError(
                f"{name} structural-node count exceeds "
                f"{_MAX_INTERACTION_MAP_STRUCTURAL_NODES}"
            )
        if type(row) is np.ndarray:
            if row.ndim != 1:
                raise ValueError(f"{name} must be two-dimensional")
            row_width = int(row.size)
            structural_nodes += row_width
            if structural_nodes > _MAX_INTERACTION_MAP_STRUCTURAL_NODES:
                raise ValueError(
                    f"{name} structural-node count exceeds "
                    f"{_MAX_INTERACTION_MAP_STRUCTURAL_NODES}"
                )
            logical_cells += row_width
            if logical_cells > _MAX_INTERACTION_MAP_CELLS:
                raise ValueError(
                    f"{name} logical-cell count exceeds {_MAX_INTERACTION_MAP_CELLS}"
                )
            normalized_row = _lossless_float64_array(
                name, row, allow_nan=allow_nan
            ).tolist()
        elif type(row) in (list, tuple):
            row_width = len(row)
            structural_nodes += row_width
            if structural_nodes > _MAX_INTERACTION_MAP_STRUCTURAL_NODES:
                raise ValueError(
                    f"{name} structural-node count exceeds "
                    f"{_MAX_INTERACTION_MAP_STRUCTURAL_NODES}"
                )
            logical_cells += row_width
            if logical_cells > _MAX_INTERACTION_MAP_CELLS:
                raise ValueError(
                    f"{name} logical-cell count exceeds {_MAX_INTERACTION_MAP_CELLS}"
                )
            normalized_row = [
                _trusted_scalar_float64(name, cell, allow_nan=allow_nan) for cell in row
            ]
        else:
            raise ValueError(f"{name} must be two-dimensional")

        if width is None:
            width = row_width
        elif row_width != width:
            raise ValueError(f"{name} must be rectangular")
        normalized_rows.append(normalized_row)

    return np.ascontiguousarray(normalized_rows, dtype=np.float64)


def residual_interaction_map(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    axis_count: int,
) -> ResidualInteractionMap:
    """Factor ``observed - expected`` using Gabriel symmetric scaling.

    Missing observed cells are represented by ``NaN`` and excluded through a
    complete-case rectangle; they are never filled with zero. Model expectations
    must be finite. Infinity is not treated as missing. ``axis_count`` is required
    because the consuming measurement contract, not this library, determines how
    many reader-visible axes are retained. Controls are sealed before caller
    evidence is inspected; accepted matrices are callback-free, lossless at the
    Rust ``f64`` boundary, and bounded by logical-cell and structural-work limits
    before dense materialization.

    References:
        Gabriel, K. R. (1971). The biplot graphic display of matrices with
            application to principal component analysis. *Biometrika, 58*(3),
            453-467. https://doi.org/10.1093/biomet/58.3.453
        Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping
            unobserved item-respondent interactions: A latent space item
            response model with interaction map. *Psychometrika, 86*(2),
            378-403. https://doi.org/10.1007/s11336-021-09762-5
    """
    axis_count_value = _axis_count(axis_count)
    observed_array = _trusted_matrix("observed", observed, allow_nan=True)
    expected_array = _trusted_matrix("expected", expected, allow_nan=False)
    if observed_array.shape != expected_array.shape:
        raise ValueError(
            "observed and expected must have the same two-dimensional shape"
        )

    rows, columns = observed_array.shape
    coordinate_cells = (rows + columns) * axis_count_value
    if coordinate_cells > _MAX_INTERACTION_MAP_COORDINATE_CELLS:
        raise ValueError(
            "interaction map coordinate request exceeds "
            f"{_MAX_INTERACTION_MAP_COORDINATE_CELLS} cells"
        )

    raw = dict(
        _core.residual_interaction_map(observed_array, expected_array, axis_count_value)
    )
    person_indices = np.asarray(raw["person_indices"], dtype=np.int64)
    item_indices = np.asarray(raw["item_indices"], dtype=np.int64)
    rows = person_indices.size
    columns = item_indices.size
    return ResidualInteractionMap(
        person_indices=person_indices,
        item_indices=item_indices,
        scored_person_count=int(raw["scored_person_count"]),
        scored_item_count=int(raw["scored_item_count"]),
        person_coordinates=np.asarray(
            raw["person_coordinates"], dtype=np.float64
        ).reshape(rows, axis_count_value),
        item_coordinates=np.asarray(raw["item_coordinates"], dtype=np.float64).reshape(
            columns, axis_count_value
        ),
        singular_values=np.asarray(raw["singular_values"], dtype=np.float64),
        axis_shares=np.asarray(raw["axis_shares"], dtype=np.float64),
        residual=np.asarray(raw["residual"], dtype=np.float64).reshape(rows, columns),
        distance=np.asarray(raw["distance"], dtype=np.float64).reshape(rows, columns),
        reconstruction=np.asarray(raw["reconstruction"], dtype=np.float64).reshape(
            rows, columns
        ),
        unexplained=np.asarray(raw["unexplained"], dtype=np.float64).reshape(
            rows, columns
        ),
        cross_share=np.asarray(
            [np.nan if value is None else value for value in raw["cross_share"]],
            dtype=np.float64,
        ).reshape(rows, columns),
    )
