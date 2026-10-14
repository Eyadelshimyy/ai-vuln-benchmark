"""IRT model specifications shared by dimension-agnostic item-family APIs.

The public fitting functions use one ``model=`` argument, following the R
``mirt`` convention: a number denotes an exploratory factor count, while a
confirmatory specification declares the loading pattern.  The current Rust
estimators implement unrestricted exploratory estimation only for one factor;
multidimensional exploratory requests fail explicitly until rotation and
identification are implemented.

References (APA 7th ed.):
    Chalmers, R. P. (2012). mirt: A multidimensional item response theory
        package for the R environment. *Journal of Statistical Software, 48*(6),
        1-29. https://doi.org/10.18637/jss.v048.i06
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConfirmatoryModel",
    "ExploratoryModel",
    "IrtModel",
    "confirmatory",
    "exploratory",
]


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
_NUMPY_COMPLEX_SCALAR_TYPES = (
    np.complex64,
    np.complex128,
    np.clongdouble,
)

_CONFIRMATORY_SHAPE_ERROR = (
    "confirmatory loading_pattern must be a non-empty 2-D items x dimensions array"
)
_CONFIRMATORY_NUMERIC_ERROR = "confirmatory loading_pattern entries must be numeric 0 or 1"
_CONFIRMATORY_REAL_ERROR = "confirmatory loading_pattern entries must be real 0 or 1"
_CONFIRMATORY_BINARY_ERROR = (
    "confirmatory loading_pattern entries must be finite and exactly 0 or 1"
)
_CONFIRMATORY_REPLAY_ERROR = "confirmatory model loading_pattern is not canonical"


def _is_exact_numpy_integer_scalar(value: object) -> bool:
    """Return whether ``value`` has an exact supported NumPy integer type."""

    value_type = type(value)
    return any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES)


def _is_exact_numpy_floating_scalar(value: object) -> bool:
    """Return whether ``value`` has an exact supported NumPy floating type."""

    value_type = type(value)
    return any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES)


def _is_exact_numpy_complex_scalar(value: object) -> bool:
    """Return whether ``value`` has an exact supported NumPy complex type."""

    value_type = type(value)
    return any(value_type is scalar_type for scalar_type in _NUMPY_COMPLEX_SCALAR_TYPES)


def _require_exploratory_dimensions(value: object) -> int:
    """Return a trusted positive factor count without caller coercion hooks."""

    if type(value) is int:
        dimensions = value
    elif _is_exact_numpy_integer_scalar(value):
        dimensions = int(value)
    else:
        raise ValueError("exploratory dimensions must be a positive integer")
    if dimensions < 1:
        raise ValueError("exploratory dimensions must be a positive integer")
    return dimensions


def _confirmatory_scalar(value: object) -> int:
    """Normalize one trusted binary loading entry without caller coercion hooks."""

    value_type = type(value)
    if value_type is complex or _is_exact_numpy_complex_scalar(value):
        raise ValueError(_CONFIRMATORY_REAL_ERROR)
    if value_type is bool or value_type is np.bool_:
        return int(value)
    if value_type is int or _is_exact_numpy_integer_scalar(value):
        if value != 0 and value != 1:
            raise ValueError(_CONFIRMATORY_BINARY_ERROR)
        return int(value)
    if value_type is float or _is_exact_numpy_floating_scalar(value):
        if not np.isfinite(value) or (value != 0 and value != 1):
            raise ValueError(_CONFIRMATORY_BINARY_ERROR)
        return int(value)
    raise ValueError(_CONFIRMATORY_NUMERIC_ERROR)


def _confirmatory_ndarray_row(
    row: np.ndarray,
    width: int | None,
) -> tuple[list[int], int]:
    """Normalize one exact numeric NumPy row without array-protocol dispatch."""

    if row.ndim != 1 or row.shape[0] < 1:
        raise ValueError(_CONFIRMATORY_SHAPE_ERROR)
    row_width = int(row.shape[0])
    if width is not None and row_width != width:
        raise ValueError(_CONFIRMATORY_SHAPE_ERROR)
    if row.dtype.kind == "c":
        raise ValueError(_CONFIRMATORY_REAL_ERROR)
    if row.dtype.kind not in "biuf":
        raise ValueError(_CONFIRMATORY_NUMERIC_ERROR)
    if not np.all(np.isfinite(row)) or not np.all((row == 0) | (row == 1)):
        raise ValueError(_CONFIRMATORY_BINARY_ERROR)
    return row.astype(np.int64, copy=False).tolist(), row_width


def _trusted_confirmatory_pattern(value: object) -> np.ndarray:
    """Return canonical binary loading evidence without caller NumPy protocols."""

    if type(value) is np.ndarray:
        raw = value
        if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
            raise ValueError(_CONFIRMATORY_SHAPE_ERROR)
        if raw.dtype.kind == "c":
            raise ValueError(_CONFIRMATORY_REAL_ERROR)
        if raw.dtype.kind not in "biuf":
            raise ValueError(_CONFIRMATORY_NUMERIC_ERROR)
        if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
            raise ValueError(_CONFIRMATORY_BINARY_ERROR)
        pattern = np.array(raw, dtype=np.int64, copy=True, order="C")
        pattern.setflags(write=False)
        return pattern

    if type(value) is not list and type(value) is not tuple:
        raise ValueError(_CONFIRMATORY_SHAPE_ERROR)
    if len(value) < 1:
        raise ValueError(_CONFIRMATORY_SHAPE_ERROR)

    normalized_rows: list[list[int]] = []
    width: int | None = None
    for row in value:
        if type(row) is np.ndarray:
            normalized, row_width = _confirmatory_ndarray_row(row, width)
        elif type(row) is list or type(row) is tuple:
            if len(row) < 1:
                raise ValueError(_CONFIRMATORY_SHAPE_ERROR)
            row_width = len(row)
            if width is not None and row_width != width:
                raise ValueError(_CONFIRMATORY_SHAPE_ERROR)
            normalized = [_confirmatory_scalar(entry) for entry in row]
        else:
            raise ValueError(_CONFIRMATORY_SHAPE_ERROR)
        if width is None:
            width = row_width
        normalized_rows.append(normalized)

    pattern = np.asarray(normalized_rows, dtype=np.int64)
    pattern.setflags(write=False)
    return pattern


def _current_confirmatory_pattern(model: "ConfirmatoryModel") -> np.ndarray:
    """Replay the package-owned canonical structure before public field use."""

    pattern = model.loading_pattern
    if type(pattern) is not np.ndarray:
        raise ValueError(_CONFIRMATORY_REPLAY_ERROR)
    if (
        pattern.dtype != np.dtype(np.int64)
        or pattern.ndim != 2
        or pattern.shape[0] < 1
        or pattern.shape[1] < 1
        or pattern.flags.writeable
        or not pattern.flags.c_contiguous
        or not np.all((pattern == 0) | (pattern == 1))
    ):
        raise ValueError(_CONFIRMATORY_REPLAY_ERROR)
    return pattern


@dataclass(frozen=True)
class ExploratoryModel:
    """An exploratory model identified by its number of latent dimensions."""

    dimensions: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dimensions",
            _require_exploratory_dimensions(self.dimensions),
        )

    @property
    def n_dims(self) -> int:
        """Derived latent dimension count."""

        return self.dimensions


@dataclass(frozen=True, eq=False)
class ConfirmatoryModel:
    """A confirmatory model defined by an items-by-dimensions loading pattern."""

    loading_pattern: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "loading_pattern",
            _trusted_confirmatory_pattern(self.loading_pattern),
        )

    @property
    def n_dims(self) -> int:
        """Derived latent dimension count."""

        return int(_current_confirmatory_pattern(self).shape[1])


IrtModel = ExploratoryModel | ConfirmatoryModel


def exploratory(dimensions: int = 1) -> ExploratoryModel:
    """Build an exploratory model specification."""

    return ExploratoryModel(dimensions)


def confirmatory(loading_pattern: np.ndarray) -> ConfirmatoryModel:
    """Build a confirmatory model specification from a binary loading pattern."""

    return ConfirmatoryModel(loading_pattern)


def _resolve_model(
    model: int | IrtModel,
    n_items: int,
) -> tuple[IrtModel, np.ndarray]:
    """Normalize a public model argument to a specification and core loading pattern."""

    if type(model) is bool:
        raise TypeError("model must be a factor count or an IRT model specification")
    if type(model) is int or _is_exact_numpy_integer_scalar(model):
        model = ExploratoryModel(model)
    if type(model) is ExploratoryModel:
        if model.dimensions != 1:
            raise NotImplementedError(
                "multidimensional exploratory loading estimation is not implemented; "
                "use models.confirmatory(...) for an identified loading structure"
            )
        return model, np.ones((n_items, 1), dtype=np.int64)
    if type(model) is ConfirmatoryModel:
        pattern = _current_confirmatory_pattern(model)
        if pattern.shape[0] != n_items:
            raise ValueError(
                "confirmatory model must have one loading-pattern row per item"
            )
        return model, pattern
    raise TypeError("model must be a factor count or an IRT model specification")
