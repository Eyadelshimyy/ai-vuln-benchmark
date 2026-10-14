"""Callback-free admission guards for Rust-owned inference diagnostics."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_REAL_KINDS = frozenset({"b", "i", "u", "f"})
_NUMPY_REAL_SCALAR_TYPES = tuple(
    dict.fromkeys(
        np.dtype(code).type
        for code in np.typecodes["All"]
        if np.dtype(code).kind in _REAL_KINDS
    )
)
_NUMPY_COMPLEX_SCALAR_TYPES = tuple(
    dict.fromkeys(
        np.dtype(code).type
        for code in np.typecodes["All"]
        if np.dtype(code).kind == "c"
    )
)
_MAX_INFERENCE_MATRIX_CELLS = 20_000_000


def _is_numpy_real_scalar(value: object, *, allow_bool: bool) -> bool:
    """Return whether ``value`` is a concrete package-trusted NumPy real scalar."""
    value_type = type(value)
    for scalar_type in _NUMPY_REAL_SCALAR_TYPES:
        if value_type is scalar_type:
            return allow_bool or scalar_type is not np.bool_
    return False


def _is_real_scalar(value: object, *, allow_bool: bool) -> bool:
    value_type = type(value)
    if value_type is bool:
        return allow_bool
    if value_type is int or value_type is float:
        return True
    return _is_numpy_real_scalar(value, allow_bool=allow_bool)


def _normalized_nonnegative_real(value: object, name: str) -> float:
    """Normalize a trusted finite non-negative Rust ``f64`` control."""
    if type(value) is np.ndarray:
        if value.ndim != 0 or value.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"{name} must be a finite non-negative float")
        return _normalized_nonnegative_real(value[()], name)

    if not _is_real_scalar(value, allow_bool=False):
        raise ValueError(f"{name} must be a finite non-negative float")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative float") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")

    if type(value) is int:
        if int(normalized) != value:
            raise ValueError(f"{name} must be losslessly representable as float64")
    elif _is_numpy_real_scalar(value, allow_bool=False):
        value_type = type(value)
        try:
            if value_type(normalized) != value:
                raise ValueError(f"{name} must be losslessly representable as float64")
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be losslessly representable as float64") from exc
    return normalized


def _normalized_positive_real(value: object, name: str) -> float:
    """Normalize a trusted finite strictly-positive Rust ``f64`` control."""
    try:
        normalized = _normalized_nonnegative_real(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be > 0 and finite") from exc
    if normalized <= 0.0:
        raise ValueError(f"{name} must be > 0 and finite")
    return normalized


def _check_matrix_cells(side: int, name: str) -> None:
    """Bound dense square-matrix work before any float64 materialization."""
    cells = side * side
    if cells > _MAX_INFERENCE_MATRIX_CELLS:
        raise ValueError(
            f"{name} resource limit exceeded: {cells} cells requested, at most "
            f"{_MAX_INFERENCE_MATRIX_CELLS} are supported"
        )


def _lossless_matrix_error(name: str) -> ValueError:
    return ValueError(f"{name} entries must be losslessly representable as float64")


def _scalar_is_lossless_float64(value: object) -> bool:
    """Return whether one already-trusted scalar preserves identity in Rust ``f64``."""
    value_type = type(value)
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    if value_type is bool or value_type is float:
        return True
    if value_type is int:
        return np.isfinite(normalized) and int(normalized) == value

    # The caller reached this helper only after exact concrete NumPy-scalar admission.
    try:
        roundtrip = value_type(normalized)
    except (OverflowError, TypeError, ValueError):
        return False
    try:
        if np.dtype(value_type).kind == "f" and np.isnan(value) and np.isnan(roundtrip):
            return True
    except TypeError:
        return False
    return bool(roundtrip == value)


def _numpy_array_float64(value: np.ndarray, name: str) -> np.ndarray:
    """Normalize one exact numeric ndarray and prove any narrowing is lossless."""
    try:
        normalized = np.ascontiguousarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _lossless_matrix_error(name) from exc

    kind = value.dtype.kind
    # Boolean, <=32-bit integers, and <=64-bit IEEE floats are exactly representable
    # in binary64 under the admitted real-numeric dtype contract.
    if kind == "b" or (kind in {"i", "u"} and value.dtype.itemsize <= 4) or (
        kind == "f" and value.dtype.itemsize <= 8
    ):
        return normalized

    row_count = value.shape[0] if value.ndim > 1 else 1
    for index in range(row_count):
        source = value[index] if value.ndim > 1 else value
        target = normalized[index] if value.ndim > 1 else normalized
        try:
            with np.errstate(invalid="ignore", over="ignore"):
                roundtrip = target.astype(value.dtype)
        except (OverflowError, TypeError, ValueError) as exc:
            raise _lossless_matrix_error(name) from exc
        equal = roundtrip == source
        if kind == "f":
            equal = equal | (np.isnan(roundtrip) & np.isnan(source))
        if not bool(np.all(equal)):
            raise _lossless_matrix_error(name)
    return normalized


def _validate_row(row: object, name: str, expected: int) -> None:
    """Validate one inert matrix row without invoking caller protocols."""
    row_type = type(row)
    if row_type is np.ndarray:
        if row.ndim != 1 or row.shape[0] != expected:
            raise ValueError(f"{name} must be a square matrix")
        if row.dtype.kind not in _REAL_KINDS:
            if row.dtype.kind == "c":
                raise ValueError(f"{name} must be real-valued")
            raise ValueError(f"{name} must contain real numeric values")
        _numpy_array_float64(row, name)
        return
    if row_type is not list and row_type is not tuple:
        raise ValueError(f"{name} must be an exact NumPy array or built-in matrix")
    if len(row) != expected:
        raise ValueError(f"{name} must be a square matrix")
    for value in row:
        if not _is_real_scalar(value, allow_bool=True):
            if type(value) is complex or _is_numpy_complex_scalar(value):
                raise ValueError(f"{name} must be real-valued")
            raise ValueError(f"{name} must contain real numeric values")
        if not _scalar_is_lossless_float64(value):
            raise _lossless_matrix_error(name)


def _is_numpy_complex_scalar(value: object) -> bool:
    value_type = type(value)
    return any(value_type is scalar_type for scalar_type in _NUMPY_COMPLEX_SCALAR_TYPES)


def _real_square_matrix(value: object, name: str) -> np.ndarray:
    """Seal, bound, and losslessly normalize square matrices before Rust."""
    value_type = type(value)
    if value_type is np.ndarray:
        if value.dtype.kind not in _REAL_KINDS:
            if value.dtype.kind == "c":
                raise ValueError(f"{name} must be real-valued")
            raise ValueError(f"{name} must contain real numeric values")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
            raise ValueError(f"{name} must be a square matrix")
        _check_matrix_cells(int(value.shape[0]), name)
        return _numpy_array_float64(value, name)

    if value_type is not list and value_type is not tuple:
        raise ValueError(f"{name} must be an exact NumPy array or built-in matrix")
    size = len(value)
    if size == 0:
        raise ValueError(f"{name} must be a square matrix")
    _check_matrix_cells(size, name)
    for row in value:
        _validate_row(row, name, size)
    try:
        return np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (OverflowError, TypeError, ValueError) as exc:
        raise _lossless_matrix_error(name) from exc


def install(inference_module: ModuleType) -> None:
    """Install idempotent inference admission wrappers on ``inference_module``."""
    if getattr(inference_module.second_order_test, "_inference_admission_safe", False):
        return

    raw_observed_information = inference_module.observed_information
    raw_second_order = inference_module.second_order_test
    raw_vcov = inference_module.vcov_from_hessian
    raw_standard_errors = inference_module.standard_errors_from_vcov

    @wraps(raw_observed_information)
    def safe_observed_information(
        responses: Any,
        factor_id: Any,
        params: Any,
        config: Any = None,
        mask: Any = None,
        backend: Any = None,
        device: Any = "cpu",
        step: Any = 1e-4,
    ):
        normalized_step = _normalized_positive_real(step, "step")
        return raw_observed_information(
            responses,
            factor_id,
            params,
            config=config,
            mask=mask,
            backend=backend,
            device=device,
            step=normalized_step,
        )

    @wraps(raw_second_order)
    def safe_second_order_test(hessian: Any, tol: Any = 1e-8):
        normalized_tol = _normalized_nonnegative_real(tol, "tol")
        matrix = _real_square_matrix(hessian, "hessian")
        return raw_second_order(matrix, tol=normalized_tol)

    @wraps(raw_vcov)
    def safe_vcov_from_hessian(hessian: Any, rcond: Any = 1e-10):
        normalized_rcond = _normalized_nonnegative_real(rcond, "rcond")
        matrix = _real_square_matrix(hessian, "hessian")
        return raw_vcov(matrix, rcond=normalized_rcond)

    @wraps(raw_standard_errors)
    def safe_standard_errors_from_vcov(vcov: Any):
        matrix = _real_square_matrix(vcov, "vcov")
        return raw_standard_errors(matrix)

    safe_observed_information._inference_admission_safe = True
    safe_second_order_test._inference_admission_safe = True
    safe_vcov_from_hessian._inference_admission_safe = True
    safe_standard_errors_from_vcov._inference_admission_safe = True

    inference_module.observed_information = safe_observed_information
    inference_module.second_order_test = safe_second_order_test
    inference_module.vcov_from_hessian = safe_vcov_from_hessian
    inference_module.standard_errors_from_vcov = safe_standard_errors_from_vcov
