"""Callback-free admission for public fit-statistics controls and BH evidence.

This composition layer replaces only Python validation and marshalling. The
S-X²/G² statistics, quadrature, BH/FDR decisions, and all production numerical
work remain owned by the compiled Rust fit-statistics implementation.
"""

from __future__ import annotations

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
_NUMPY_FLOAT_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_BH_HARDENED_ATTR = "__fast_mlsirm_bh_admission_hardened__"
_MAX_PROBABILITY_TREE_DEPTH = 64
_MAX_BH_PROBABILITY_CELLS = 20_000_000
_MAX_BH_STRUCTURAL_NODES = 40_000_000


def _trusted_integer(value: Any, name: str) -> int:
    """Return one exact trusted integer without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _trusted_integer_real(value: Any, name: str) -> float:
    """Return an integer-valued real only when float64 preserves it exactly."""
    integer = int(value)
    try:
        converted = float(integer)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if int(converted) != integer:
        raise ValueError(f"{name} must be exactly representable as float64")
    return converted


def _trusted_numpy_real(value: Any, name: str) -> float:
    """Return a trusted NumPy real only when Rust f64 preserves its value."""
    value_type = type(value)
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if value_type(converted) != value:
        raise ValueError(f"{name} must be exactly representable as float64")
    return converted


def _trusted_real(value: Any, name: str) -> float:
    """Return one exact trusted real scalar without caller conversion hooks."""
    value_type = type(value)
    if value_type is int:
        return _trusted_integer_real(value, name)
    if value_type is float:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return _trusted_integer_real(value, name)
    if any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES):
        return _trusted_numpy_real(value, name)
    raise ValueError(f"{name} must be a finite number")


def _validate_sx2_controls(
    q_theta: Any,
    q_xi: Any,
    min_expected: Any,
    fdr_q: Any,
    min_effect: Any,
) -> tuple[int, int, float, float, float]:
    """Validate S-X² scalar controls without executing caller callbacks."""
    quadrature = []
    for name, value in (("q_theta", q_theta), ("q_xi", q_xi)):
        normalized = _trusted_integer(value, name)
        # #1929: no node-count cap; the Rust core generates any n >= 1 rule
        # on demand (Golub & Welsch, 1969) and guards allocation overflow.
        if normalized < 1:
            raise ValueError(f"{name} must be >= 1")
        quadrature.append(normalized)

    numeric = []
    for name, value in (
        ("min_expected", min_expected),
        ("fdr_q", fdr_q),
        ("min_effect", min_effect),
    ):
        converted = _trusted_real(value, name)
        if not np.isfinite(converted):
            raise ValueError(f"{name} must be a finite number")
        numeric.append(converted)

    min_expected_value, fdr_q_value, min_effect_value = numeric
    if min_expected_value <= 0.0:
        raise ValueError("min_expected must be positive")
    if not 0.0 < fdr_q_value <= 1.0:
        raise ValueError("fdr_q must be in (0, 1]")
    if min_effect_value < 0.0:
        raise ValueError("min_effect must be non-negative")
    return (
        quadrature[0],
        quadrature[1],
        min_expected_value,
        fdr_q_value,
        min_effect_value,
    )


def _bh_resource_error() -> str:
    """Return the stable BH logical-cell resource diagnostic."""
    return f"p_values exceed the {_MAX_BH_PROBABILITY_CELLS:,}-cell BH resource limit"


def _add_bh_cells(current: int, added: int) -> int:
    """Charge logical p-value cells without crossing the package envelope."""
    total = current + added
    if total > _MAX_BH_PROBABILITY_CELLS:
        raise ValueError(_bh_resource_error())
    return total


def _trusted_probability_tree(value: Any) -> int:
    """Preflight inert probability evidence with bounded logical/structural work."""
    # Each frame stores [object, depth, next_child_index, entered, logical_cells].
    # Children are pushed one at a time so a huge malformed built-in fan-out
    # cannot allocate an equally huge validation stack before the node budget is
    # checked. Shared acyclic containers retain occurrence semantics: every
    # occurrence is charged against both structural work and logical p-value
    # cells, while active-path identity catches true cycles.
    frames: list[list[Any]] = [[value, 0, 0, False, 0]]
    active_container_ids: set[int] = set()
    structural_nodes = 0

    while frames:
        frame = frames[-1]
        current = frame[0]
        depth = int(frame[1])
        current_type = type(current)

        if current_type is np.ndarray:
            cells = int(current.size)
            if cells > _MAX_BH_PROBABILITY_CELLS:
                raise ValueError(_bh_resource_error())
            if np.iscomplexobj(current) or current.dtype.kind not in "biuf":
                raise ValueError("p_values must contain real numeric probabilities")
            frames.pop()
            if frames:
                frames[-1][4] = _add_bh_cells(int(frames[-1][4]), cells)
            else:
                return cells
            continue

        if (
            current_type is bool
            or current_type is int
            or current_type is float
            or current_type is np.bool_
            or any(current_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES)
            or any(current_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES)
        ):
            frames.pop()
            if frames:
                frames[-1][4] = _add_bh_cells(int(frames[-1][4]), 1)
            else:
                return 1
            continue

        if current_type is not list and current_type is not tuple:
            raise ValueError("p_values must contain real numeric probabilities")

        identity = id(current)
        if not bool(frame[3]):
            if identity in active_container_ids:
                raise ValueError("p_values must not contain cyclic containers")
            if depth >= _MAX_PROBABILITY_TREE_DEPTH:
                raise ValueError("p_values nesting exceeds the supported depth")
            active_container_ids.add(identity)
            frame[3] = True

        child_index = int(frame[2])
        if child_index < len(current):
            frame[2] = child_index + 1
            structural_nodes += 1
            if structural_nodes > _MAX_BH_STRUCTURAL_NODES:
                raise ValueError(
                    "p_values exceeded structural traversal budget of "
                    f"{_MAX_BH_STRUCTURAL_NODES:,} nodes"
                )
            frames.append([current[child_index], depth + 1, 0, False, 0])
            continue

        cells = int(frame[4])
        active_container_ids.remove(identity)
        frames.pop()
        if frames:
            frames[-1][4] = _add_bh_cells(int(frames[-1][4]), cells)
        else:
            return cells

    return 0


def _trusted_probability_array(value: Any) -> np.ndarray:
    """Return package-owned bounded float64 p-values while preserving NaN missingness."""
    value_type = type(value)
    if value_type is np.ndarray:
        if int(value.size) > _MAX_BH_PROBABILITY_CELLS:
            raise ValueError(_bh_resource_error())
        array = value
    elif value_type is list or value_type is tuple:
        _trusted_probability_tree(value)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("p_values must contain real numeric probabilities") from error
    elif (
        value_type is bool
        or value_type is int
        or value_type is float
        or value_type is np.bool_
        or any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES)
        or any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES)
    ):
        array = np.asarray(value)
    else:
        raise ValueError("p_values must contain real numeric probabilities")

    if int(array.size) > _MAX_BH_PROBABILITY_CELLS:
        raise ValueError(_bh_resource_error())
    if np.iscomplexobj(array) or array.dtype.kind not in "biuf":
        raise ValueError("p_values must contain real numeric probabilities")
    finite = np.isfinite(array)
    if np.any(np.isinf(array)) or np.any(finite & ((array < 0) | (array > 1))):
        raise ValueError("p_values must be NaN or finite probabilities in [0, 1]")
    try:
        normalized = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("p_values must contain real numeric probabilities") from error
    if array.dtype.kind != "b":
        round_trip = normalized.astype(array.dtype, copy=False)
        if not np.array_equal(round_trip, array, equal_nan=True):
            raise ValueError("p_values must be exactly representable as float64")
    return normalized


def _bind_legacy_bh(function: Callable[..., Any]) -> None:
    """Rebind the historical package export to the hardened BH callable."""
    from . import _legacy_init

    if hasattr(_legacy_init, "benjamini_hochberg"):
        _legacy_init.benjamini_hochberg = function


def install(fitstats_module: ModuleType) -> None:
    """Install callback-free S-X² controls and BH evidence admission."""
    fitstats_module._validate_sx2_controls = _validate_sx2_controls

    current_bh: Callable[..., Any] = fitstats_module.benjamini_hochberg
    if getattr(current_bh, _BH_HARDENED_ATTR, False):
        _bind_legacy_bh(current_bh)
        return
    original_bh = current_bh

    @wraps(original_bh)
    def safe_benjamini_hochberg(p_values: Any, q: Any = 0.05) -> np.ndarray:
        """Validate BH threshold and probability evidence before Rust discovery."""
        q_value = _trusted_real(q, "q")
        if not np.isfinite(q_value) or not 0.0 < q_value <= 1.0:
            raise ValueError("q must be in (0, 1]")
        probabilities = _trusted_probability_array(p_values)
        output_shape = probabilities.shape
        flat_probabilities = np.ascontiguousarray(probabilities.ravel())
        result = original_bh(flat_probabilities, q_value)
        return np.asarray(result, dtype=bool).reshape(output_shape)

    setattr(safe_benjamini_hochberg, _BH_HARDENED_ATTR, True)
    fitstats_module.benjamini_hochberg = safe_benjamini_hochberg
    _bind_legacy_bh(safe_benjamini_hochberg)