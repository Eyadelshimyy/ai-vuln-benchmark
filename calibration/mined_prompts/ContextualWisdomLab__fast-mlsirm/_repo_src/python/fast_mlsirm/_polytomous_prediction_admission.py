"""Callback-free public evidence admission for GRM/GPCM predictions."""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType

import numpy as np

from .config import MAX_POLYTOMOUS_CATEGORIES


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
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_NUMPY_COMPLEX_SCALAR_TYPES = (np.complex64, np.complex128, np.clongdouble)
_TRUSTED_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    np.bool_,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
    *_NUMPY_COMPLEX_SCALAR_TYPES,
)
_MAX_PREDICTION_CELLS = 20_000_000
_MARKER = "__fast_mlsirm_polytomous_prediction_admission__"


def _integer_is_exact_float64(value: object) -> bool:
    """Return whether a trusted integer scalar survives float64 exactly."""

    integer = int(value)
    try:
        converted = float(integer)
    except OverflowError:
        return False
    return np.isfinite(converted) and int(converted) == integer


def _preflight(value: object, name: str, max_ndim: int) -> tuple[int, ...] | None:
    """Reject unsafe evidence before NumPy and return a proven rectangular shape."""

    error = f"{name} must be a trusted NumPy array or built-in sequence"
    rank_error = f"{name} must be at most {max_ndim}-D"
    resource_error = (
        f"{name} exceeds the {_MAX_PREDICTION_CELLS:,}-cell prediction evidence budget"
    )
    exact_error = f"{name} must be exactly representable as float64"
    active_container_ids: set[int] = set()
    subtree_metrics: dict[int, tuple[int, int, tuple[int, ...] | None]] = {}

    def visit(item: object, depth: int) -> tuple[int, int, tuple[int, ...] | None]:
        item_type = type(item)

        if item_type is np.ndarray:
            item_rank = int(item.ndim)
            if depth + item_rank > max_ndim:
                raise ValueError(rank_error)
            count = int(item.size)
            if count > _MAX_PREDICTION_CELLS:
                raise ValueError(resource_error)
            return count, item_rank, tuple(int(size) for size in item.shape)

        if any(item_type is scalar_type for scalar_type in _TRUSTED_SCALAR_TYPES):
            if depth > max_ndim:
                raise ValueError(rank_error)
            if (item_type is int or item_type in _NUMPY_INTEGER_SCALAR_TYPES) and not (
                _integer_is_exact_float64(item)
            ):
                raise ValueError(exact_error)
            return 1, 0, ()

        if item_type is not list and item_type is not tuple:
            raise ValueError(error)
        if depth + 1 > max_ndim:
            raise ValueError(rank_error)

        item_id = id(item)
        if item_id in active_container_ids:
            raise ValueError(error)
        cached = subtree_metrics.get(item_id)
        if cached is not None:
            count, subtree_rank, shape = cached
            if depth + subtree_rank > max_ndim:
                raise ValueError(rank_error)
            return count, subtree_rank, shape

        active_container_ids.add(item_id)
        total = 0
        max_child_rank = 0
        first_child_shape: tuple[int, ...] | None = None
        rectangular = True
        has_child = False
        try:
            for child in item:
                count, child_rank, child_shape = visit(child, depth + 1)
                if count > _MAX_PREDICTION_CELLS - total:
                    raise ValueError(resource_error)
                total += count
                max_child_rank = max(max_child_rank, child_rank)
                if not has_child:
                    first_child_shape = child_shape
                    has_child = True
                elif child_shape != first_child_shape:
                    rectangular = False
        finally:
            active_container_ids.remove(item_id)

        subtree_rank = 1 + max_child_rank
        if not has_child:
            shape: tuple[int, ...] | None = (0,)
        elif rectangular and first_child_shape is not None:
            shape = (len(item), *first_child_shape)
        elif rectangular and first_child_shape == ():
            shape = (len(item),)
        else:
            shape = None
        result = (total, subtree_rank, shape)
        subtree_metrics[item_id] = result
        return result

    _, _, shape = visit(value, 0)
    return shape


def _prediction_cells_from_shapes(
    theta_shape: tuple[int, ...] | None,
    slope_shape: tuple[int, ...] | None,
    cat_params_shape: tuple[int, ...] | None,
) -> int | None:
    """Return a provable joint output size without materializing evidence."""

    if (
        theta_shape is None
        or slope_shape is None
        or cat_params_shape is None
        or len(theta_shape) != 1
        or len(slope_shape) != 1
        or len(cat_params_shape) != 2
        or slope_shape[0] <= 0
        or cat_params_shape[0] != slope_shape[0]
        or cat_params_shape[1] < 1
    ):
        return None
    return int(theta_shape[0]) * int(slope_shape[0]) * (int(cat_params_shape[1]) + 1)


def _raise_if_invalid_category_shape(cat_params_shape: tuple[int, ...] | None) -> None:
    """Replay the fitted-model category ceiling from trusted shape metadata."""

    if cat_params_shape is None or len(cat_params_shape) != 2:
        return
    n_cat = int(cat_params_shape[1]) + 1
    if n_cat > MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(f"n_cat must be in 2..={MAX_POLYTOMOUS_CATEGORIES}")


def _raise_if_oversized_prediction_grid(cells: int | None) -> None:
    """Fail closed when a provable prediction grid exceeds the public budget."""

    if cells is not None and cells > _MAX_PREDICTION_CELLS:
        raise ValueError(
            f"prediction grid of {cells:,} cells exceeds the "
            f"{_MAX_PREDICTION_CELLS:,} prediction-cell limit"
        )


def _real_numeric_array(value: object, name: str) -> np.ndarray:
    """Materialize preflighted evidence and verify lossless float64 representation."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = raw.astype(np.float64, copy=False)
        round_tripped = converted.astype(raw.dtype, copy=False)
    if raw.dtype.kind == "f":
        lossless = np.array_equal(raw, round_tripped, equal_nan=True)
    else:
        lossless = np.array_equal(raw, round_tripped)
    if not lossless:
        raise ValueError(f"{name} must be exactly representable as float64")
    return converted


def install(module: ModuleType) -> None:
    """Install trusted prediction evidence admission on the polytomous module."""

    current = module._polytomous_predictions
    if getattr(current, _MARKER, False):
        return

    def guarded_predictions(fit, theta):
        if type(fit) is not module.PolytomousFit:
            raise TypeError("fit must be a PolytomousFit")

        # Establish every field's callback/rank/resource contract before any
        # one field reaches NumPy materialization. Shape metadata also makes
        # the fitted category domain and joint output budget decidable before
        # float64 copies are possible.
        theta_shape = _preflight(theta, "theta", 1)
        slope_shape = _preflight(fit.slope, "fit.slope", 1)
        cat_params_shape = _preflight(fit.cat_params, "fit.cat_params", 2)
        _raise_if_invalid_category_shape(cat_params_shape)
        _raise_if_oversized_prediction_grid(
            _prediction_cells_from_shapes(theta_shape, slope_shape, cat_params_shape)
        )

        trusted_theta = _real_numeric_array(theta, "theta")
        trusted_slope = _real_numeric_array(fit.slope, "fit.slope")
        trusted_cat_params = _real_numeric_array(fit.cat_params, "fit.cat_params")
        if not np.all(np.isfinite(trusted_theta)):
            raise ValueError("theta must be a non-empty finite 1-D array")
        if not np.all(np.isfinite(trusted_slope)) or not np.all(
            np.isfinite(trusted_cat_params)
        ):
            raise ValueError("fit item parameters must be finite")

        if (
            trusted_theta.ndim == 1
            and trusted_slope.ndim == 1
            and trusted_cat_params.ndim == 2
            and trusted_slope.size > 0
            and trusted_cat_params.shape[0] == trusted_slope.size
            and trusted_cat_params.shape[1] >= 1
        ):
            n_cat = int(trusted_cat_params.shape[1]) + 1
            cells = int(trusted_theta.size) * int(trusted_slope.size) * n_cat
            _raise_if_oversized_prediction_grid(cells)

        trusted_theta = np.ascontiguousarray(trusted_theta)
        trusted_slope = np.ascontiguousarray(trusted_slope)
        trusted_cat_params = np.ascontiguousarray(trusted_cat_params)
        trusted_fit = replace(
            fit,
            slope=trusted_slope,
            cat_params=trusted_cat_params,
        )
        return current(trusted_fit, trusted_theta)

    setattr(guarded_predictions, _MARKER, True)
    module._polytomous_predictions = guarded_predictions
