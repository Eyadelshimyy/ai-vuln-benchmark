"""Rating Scale Model (Andrich, 1978): a Rasch-family polytomous model whose
category thresholds are shared across items, estimated by marginal-ML EM in the
Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES
from .irt_contract import MIN_IRT_ITEMS, validate_irt_response_matrix


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
_TRUSTED_RESPONSE_SCALAR_TYPES = (
    bool,
    np.bool_,
    int,
    float,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
)
_MAX_RSM_RESPONSE_CELLS = 20_000_000
_MAX_RSM_RESPONSE_STRUCTURAL_NODES = 2 * _MAX_RSM_RESPONSE_CELLS
_RSM_RESOURCE_ERROR = (
    f"responses exceed the {_MAX_RSM_RESPONSE_CELLS}-cell RSM evidence budget"
)
_RSM_MATRIX_SHAPE_ERROR = "responses must be a 2-D persons x items array"
_RSM_MINIMUM_SHAPE_ERROR = (
    "responses must contain at least one person and at least two item columns"
)


def _rsm_structural_resource_error() -> str:
    """Return the current structural-work diagnostic, including test overrides."""

    return (
        "responses exceed the "
        f"{_MAX_RSM_RESPONSE_STRUCTURAL_NODES}-node RSM structural evidence budget"
    )


@dataclass
class RsmFit:
    """Fitted rating scale model (Andrich, 1978).

    ``item_location`` is the per-item location ``delta_i``; ``thresholds`` the
    ``n_cat-1`` common category thresholds ``tau_k`` (shared across all items,
    centered so they sum to 0); ``theta`` the per-person EAP trait. The
    adjacent-category log-odds are ``ln[P(k)/P(k-1)] = theta - delta_i - tau_k``."""

    item_location: np.ndarray
    thresholds: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int


def _trusted_optional_category_count(value: int | None) -> int | None:
    """Admit the established built-in category-count contract without callbacks."""
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("n_cat must be an integer >= 2")
    if not 2 <= value <= MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(f"n_cat must be an integer in 2..{MAX_POLYTOMOUS_CATEGORIES}")
    return value


def _trusted_quadrature_points(value: int) -> int:
    """Return an established RSM quadrature size without caller hash/coercion hooks."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise ValueError("q_theta must be an integer >= 1")
    # #1929: no node-count cap; the Rust core generates any n >= 1 rule
    # on demand (Golub & Welsch, 1969) and guards allocation overflow.
    if normalized < 1:
        raise ValueError("q_theta must be >= 1")
    return normalized


def _trusted_iteration_cap(value: int) -> int:
    """Admit the established built-in iteration cap without caller callbacks."""
    if type(value) is not int or not 1 <= value <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be an integer in 1..{MAX_MAX_ITER}")
    return value


def _trusted_positive_tolerance(value: float) -> float:
    """Return a finite positive value that survives the Rust ``f64`` boundary exactly."""
    error = "tol must be finite and > 0"
    value_type = type(value)
    try:
        if value_type is int:
            normalized = float(value)
            if not np.isfinite(normalized) or int(normalized) != value:
                raise ValueError(error)
        elif value_type is float:
            normalized = value
        elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            normalized = float(value)
            if not np.isfinite(normalized) or int(normalized) != int(value):
                raise ValueError(error)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES):
            normalized = float(value)
            if not np.isfinite(normalized) or value_type(normalized) != value:
                raise ValueError(error)
        else:
            raise ValueError(error)
    except (OverflowError, ValueError):
        raise ValueError(error) from None
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(error)
    return normalized


def _trusted_response_source(value: object) -> object:
    """Admit inert, bounded response containers before NumPy invokes protocols."""
    if type(value) is np.ndarray:
        if int(value.size) > _MAX_RSM_RESPONSE_CELLS:
            raise ValueError(_RSM_RESOURCE_ERROR)
        if value.ndim != 2:
            raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
        if int(value.shape[0]) < 1 or int(value.shape[1]) < MIN_IRT_ITEMS:
            raise ValueError(_RSM_MINIMUM_SHAPE_ERROR)
        return value
    if type(value) is not list and type(value) is not tuple:
        raise ValueError("responses must be a real numeric array")

    logical_cells = 0
    structural_nodes = 0
    matrix_like = len(value) > 0
    matrix_width: int | None = None
    for row in value:
        structural_nodes += 1
        if structural_nodes > _MAX_RSM_RESPONSE_STRUCTURAL_NODES:
            raise ValueError(_rsm_structural_resource_error())
        row_type = type(row)
        if row_type is np.ndarray:
            logical_cells += int(row.size)
            if logical_cells > _MAX_RSM_RESPONSE_CELLS:
                raise ValueError(_RSM_RESOURCE_ERROR)
            if row.ndim != 1:
                matrix_like = False
            else:
                row_width = int(row.shape[0])
                if matrix_width is None:
                    matrix_width = row_width
                elif matrix_width != row_width:
                    matrix_like = False
            continue
        if row_type is list or row_type is tuple:
            row_length = len(row)
            logical_cells += row_length
            if logical_cells > _MAX_RSM_RESPONSE_CELLS:
                raise ValueError(_RSM_RESOURCE_ERROR)
            structural_nodes += row_length
            if structural_nodes > _MAX_RSM_RESPONSE_STRUCTURAL_NODES:
                raise ValueError(_rsm_structural_resource_error())
            if matrix_width is None:
                matrix_width = row_length
            elif matrix_width != row_length:
                matrix_like = False
            continue
        # Flat built-in scalar sequences are known not to satisfy the established
        # persons-by-items matrix contract, but still charge their bounded work
        # before the stable 2-D diagnostic is replayed below.
        matrix_like = False
        if row_type not in _TRUSTED_RESPONSE_SCALAR_TYPES:
            raise ValueError("responses must be a real numeric array")
        logical_cells += 1
        if logical_cells > _MAX_RSM_RESPONSE_CELLS:
            raise ValueError(_RSM_RESOURCE_ERROR)

    if not matrix_like:
        raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
    if matrix_width is not None and matrix_width < MIN_IRT_ITEMS:
        raise ValueError(_RSM_MINIMUM_SHAPE_ERROR)

    # Value-wise validation is intentionally deferred until resource, matrix-rank,
    # rectangularity, and structurally impossible item-count checks have completed.
    for row in value:
        if type(row) is list or type(row) is tuple:
            if any(type(cell) not in _TRUSTED_RESPONSE_SCALAR_TYPES for cell in row):
                raise ValueError("responses must be a real numeric array")
    return value


def _real_numeric_response_matrix(value: object) -> np.ndarray:
    """Admit bounded real numeric response storage before ``float64`` marshalling."""
    source = _trusted_response_source(value)
    array = np.asarray(source)
    if np.iscomplexobj(array) or array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("responses must be a real numeric array")
    return np.ascontiguousarray(array, dtype=np.float64)


def fit_rsm(
    responses: np.ndarray,
    n_cat: int | None = None,
    *,
    q_theta: int,
    max_iter: int,
    tol: float,
) -> RsmFit:
    """Fit the rating scale model (compute in Rust; Andrich, 1978).

    The RSM is the Rasch-family polytomous model for items on a common rating scale
    (e.g. Likert): every item has its own location ``delta_i``, but the ``K-1``
    category thresholds ``tau_k`` are *shared across all items*. The
    adjacent-category log-odds are ``ln[P(X=k)/P(X=k-1)] = theta - delta_i - tau_k``,
    ``theta ~ N(0,1)``. This is a constrained partial-credit model (the PCM has
    item-specific thresholds); at ``K=2`` it reduces to the Rasch model. Estimated by
    marginal-ML EM with a Gauss-Hermite trait grid; the item locations and the shared
    thresholds are updated by a monotone ECM step and the thresholds are centered to
    sum to zero.

    ``responses`` is a persons x items array of integer category indices
    ``0..n_cat-1`` with at least two item columns (``NaN`` marks a missing cell,
    dropped under a missing-at-random assumption). ``n_cat`` defaults to
    ``max(responses) + 1``.

    References (APA 7th ed.):
        Andrich, D. (1978). A rating formulation for ordered response categories.
            *Psychometrika, 43*(4), 561-573. https://doi.org/10.1007/BF02293814
    """
    n_cat = _trusted_optional_category_count(n_cat)
    q_theta = _trusted_quadrature_points(q_theta)
    max_iter = _trusted_iteration_cap(max_iter)
    tol = _trusted_positive_tolerance(tol)

    y = _real_numeric_response_matrix(responses)
    if y.ndim != 2:
        raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
    n_persons, n_items = y.shape
    if n_persons < 1 or n_items < MIN_IRT_ITEMS:
        raise ValueError(_RSM_MINIMUM_SHAPE_ERROR)
    missing = np.isnan(y)
    if np.any(~missing & ~np.isfinite(y)):
        raise ValueError("observed responses must be finite integer categories")
    observed = ~missing
    obs_values = y[observed]
    if obs_values.size and (
        np.any(obs_values != np.floor(obs_values)) or np.any(obs_values < 0)
    ):
        raise ValueError("observed responses must be non-negative integer categories")
    if n_cat is None:
        if obs_values.size == 0:
            raise ValueError("responses has no observed values")
        n_cat = int(obs_values.max()) + 1
        if n_cat < 2:
            raise ValueError("responses must contain at least two categories")
        if n_cat > MAX_POLYTOMOUS_CATEGORIES:
            raise ValueError(
                f"responses imply more than {MAX_POLYTOMOUS_CATEGORIES} categories"
            )
    if obs_values.size and np.any(obs_values >= n_cat):
        raise ValueError(
            f"observed responses must be integer categories in 0..{n_cat - 1}"
        )
    missing_items = np.flatnonzero(~observed.any(axis=0))
    if missing_items.size:
        raise ValueError(f"item {int(missing_items[0])} has no observed responses")
    validate_irt_response_matrix(y, "polytomous", n_categories=n_cat)
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rsm"):
        raise RuntimeError("fit_rsm requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.fit_rsm(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        n_cat,
        q_theta,
        max_iter,
        tol,
    )
    return RsmFit(
        item_location=np.asarray(res["item_location"], dtype=np.float64),
        thresholds=np.asarray(res["thresholds"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
    )
