"""Explicit response-shape checks for cross-component IRT inputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, TypeVar

import numpy as np

from .config import MAX_POLYTOMOUS_CATEGORIES

IRTItemType = Literal["dichotomous", "polytomous"]
MIN_IRT_ITEMS = 2
MIN_IRT_PERSONS = 5
MIN_OBSERVED_PER_ITEM = 3
MIN_ITEM_DISTINCT_VALUES = 2
MIN_FACTOR_ANCHOR_ITEMS = 2
MAX_IRT_RESPONSE_CELLS = 20_000_000
_FitResultT = TypeVar("_FitResultT")
_TRUSTED_RESPONSE_SCALAR_TYPES = frozenset(
    {
        bool,
        int,
        float,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    }
)
_TRUSTED_NUMPY_INTEGER_SCALAR_TYPES = tuple(
    np.dtype(code).type
    for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "p", "P")
)


def _is_exact_numpy_integer_scalar_type(value_type: type) -> bool:
    """Return whether ``value_type`` is a package-trusted NumPy integer type."""
    return any(
        value_type is trusted_type
        for trusted_type in _TRUSTED_NUMPY_INTEGER_SCALAR_TYPES
    )


def _readiness_integer(value: object, name: str, minimum: int) -> int:
    """Normalize one inert readiness integer without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif _is_exact_numpy_integer_scalar_type(value_type):
        normalized = int(value)
    else:
        raise TypeError(f"{name} must be an integer >= {minimum}")
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _validate_item_contract(
    item_type: object,
    n_categories: object,
) -> None:
    """Validate IRT family/category semantics before caller-owned data work."""
    if type(item_type) is not str or item_type not in {"dichotomous", "polytomous"}:
        raise ValueError("item_type must be 'dichotomous' or 'polytomous'")
    if item_type == "dichotomous" and n_categories is not None:
        raise ValueError("n_categories is only valid for polytomous responses")
    if item_type == "polytomous" and (
        type(n_categories) is not int
        or not 2 <= n_categories <= MAX_POLYTOMOUS_CATEGORIES
    ):
        raise ValueError(
            "polytomous responses require n_categories in "
            f"2..{MAX_POLYTOMOUS_CATEGORIES}"
        )


def _preflight_numeric_tree(value: object, *, error: str) -> None:
    """Validate and bound built-in numeric trees without caller callbacks."""
    stack: list[tuple[object, int | None]] = [(value, None)]
    active_containers: set[int] = set()
    logical_cells = 0
    visited_nodes = 0
    structural_limit = 2 * MAX_IRT_RESPONSE_CELLS + 1
    while stack:
        current, next_index = stack.pop()
        current_type = type(current)
        if current_type in (list, tuple):
            identity = id(current)
            if next_index is None:
                visited_nodes += 1
                if visited_nodes > structural_limit:
                    raise ValueError(f"{error}; evidence exceeds the structural-work limit")
                if identity in active_containers:
                    raise ValueError(f"{error}; cyclic list/tuple containers are not supported")
                active_containers.add(identity)
                if len(current) == 0:
                    active_containers.remove(identity)
                else:
                    stack.append((current, 1))
                    stack.append((current[0], None))
            elif next_index < len(current):
                stack.append((current, next_index + 1))
                stack.append((current[next_index], None))
            else:
                active_containers.remove(identity)
            continue
        visited_nodes += 1
        if visited_nodes > structural_limit:
            raise ValueError(f"{error}; evidence exceeds the structural-work limit")
        if current_type is np.ndarray:
            if current.dtype.kind not in {"b", "i", "u", "f"}:
                raise ValueError(error)
            logical_cells += int(current.size)
        elif current_type in _TRUSTED_RESPONSE_SCALAR_TYPES:
            logical_cells += 1
        else:
            raise ValueError(error)
        if logical_cells > MAX_IRT_RESPONSE_CELLS:
            raise ValueError(
                f"{error}; evidence must contain at most {MAX_IRT_RESPONSE_CELLS:,} cells"
            )


def _validate_response_shape(source: np.ndarray) -> None:
    """Reject rank/resource-invalid response shapes before dense conversion."""
    if source.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items matrix")
    n_persons, n_items = source.shape
    if n_persons < 1:
        raise ValueError("responses must contain at least one person")
    if n_items < MIN_IRT_ITEMS:
        raise ValueError(
            "IRT responses must contain at least two item columns; "
            "a scalar or one-item result is not an IRT experiment"
        )
    if source.size > MAX_IRT_RESPONSE_CELLS:
        raise ValueError(
            f"responses must contain at most {MAX_IRT_RESPONSE_CELLS:,} cells"
        )


def _real_numeric_response_matrix(
    responses: Iterable[Iterable[float]] | np.ndarray,
) -> np.ndarray:
    """Marshal trusted real response storage without caller numeric callbacks."""
    if type(responses) is np.ndarray:
        source = responses
    elif type(responses) in (list, tuple):
        _preflight_numeric_tree(responses, error="responses must be a real numeric matrix")
        try:
            source = np.asarray(responses)
        except (TypeError, ValueError) as exc:
            raise ValueError("responses must be a real numeric matrix") from exc
    else:
        raise ValueError("responses must be a real numeric matrix")

    if source.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("responses must be a real numeric matrix")
    _validate_response_shape(source)
    try:
        return np.ascontiguousarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("responses must be a real numeric matrix") from exc


def _trusted_mask_matrix(mask: object, shape: tuple[int, ...]) -> np.ndarray:
    """Marshal trusted mask evidence without caller array/truth callbacks."""
    if type(mask) is np.ndarray:
        source = mask
    elif type(mask) in (list, tuple):
        _preflight_numeric_tree(mask, error="mask must be Boolean or numeric evidence")
        try:
            source = np.asarray(mask)
        except (TypeError, ValueError) as exc:
            raise ValueError("mask must be Boolean or numeric evidence") from exc
    else:
        raise ValueError("mask must be Boolean or numeric evidence")

    if source.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("mask must be Boolean or numeric evidence")
    if source.shape != shape:
        raise ValueError("mask shape must match responses")
    try:
        return np.ascontiguousarray(source, dtype=bool)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mask must be Boolean or numeric evidence") from exc


def validate_irt_response_matrix(
    responses: Iterable[Iterable[float]] | np.ndarray,
    item_type: IRTItemType,
    *,
    n_categories: int | None = None,
) -> np.ndarray:
    """Validate and return a persons-by-items matrix for an IRT experiment.

    This is the integration boundary for response data produced by another
    component, such as an LLM judge. It deliberately requires at least two
    item columns. Low-level one-item numerical primitives remain available for
    diagnostics and unit tests, but a one-item result is not an IRT experiment
    contract.

    NaN denotes a missing response. Dichotomous observations are 0/1;
    polytomous observations are integer category indices in
    0..n_categories-1.
    """
    _validate_item_contract(item_type, n_categories)

    matrix = _real_numeric_response_matrix(responses)
    if matrix.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items matrix")
    n_persons, n_items = matrix.shape
    if n_persons < 1:
        raise ValueError("responses must contain at least one person")
    if n_items < MIN_IRT_ITEMS:
        raise ValueError(
            "IRT responses must contain at least two item columns; "
            "a scalar or one-item result is not an IRT experiment"
        )

    missing = np.isnan(matrix)
    if np.any(~missing & ~np.isfinite(matrix)):
        raise ValueError("observed responses must be finite or NaN")
    observed = matrix[~missing]
    if observed.size and np.any(observed != np.floor(observed)):
        raise ValueError("observed responses must be integer category values")
    if item_type == "dichotomous":
        if observed.size and np.any((observed < 0) | (observed > 1)):
            raise ValueError("dichotomous responses must be 0, 1, or NaN")
    elif observed.size and np.any((observed < 0) | (observed >= int(n_categories))):
        raise ValueError(
            f"polytomous responses must be in 0..{int(n_categories) - 1} or NaN"
        )
    return matrix


def validate_irt_experiment_readiness(
    responses: Iterable[Iterable[float]] | np.ndarray,
    item_type: IRTItemType,
    *,
    n_categories: int | None = None,
    min_persons: int = MIN_IRT_PERSONS,
    min_observed_per_item: int = MIN_OBSERVED_PER_ITEM,
    min_item_distinct_values: int = MIN_ITEM_DISTINCT_VALUES,
    factor_ids: Sequence[object] | None = None,
    min_items_per_factor: int = MIN_FACTOR_ANCHOR_ITEMS,
) -> np.ndarray:
    """Validate response volume and information before IRT interpretation.

    This does not replace the shape contract in
    :func:`validate_irt_response_matrix`; it adds experiment-readiness checks
    used for production claims where tiny or one-dimensional response sets can
    produce unstable IRT estimates.
    """
    matrix = validate_irt_response_matrix(
        responses,
        item_type,
        n_categories=n_categories,
    )
    n_persons, n_items = matrix.shape

    min_persons = _readiness_integer(min_persons, "min_persons", 1)
    if n_persons < min_persons:
        raise ValueError(
            f"IRT experiment requires at least {min_persons} persons; "
            f"received {n_persons}"
        )

    min_observed_per_item = _readiness_integer(
        min_observed_per_item, "min_observed_per_item", 1
    )
    if min_observed_per_item > n_persons:
        raise ValueError(
            f"min_observed_per_item cannot exceed the number of persons ({n_persons})"
        )

    min_item_distinct_values = _readiness_integer(
        min_item_distinct_values, "min_item_distinct_values", 2
    )
    if item_type == "polytomous" and min_item_distinct_values > int(n_categories):
        raise ValueError(
            "min_item_distinct_values cannot exceed n_categories for polytomous responses"
        )

    observed = ~np.isnan(matrix)
    item_observed = np.asarray(observed.sum(axis=0), dtype=np.int64)
    weakly_observed_items = np.nonzero(item_observed < min_observed_per_item)[0].tolist()
    if weakly_observed_items:
        raise ValueError(
            "each IRT item must retain at least min_observed_per_item non-missing responses; "
            f"missing items: {weakly_observed_items}"
        )

    for item_index in range(n_items):
        col = matrix[:, item_index]
        observed_col = col[observed[:, item_index]]
        if observed_col.size == 0:
            raise ValueError("each item must have at least one observed response")
        unique_values = np.unique(observed_col)
        if unique_values.size < min_item_distinct_values:
            raise ValueError(
                "each IRT item must show at least "
                f"{min_item_distinct_values} distinct observed category values; "
                f"item {item_index} has {unique_values.size}"
            )
        if item_type == "polytomous":
            observed_categories = {int(value) for value in unique_values}
            missing_categories = [
                category
                for category in range(int(n_categories))
                if category not in observed_categories
            ]
            if missing_categories:
                raise ValueError(
                    "each polytomous item must observe every declared category; "
                    f"item {item_index} is missing categories {missing_categories}"
                )
    if factor_ids is not None:
        if isinstance(factor_ids, (str, bytes)) or not isinstance(
            factor_ids, (Sequence, np.ndarray)
        ):
            raise ValueError(
                "factor_ids must be a sequence of hashable factor labels or memberships"
            )
        if isinstance(factor_ids, np.ndarray) and factor_ids.ndim != 1:
            raise ValueError("factor_ids must be a one-dimensional sequence")
        if len(factor_ids) != n_items:
            raise ValueError(
                "factor_ids must contain one factor label for each item"
            )
        min_items_per_factor = _readiness_integer(
            min_items_per_factor, "min_items_per_factor", 1
        )
        labels = factor_ids.tolist() if isinstance(factor_ids, np.ndarray) else factor_ids

        def _memberships(label: object) -> tuple[object, ...]:
            """Normalize one item label into unique factor memberships."""
            if isinstance(label, np.ndarray):
                if label.ndim == 0:
                    values = (label.item(),)
                elif label.ndim == 1:
                    values = tuple(label.tolist())
                else:
                    raise ValueError(
                        "factor memberships must be one-dimensional sequences"
                    )
            elif isinstance(label, (list, tuple)) and not isinstance(label, (str, bytes)):
                values = tuple(label)
            else:
                values = (label,)
            if not values:
                raise ValueError("each item must have at least one factor membership")
            return tuple(dict.fromkeys(values))

        try:
            memberships = [_memberships(label) for label in labels]
            counts = Counter(
                factor for item_memberships in memberships for factor in item_memberships
            )
        except TypeError as exc:
            raise ValueError(
                "factor_ids must contain hashable factor labels or memberships"
            ) from exc
        low_coverage_factors = [
            str(factor)
            for factor, count in counts.items()
            if count < min_items_per_factor
        ]
        if low_coverage_factors:
            raise ValueError(
                "each factor anchor must appear on at least min_items_per_factor items; "
                f"under-covered factors: {', '.join(low_coverage_factors)}"
            )

    return matrix


def fit_irt_experiment(  # noqa: UP047  # PEP 695 syntax would break Python 3.10 support.
    fit_callable: Callable[..., _FitResultT],
    responses: Iterable[Iterable[float]] | np.ndarray,
    item_type: IRTItemType,
    *,
    n_categories: int | None = None,
    factor_ids: Sequence[object] | None = None,
    **fit_kwargs: object,
) -> _FitResultT:
    """Run a production IRT fit only after the readiness gate passes.

    Public numerical fitters intentionally remain usable for small or
    degenerate diagnostic fixtures. Production and benchmark callers must use
    this boundary so unstable estimates cannot be presented as experiment
    evidence. The callable receives the validated persons-by-items matrix as
    its first positional argument. A caller-supplied ``mask`` is consumed by
    this boundary; the normalized matrix already carries its missing-response
    semantics, so the raw mask is never forwarded to the numerical fitter.
    """
    _validate_item_contract(item_type, n_categories)
    mask = fit_kwargs.get("mask")
    normalized = _normalize_experiment_responses(responses, item_type, mask)
    matrix = validate_irt_experiment_readiness(
        normalized,
        item_type,
        n_categories=n_categories,
        factor_ids=factor_ids,
    )
    call_kwargs = dict(fit_kwargs)
    call_kwargs.pop("mask", None)
    return fit_callable(matrix, **call_kwargs)


def _normalize_experiment_responses(
    responses: Iterable[Iterable[float]] | np.ndarray,
    item_type: IRTItemType,
    mask: object | None,
) -> np.ndarray:
    """Apply public fitter missing-response semantics before readiness checks."""
    matrix = _real_numeric_response_matrix(responses)
    if mask is None:
        active = np.ones(matrix.shape, dtype=bool)
    else:
        active = _trusted_mask_matrix(mask, matrix.shape)
    if item_type == "dichotomous":
        observed = active & np.isfinite(matrix) & (matrix != -1)
    else:
        observed = active & np.isfinite(matrix) & (matrix >= 0)
    return np.where(observed, matrix, np.nan)


__all__ = [
    "IRTItemType",
    "MIN_FACTOR_ANCHOR_ITEMS",
    "MIN_IRT_ITEMS",
    "MIN_IRT_PERSONS",
    "MIN_ITEM_DISTINCT_VALUES",
    "MIN_OBSERVED_PER_ITEM",
    "fit_irt_experiment",
    "validate_irt_experiment_readiness",
    "validate_irt_response_matrix",
]
