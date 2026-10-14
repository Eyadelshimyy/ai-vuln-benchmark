"""Expected graded-item scores with caller-controlled factor integration.

Supports marginalizing one graded item's expected category score over a
caller-supplied subset of latent dimensions while the remaining coordinates
(including a plugged general factor) stay at fixed values.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

from .estimators.mmle import equal_probability_normal_nodes
from .polytomous import (
    PolytomousFit,
    _bounded_integer,
    predict_expected_response_polytomous,
)

__all__ = ["compute_expected_graded_item_score", "expected_graded_scale_score"]

_INTEGRATION_WEIGHT_TOLERANCE = 1e-12
# Evaluation block: grid points per Rust kernel call. This bounds working memory
# only; it never limits node counts or total work (see ``max_grid_points``).
# 65_536 points * MAX_POLYTOMOUS_CATEGORIES (64) = 4,194,304 prediction cells,
# below the 20,000,000-cell single-call limit in ``_polytomous_predictions``.
_GRID_BLOCK_POINTS = 65_536


def _validated_grid_budget(value) -> int:
    """Exact positive integer work budget (no upper bound; caller-owned)."""
    value_type = type(value)
    if value_type is not int and not (
        isinstance(value, np.integer) and not isinstance(value, np.bool_)
    ):
        raise ValueError("max_grid_points must be a positive integer")
    budget = int(value)
    if budget < 1:
        raise ValueError("max_grid_points must be a positive integer")
    return budget


def _validated_node_count(value) -> int:
    """Exact positive node count, same contract as ``equal_probability_normal_nodes``.

    Converted to a Python ``int`` *before* any power is taken so NumPy integer
    scalars cannot wrap (``np.int64(2**32) ** 2 == 0``).
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("n_nodes must be a positive integer")
    count = int(value)
    if count < 1:
        raise ValueError("n_nodes must be >= 1")
    return count


def _admit_grid_work(work: int, budget: int, detail: str) -> None:
    """Fail closed when exact Cartesian work exceeds the caller's budget."""
    if work > budget:
        try:
            message = (
                f"Cartesian quadrature grid has {work} points ({detail}), exceeding "
                f"max_grid_points={budget}; pass a larger max_grid_points to admit this work"
            )
        except ValueError:
            # Decimal conversion may exceed Python's integer-string safety limit.
            # Keep exact admission unchanged and never relax the process-wide limit.
            message = "Cartesian quadrature grid exceeds max_grid_points; integer counts omitted"
        raise ValueError(message) from None


def _grm_expected_scores(thresholds: np.ndarray, linear_predictors: np.ndarray) -> np.ndarray:
    """Return ``E[Y | eta]`` for one Samejima GRM item at each predictor.

    Delegates to the compiled Rust prediction kernel through a unit-slope
    item, so ``eta`` enters as the kernel's ``theta`` and the Rust core owns
    threshold validation (strictly decreasing, finite).
    """
    cell = PolytomousFit(
        model="grm",
        slope=np.ones(1, dtype=np.float64),
        cat_params=thresholds[None, :],
        loglik=float("nan"),
        n_iter=0,
        converged=True,
        termination_reason="marginalized",
    )
    return predict_expected_response_polytomous(cell, linear_predictors)[:, 0]


def _validated_columns(integrate_columns: Sequence[int], n_dims: int) -> list[int]:
    """Exact integer column indices in ``[0, n_dims)`` with no duplicates."""
    columns = [
        _bounded_integer(column, "integrate_columns entries", 0, n_dims - 1)
        for column in integrate_columns
    ]
    if len(columns) != len(set(columns)):
        raise ValueError("integrate_columns must not contain duplicates")
    return columns


def _validate_integration_axis_weights(weights: np.ndarray, axis_index: int) -> None:
    """Reject integration weights that are not a probability measure on one axis."""
    label = f"integration_weights[{axis_index}]"
    if np.any(weights < 0.0):
        raise ValueError(f"{label} must be non-negative")
    try:
        weight_sum = math.fsum(float(weight) for weight in weights)
    except OverflowError:
        # fsum raises instead of returning inf when finite weights overflow.
        raise ValueError(f"{label} must sum to one") from None
    if not math.isclose(
        weight_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=_INTEGRATION_WEIGHT_TOLERANCE,
    ):
        raise ValueError(f"{label} must sum to one within tolerance")


def compute_expected_graded_item_score(
    slope: np.ndarray,
    thresholds: np.ndarray,
    theta: np.ndarray,
    integrate_columns: Sequence[int],
    integration_nodes: Sequence[np.ndarray],
    integration_weights: Sequence[np.ndarray],
    *,
    max_grid_points: int,
) -> float:
    """Return the expected category score of one graded item after integration.

    The compensatory linear predictor is ``eta = slope @ theta_eff``. Entries
    of ``theta`` that are not listed in ``integrate_columns`` remain fixed
    (for example a plugged general factor). Each integrated column is replaced
    in turn by the corresponding caller-supplied nodes; when more than one
    column is integrated, the quadrature runs on the Cartesian product with
    product weights. Each integrated axis carries its own probability measure:
    weights must be finite, non-negative, and sum to one within a tight
    absolute tolerance; invalid axes fail closed with no silent renormalization.
    Nodes are used as given — the caller is responsible for any prior scaling.

    Category probabilities follow Samejima's (1969) graded response model:
    each category is the difference of adjacent cumulative boundary curves
    (Samejima, 1969, p. 20, Eq. 4-4), here in the logistic form of the
    boundary curve (Samejima, 1969, p. 32, Eq. 5-36) with the scaling factor
    absorbed into the slope, ``P(Y >= k | eta) = sigmoid(eta + beta_k)``,
    evaluated by the compiled Rust core. The integration itself is the
    caller-defined discrete quadrature described above.

    Parameters
    ----------
    slope:
        Item discrimination vector, one entry per latent dimension.
    thresholds:
        ``K - 1`` strictly ordered cumulative boundary intercepts for the
        graded item.
    theta:
        Fixed latent vector with non-integrated coordinates already set
        (including the general factor when it is not integrated).
    integrate_columns:
        Column indices to replace with quadrature nodes.
    integration_nodes:
        One finite 1-D node array per integrated column, in the same order
        as ``integrate_columns``.
    integration_weights:
        One finite 1-D weight array per integrated column, aligned with
        ``integration_nodes``. Each array is an independent discrete
        probability measure over that axis (non-negative entries summing to
        one). Multi-axis integration uses the product measure on the Cartesian
        product of node grids.
    max_grid_points:
        Required caller-owned work budget: the exact Cartesian grid size
        ``prod(len(nodes) for nodes in integration_nodes)`` (1 when nothing is
        integrated) must not exceed it. Checked with exact integer arithmetic
        before any grid-sized allocation; node counts are never capped. The
        grid is evaluated in fixed-size blocks, so no grid-sized array is
        allocated; block partial sums are streamed to ``math.fsum`` rather
        than retained in a list. Run time grows with the grid size.

    Returns
    -------
    float
        The weighted expected category score ``sum_k k * P(Y = k)`` after
        integrating over the requested columns.

    References
    ----------
    Samejima, F. (1969). Estimation of latent ability using a response pattern
    of graded scores. *Psychometrika, 34*(S1), 1-97.
    https://doi.org/10.1007/BF03372160
    """
    slope_arr = np.asarray(slope, dtype=np.float64)
    if slope_arr.ndim != 1 or slope_arr.size == 0:
        raise ValueError("slope must be a non-empty 1-D array")
    if not np.all(np.isfinite(slope_arr)):
        raise ValueError("slope must be finite")

    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1 or theta_arr.shape != slope_arr.shape:
        raise ValueError("theta must be a 1-D array with the same length as slope")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must be finite")

    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    if thresholds_arr.ndim != 1 or thresholds_arr.size < 1:
        raise ValueError("thresholds must be a 1-D array of length K-1 >= 1")
    if not np.all(np.isfinite(thresholds_arr)):
        raise ValueError("thresholds must be finite")

    columns = _validated_columns(integrate_columns, slope_arr.size)

    if len(columns) != len(integration_nodes) or len(columns) != len(integration_weights):
        raise ValueError(
            "integration_nodes and integration_weights must align with integrate_columns"
        )

    budget = _validated_grid_budget(max_grid_points)

    node_arrays: list[np.ndarray] = []
    weight_arrays: list[np.ndarray] = []
    for index, column in enumerate(columns):
        nodes = np.asarray(integration_nodes[index], dtype=np.float64)
        weights = np.asarray(integration_weights[index], dtype=np.float64)
        if nodes.ndim != 1 or nodes.size == 0:
            raise ValueError(f"integration_nodes[{index}] must be a non-empty 1-D array")
        if weights.ndim != 1 or weights.shape != nodes.shape:
            raise ValueError(
                f"integration_weights[{index}] must be a 1-D array matching its nodes"
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError("integration nodes must be finite")
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"integration_weights[{index}] must be finite")
        _validate_integration_axis_weights(weights, index)
        node_arrays.append(nodes)
        weight_arrays.append(weights)

    sizes = [int(nodes.size) for nodes in node_arrays]
    _admit_grid_work(math.prod(sizes), budget, f"per-axis node counts {sizes}")

    fixed_theta = theta_arr.copy()
    fixed_theta[columns] = 0.0
    base = float(slope_arr @ fixed_theta)
    axis_eta = [slope_arr[column] * node_arrays[axis] for axis, column in enumerate(columns)]
    return _blocked_expected_score(thresholds_arr, base, axis_eta, weight_arrays)


def _blocked_expected_score(
    thresholds: np.ndarray,
    base: float,
    axis_eta: list[np.ndarray],
    axis_weights: list[np.ndarray],
) -> float:
    """Product-measure expectation evaluated in blocks of at most ``block`` points.

    Axes split into iterated outer axes, one sliced axis, and a trailing
    vectorized suffix whose size is at most ``block``; each kernel call sees at
    most ``block`` points, so memory is O(block + sum of axis sizes). Partial
    sums are combined with ``math.fsum``. A grid that fits in one block takes
    the same single-call path (same summation order) as a full broadcast.
    """
    block = _GRID_BLOCK_POINTS
    sizes = [values.size for values in axis_eta]
    n_axes = len(sizes)
    start, suffix_points = n_axes, 1
    while start > 0 and suffix_points * sizes[start - 1] <= block:
        start -= 1
        suffix_points *= sizes[start]
    # Seed with ``base`` only for a single-block grid so its additions match
    # the historical broadcast order bit for bit.
    suffix_eta = np.asarray(base if start == 0 else 0.0)
    suffix_weight = np.ones((), dtype=np.float64)
    for offset, axis in enumerate(range(start, n_axes)):
        shape = [1] * (n_axes - start)
        shape[offset] = -1
        suffix_eta = suffix_eta + axis_eta[axis].reshape(shape)
        suffix_weight = suffix_weight * axis_weights[axis].reshape(shape)
    suffix_shape = tuple(sizes[start:])
    suffix_eta = np.broadcast_to(suffix_eta, suffix_shape).reshape(-1)
    suffix_weight = np.broadcast_to(suffix_weight, suffix_shape).reshape(-1)
    if start == 0:
        expected = _grm_expected_scores(thresholds, suffix_eta)
        return float(np.dot(suffix_weight, expected))

    split = start - 1
    run = block // suffix_points
    def partials():
        for outer in itertools.product(*(range(size) for size in sizes[:split])):
            shift = base
            outer_weight = 1.0
            for axis, position in enumerate(outer):
                shift += float(axis_eta[axis][position])
                outer_weight *= float(axis_weights[axis][position])
            for low in range(0, sizes[split], run):
                high = min(low + run, sizes[split])
                eta = (shift + axis_eta[split][low:high])[:, None] + suffix_eta[None, :]
                weight = axis_weights[split][low:high, None] * suffix_weight[None, :]
                expected = _grm_expected_scores(thresholds, eta.reshape(-1))
                yield outer_weight * float(np.dot(weight.reshape(-1), expected))

    return math.fsum(partials())


def expected_graded_scale_score(
    slopes: np.ndarray,
    thresholds: np.ndarray,
    theta: np.ndarray,
    integrate_columns: Sequence[Sequence[int]],
    factor_variances: np.ndarray,
    *,
    n_nodes: int,
    max_grid_points: int,
) -> float:
    """Return the expected total graded scale score ``E[T | G]`` after integration.

    Builds equal-probability standard-normal quadrature with
    :func:`~fast_mlsirm.estimators.mmle.equal_probability_normal_nodes`,
    scales integrated dimensions by ``sqrt(factor_variances[d])`` (mean zero),
    and sums per-item expectations from
    :func:`~fast_mlsirm.graded_item.compute_expected_graded_item_score`.
    Non-integrated coordinates of ``theta`` (for example a plugged general
    factor) remain fixed at caller-supplied values.

    Parameters
    ----------
    slopes:
        ``(n_items, n_dims)`` item discrimination matrix.
    thresholds:
        ``(n_items, K - 1)`` cumulative boundary intercepts; every row must
        have the same number of thresholds.
    theta:
        Fixed latent vector with non-integrated coordinates already set.
    integrate_columns:
        Per-item column indices to replace with scaled quadrature nodes.
    factor_variances:
        ``(n_dims,)`` latent factor **variances** (not standard deviations).
        Integrated columns use nodes ``sqrt(factor_variances[d]) * z`` where
        ``z`` are the standard-normal equal-probability nodes.
    n_nodes:
        Number of equal-probability quadrature nodes per integrated dimension.
        Required with no default; callers choose precision for their study.
    max_grid_points:
        Required caller-owned work budget for the whole scale: the exact total
        ``sum_i n_nodes ** len(integrate_columns[i])`` grid points must not
        exceed it. Checked with exact Python integers after every item's
        columns are validated and before any quadrature node is constructed or
        any item is evaluated. Node counts are never capped by the package.

    Returns
    -------
    float
        ``sum_i E[Y_i | G]`` after integrating the requested columns for
        each item.

    References
    ----------
    Samejima, F. (1969). Estimation of latent ability using a response pattern
    of graded scores. *Psychometrika, 34*(S1), 1-97.
    https://doi.org/10.1007/BF03372160

    Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the
    normal distribution. *Applied Statistics, 37*(3), 477-484.
    https://doi.org/10.2307/2347330
    """
    slopes_arr = np.asarray(slopes, dtype=np.float64)
    if slopes_arr.ndim != 2 or slopes_arr.shape[0] == 0:
        raise ValueError("slopes must be a non-empty 2-D array")
    if not np.all(np.isfinite(slopes_arr)):
        raise ValueError("slopes must be finite")

    n_items, n_dims = slopes_arr.shape

    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1 or theta_arr.shape != (n_dims,):
        raise ValueError("theta must be a 1-D array with length n_dims")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must be finite")

    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    if thresholds_arr.ndim != 2 or thresholds_arr.shape[0] != n_items:
        raise ValueError("thresholds must be a 2-D array with shape (n_items, K-1)")
    if thresholds_arr.shape[1] < 1:
        raise ValueError("thresholds must have at least one boundary per item")
    if not np.all(np.isfinite(thresholds_arr)):
        raise ValueError("thresholds must be finite")

    variances = np.asarray(factor_variances, dtype=np.float64)
    if variances.ndim != 1 or variances.shape != (n_dims,):
        raise ValueError("factor_variances must be a 1-D array with length n_dims")
    if not np.all(np.isfinite(variances)):
        raise ValueError("factor_variances must be finite")
    if np.any(variances < 0.0):
        raise ValueError("factor_variances must be non-negative")

    if len(integrate_columns) != n_items:
        raise ValueError("integrate_columns must have one entry per item")

    item_columns = [_validated_columns(columns, n_dims) for columns in integrate_columns]
    node_count = _validated_node_count(n_nodes)
    budget = _validated_grid_budget(max_grid_points)
    item_work = [node_count ** len(columns) for columns in item_columns]
    _admit_grid_work(
        sum(item_work),
        budget,
        "sum over items of n_nodes ** len(integrate_columns[i])",
    )

    # Node construction is O(n_nodes); it is covered by the admitted work
    # because n_nodes <= sum(item_work) whenever any item integrates. When no
    # item integrates, no nodes are built at all.
    scaled_by_column: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if any(item_columns):
        base_nodes, base_weights = equal_probability_normal_nodes(node_count)
        for column in sorted({column for columns in item_columns for column in columns}):
            scale = float(np.sqrt(variances[column]))
            scaled_by_column[column] = (base_nodes * scale, base_weights.copy())

    total = 0.0
    for item_index, columns in enumerate(item_columns):
        item_nodes = tuple(scaled_by_column[column][0] for column in columns)
        item_weights = tuple(scaled_by_column[column][1] for column in columns)
        total += compute_expected_graded_item_score(
            slopes_arr[item_index],
            thresholds_arr[item_index],
            theta_arr,
            columns,
            item_nodes,
            item_weights,
            max_grid_points=item_work[item_index],
        )
    return total
