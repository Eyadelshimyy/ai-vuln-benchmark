"""Cartesian work admission for graded-item expected scores (PR #2062 review).

Node counts are never capped or defaulted (AGENTS.md maintainer steering 1);
the caller owns an explicit ``max_grid_points`` work budget that is checked
with exact Python integers before any grid-sized allocation, and evaluation is
blocked so memory does not scale with ``n ** d``.
"""

from __future__ import annotations

import math
import tracemalloc

import numpy as np
import pytest

import fast_mlsirm.graded_item as graded_item
from fast_mlsirm.estimators.mmle import equal_probability_normal_nodes
from fast_mlsirm.graded_item import (
    compute_expected_graded_item_score,
    expected_graded_scale_score,
)

SLOPE = np.array([1.1, 0.8, 0.6, 0.5])
THETA = np.array([0.3, 0.0, 0.0, 0.0])
THRESHOLDS = np.array([1.05, 0.10, -0.70])


def _axes(n_nodes: int, n_axes: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    nodes, weights = equal_probability_normal_nodes(n_nodes)
    return [nodes * (1.0 + 0.1 * axis) for axis in range(n_axes)], [weights] * n_axes


def _forbid_kernel(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def unexpected(thresholds, linear_predictors):
        calls.append(int(np.asarray(linear_predictors).size))
        raise AssertionError("kernel must not run before work admission")

    monkeypatch.setattr(graded_item, "_grm_expected_scores", unexpected)
    return calls


def _decomposed_reference(thresholds, columns, nodes, weights, *, budget) -> float:
    """sum_j w_j E[Y | theta_c0 = z_j] using only grids of size n ** (d - 1)."""
    terms = []
    for node, weight in zip(nodes[0], weights[0], strict=True):
        theta = THETA.copy()
        theta[columns[0]] = node
        terms.append(
            weight
            * compute_expected_graded_item_score(
                SLOPE, thresholds, theta, columns[1:], nodes[1:], weights[1:],
                max_grid_points=budget,
            )
        )
    return math.fsum(terms)


# --- item helper: required caller-owned budget --------------------------------


def test_item_score_requires_max_grid_points_keyword() -> None:
    nodes, weights = _axes(3, 1)
    with pytest.raises(TypeError, match="max_grid_points"):
        compute_expected_graded_item_score(SLOPE, THRESHOLDS, THETA, (1,), nodes, weights)


@pytest.mark.parametrize("budget", [0, -1, True, np.bool_(True), 2.0, "9", None])
def test_item_score_rejects_invalid_budget(budget) -> None:
    nodes, weights = _axes(3, 1)
    with pytest.raises(ValueError, match="max_grid_points must be a positive integer"):
        compute_expected_graded_item_score(
            SLOPE, THRESHOLDS, THETA, (1,), nodes, weights, max_grid_points=budget
        )


def test_item_score_budget_is_inclusive_and_accepts_numpy_integer() -> None:
    nodes, weights = _axes(3, 2)
    exact = compute_expected_graded_item_score(
        SLOPE, THRESHOLDS, THETA, (1, 2), nodes, weights, max_grid_points=9
    )
    numpy_budget = compute_expected_graded_item_score(
        SLOPE, THRESHOLDS, THETA, (1, 2), nodes, weights, max_grid_points=np.int64(9)
    )
    assert exact == numpy_budget
    with pytest.raises(ValueError, match=r"grid has 9 points .* exceeding max_grid_points=8"):
        compute_expected_graded_item_score(
            SLOPE, THRESHOLDS, THETA, (1, 2), nodes, weights, max_grid_points=8
        )


def test_item_score_rejects_over_budget_before_kernel(monkeypatch) -> None:
    calls = _forbid_kernel(monkeypatch)
    nodes, weights = _axes(241, 3)
    with pytest.raises(
        ValueError,
        match=r"grid has 13997521 points \(per-axis node counts \[241, 241, 241\]\), "
        r"exceeding max_grid_points=13997520",
    ):
        compute_expected_graded_item_score(
            SLOPE, THRESHOLDS, THETA, (1, 2, 3), nodes, weights,
            max_grid_points=241**3 - 1,
        )
    assert calls == []


def test_item_score_work_count_is_exact_beyond_int64() -> None:
    """10**20 points: exact Python int, no int64 wrap, no allocation, instant reject."""
    nodes = [np.broadcast_to(np.float64(0.0), (10**5,))] * 4
    weights = [np.broadcast_to(np.float64(1e-5), (10**5,))] * 4
    slope = np.array([1.0, 0.5, 0.5, 0.5, 0.5])
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=r"grid has 100000000000000000000 points"):
            compute_expected_graded_item_score(
                slope, THRESHOLDS, np.zeros(5), (1, 2, 3, 4), nodes, weights,
                max_grid_points=2**63 - 1,
            )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 64 * 1024 * 1024


# --- item helper: blocked evaluation (no allocation in n ** d) ---------------


def test_three_axes_121_nodes_twelve_categories_is_admitted_and_exact() -> None:
    """HEAD: 121**3 * 12 = 21,258,732 cells trips the kernel's 20M single-call limit."""
    thresholds = np.linspace(2.5, -2.5, 11)
    nodes, weights = _axes(121, 3)
    result = compute_expected_graded_item_score(
        SLOPE, thresholds, THETA, (1, 2, 3), nodes, weights, max_grid_points=121**3
    )
    reference = _decomposed_reference(thresholds, (1, 2, 3), nodes, weights, budget=121**2)
    assert result == pytest.approx(reference, rel=1e-12)


def test_kernel_calls_are_blocked_independent_of_grid_size(monkeypatch) -> None:
    sizes: list[int] = []
    original = graded_item._grm_expected_scores

    def recording(thresholds, linear_predictors):
        sizes.append(int(np.asarray(linear_predictors).size))
        return original(thresholds, linear_predictors)

    monkeypatch.setattr(graded_item, "_grm_expected_scores", recording)
    nodes, weights = _axes(121, 3)
    compute_expected_graded_item_score(
        SLOPE, THRESHOLDS, THETA, (1, 2, 3), nodes, weights, max_grid_points=121**3
    )
    assert sum(sizes) == 121**3
    assert len(sizes) > 1
    assert max(sizes) <= graded_item._GRID_BLOCK_POINTS


def test_peak_traced_memory_does_not_scale_with_grid() -> None:
    nodes, weights = _axes(121, 3)
    tracemalloc.start()
    try:
        compute_expected_graded_item_score(
            SLOPE, THRESHOLDS, THETA, (1, 2, 3), nodes, weights, max_grid_points=121**3
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # HEAD materializes a 121**3 float64 grid (~13.5 MiB) plus weights (~28.7 MiB peak).
    assert peak < 8 * 1024 * 1024


@pytest.mark.parametrize(
    ("n_nodes", "n_axes"), [(241, 1), (121, 2), (241, 2), (61, 3), (41, 3)]
)
def test_blocked_matches_unblocked_reference(n_nodes: int, n_axes: int) -> None:
    nodes, weights = _axes(n_nodes, n_axes)
    columns = tuple(range(1, n_axes + 1))
    result = compute_expected_graded_item_score(
        SLOPE, THRESHOLDS, THETA, columns, nodes, weights, max_grid_points=n_nodes**n_axes
    )
    eta = np.full((), float(SLOPE @ np.where(np.isin(np.arange(4), columns), 0.0, THETA)))
    joint = np.ones(())
    for axis, column in enumerate(columns):
        shape = [1] * n_axes
        shape[axis] = -1
        eta = eta + SLOPE[column] * nodes[axis].reshape(shape)
        joint = joint * weights[axis].reshape(shape)
    expected = graded_item._grm_expected_scores(THRESHOLDS, eta.reshape(-1))
    reference = float(np.dot(np.broadcast_to(joint, eta.shape).reshape(-1), expected))
    assert result == pytest.approx(reference, rel=1e-12)


# --- scale helper: whole-scale preflight ------------------------------------

SCALE_SLOPES = np.array([[1.2, 0.85, 0.40, -0.25], [0.95, 0.70, 0.55, -0.15]])
SCALE_THRESHOLDS = np.array([[1.05, 0.10, -0.70], [0.90, -0.05, -0.85]])
SCALE_VARIANCES = np.array([1.0, 2.25, 4.0, 1.0])


def _scale(**overrides) -> float:
    kwargs = {
        "slopes": SCALE_SLOPES,
        "thresholds": SCALE_THRESHOLDS,
        "theta": THETA,
        "integrate_columns": [(1,), (1, 2)],
        "factor_variances": SCALE_VARIANCES,
        "n_nodes": 3,
        "max_grid_points": 3 + 9,
    }
    kwargs.update(overrides)
    return expected_graded_scale_score(**kwargs)


def test_scale_requires_max_grid_points_keyword() -> None:
    kwargs = {"n_nodes": 3}
    with pytest.raises(TypeError, match="max_grid_points"):
        expected_graded_scale_score(
            SCALE_SLOPES, SCALE_THRESHOLDS, THETA, [(1,), (1, 2)], SCALE_VARIANCES, **kwargs
        )


def test_scale_budget_is_sum_of_item_grids_inclusive() -> None:
    assert _scale(max_grid_points=12) == _scale(max_grid_points=10**6)
    with pytest.raises(ValueError, match=r"grid has 12 points .* exceeding max_grid_points=11"):
        _scale(max_grid_points=11)


def test_scale_rejects_before_building_nodes(monkeypatch) -> None:
    calls = _forbid_kernel(monkeypatch)

    def no_nodes(n):
        raise AssertionError("nodes must not be built before work admission")

    monkeypatch.setattr(graded_item, "equal_probability_normal_nodes", no_nodes)
    with pytest.raises(ValueError, match=rf"grid has {2 * 191**3 + 191**2} points"):
        _scale(
            slopes=np.vstack([SCALE_SLOPES, SCALE_SLOPES[:1]]),
            thresholds=np.vstack([SCALE_THRESHOLDS, SCALE_THRESHOLDS[:1]]),
            integrate_columns=[(1, 2, 3), (1, 2), (1, 2, 3)],
            n_nodes=191,
            max_grid_points=191**3,
        )
    assert calls == []


def test_scale_rejects_huge_n_nodes_without_wrap_or_allocation(monkeypatch) -> None:
    calls = _forbid_kernel(monkeypatch)
    monkeypatch.setattr(
        graded_item,
        "equal_probability_normal_nodes",
        lambda n: (_ for _ in ()).throw(AssertionError("no node construction")),
    )
    with pytest.raises(ValueError, match=r"grid has \d{20,} points"):
        _scale(n_nodes=np.int64(2**32), max_grid_points=2**63 - 1)
    assert calls == []


def test_scale_validates_every_item_before_any_work(monkeypatch) -> None:
    calls = _forbid_kernel(monkeypatch)
    with pytest.raises(ValueError, match="integrate_columns entries"):
        _scale(integrate_columns=[(1,), (1, 9)])
    assert calls == []


@pytest.mark.parametrize("n_nodes", [0, True, 2.0])
def test_scale_node_count_type_errors_unchanged(n_nodes) -> None:
    with pytest.raises(ValueError, match="n_nodes must be"):
        _scale(n_nodes=n_nodes)


def test_study_scale_241_nodes_is_admitted_by_caller_budget() -> None:
    """No package cap: 241 nodes over 2 axes runs when the caller admits 241 + 241**2."""
    result = _scale(n_nodes=241, max_grid_points=241 + 241**2)
    assert 0.0 <= result <= 2 * SCALE_THRESHOLDS.shape[1]


def test_scale_without_integration_builds_no_nodes(monkeypatch) -> None:
    """n_nodes work is O(n); with nothing integrated it must not run at all."""
    monkeypatch.setattr(
        graded_item,
        "equal_probability_normal_nodes",
        lambda n: (_ for _ in ()).throw(AssertionError("no node construction")),
    )
    result = _scale(integrate_columns=[(), ()], n_nodes=10**12, max_grid_points=2)
    plug_in = sum(
        compute_expected_graded_item_score(
            SCALE_SLOPES[i], SCALE_THRESHOLDS[i], THETA, (), (), (), max_grid_points=1
        )
        for i in range(2)
    )
    assert result == plug_in


def test_single_axis_beyond_kernel_cell_limit_is_admitted() -> None:
    """HEAD: 312_501 nodes * 64 categories = 20,000,064 cells > the 20M kernel limit.

    That limit is a de facto node cap on one axis; blocking removes it.
    Oracle: independent NumPy GRM cell, evaluated in slices.
    """
    from fast_mlsirm.estimators.marginal import compute_grm_category_logprobs

    n_nodes = 312_501
    thresholds = np.linspace(3.0, -3.0, 63)
    nodes = np.linspace(-4.0, 4.0, n_nodes)
    weights = np.full(n_nodes, 1.0 / n_nodes)
    slope = np.array([1.0, 0.7])
    theta = np.array([0.2, 0.0])
    result = compute_expected_graded_item_score(
        slope, thresholds, theta, (1,), (nodes,), (weights,), max_grid_points=n_nodes
    )
    categories = np.arange(thresholds.size + 1)
    parts = []
    for low in range(0, n_nodes, 50_000):
        eta = 0.2 + 0.7 * nodes[low : low + 50_000]
        probs = np.exp(compute_grm_category_logprobs(eta, thresholds))
        parts.append(float(weights[low : low + 50_000] @ (probs @ categories)))
    assert result == pytest.approx(math.fsum(parts), rel=1e-12)
