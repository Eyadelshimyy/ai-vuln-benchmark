"""Public contracts and numerical recovery tests for factor rotation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rotation import (
    RotationCriterionInfo,
    RotationSolution,
    available_rotation_criteria,
    rotate_factor_loadings,
    rotation_criterion_value_gradient,
)


def _simple_pattern() -> np.ndarray:
    """Return a population pattern with two well-separated factors."""
    return np.asarray(
        [
            [0.82, 0.05],
            [0.76, 0.08],
            [0.69, 0.12],
            [0.07, 0.80],
            [0.10, 0.72],
            [0.14, 0.66],
        ],
        dtype=np.float64,
    )


def _mixed_orthogonal() -> np.ndarray:
    """Mix the population pattern by an arbitrary orthogonal transform."""
    angle = 0.58
    transform = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return _simple_pattern() @ transform


def _cross_loading_energy(loadings: np.ndarray) -> float:
    """Return squared energy outside each row's largest absolute loading."""
    squared = np.square(loadings)
    return float(np.sum(squared) - np.sum(np.max(squared, axis=1)))


def test_catalogue_is_unique_and_covers_major_criterion_families() -> None:
    """The registry exposes a broad, immutable, inspectable criterion catalogue."""
    catalogue = available_rotation_criteria()
    assert isinstance(catalogue, tuple)
    assert all(isinstance(item, RotationCriterionInfo) for item in catalogue)
    names = [item.name for item in catalogue]
    assert len(names) >= 25
    assert len(names) == len(set(names))
    assert {
        "varimax",
        "quartimin",
        "geomin",
        "target",
        "bifactor",
        "bigeomin",
        "simplimax",
        "bentler",
        "infomax",
        "crawford_ferguson",
    }.issubset(names)
    target = next(item for item in catalogue if item.name == "target")
    assert target.requires_target
    assert target.orthogonal and target.oblique


@pytest.mark.parametrize(
    ("criterion", "kwargs"),
    [
        ("varimax", {}),
        ("geomin", {"delta": 0.03}),
        ("oblimin", {"gamma": 0.2}),
        (
            "target",
            {
                "target": np.asarray(
                    [[0.7, 0.0], [0.7, 0.0], [np.nan, np.nan], [0.0, 0.7]],
                    dtype=np.float64,
                )
            },
        ),
    ],
)
def test_criterion_gradients_match_independent_central_differences(
    criterion: str,
    kwargs: dict[str, object],
) -> None:
    """Rust analytic gradients agree with an independent Python finite difference."""
    loadings = np.asarray(
        [[0.8, 0.2], [0.1, 0.7], [0.5, -0.4], [0.3, 0.6]],
        dtype=np.float64,
    )
    value, gradient = rotation_criterion_value_gradient(loadings, criterion, **kwargs)
    assert np.isfinite(value)
    assert gradient.shape == loadings.shape
    step = 1e-6
    plus = loadings.copy()
    minus = loadings.copy()
    plus[1, 0] += step
    minus[1, 0] -= step
    plus_value, _ = rotation_criterion_value_gradient(plus, criterion, **kwargs)
    minus_value, _ = rotation_criterion_value_gradient(minus, criterion, **kwargs)
    numeric = (plus_value - minus_value) / (2.0 * step)
    assert gradient[1, 0] == pytest.approx(numeric, abs=4e-4)


def test_orthogonal_multistart_varimax_recovers_simple_structure_deterministically() -> None:
    """Varimax reduces cross-loading energy and preserves reproduced covariance."""
    mixed = _mixed_orthogonal()
    first = rotate_factor_loadings(
        mixed,
        "varimax",
        n_starts=12,
        seed=71,
        tolerance=1e-7,
        max_threads=2,
    )
    second = rotate_factor_loadings(
        mixed,
        "varimax",
        n_starts=12,
        seed=71,
        tolerance=1e-7,
        max_threads=2,
    )
    assert isinstance(first, RotationSolution)
    np.testing.assert_array_equal(first.pattern_matrix, second.pattern_matrix)
    np.testing.assert_array_equal(first.start_values, second.start_values)
    assert first.mode == "orthogonal"
    assert first.criterion == "varimax"
    assert first.backend == "rust_cpu_coarse_multithreaded"
    assert first.worker_count == 2
    assert first.n_starts == 12
    assert first.basin_support >= 1
    assert first.distinct_minima >= 1
    assert first.best_start_index < first.n_starts
    assert first.start_converged.dtype == np.bool_
    assert first.start_values.shape == (12,)
    np.testing.assert_allclose(first.factor_correlation, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(first.structure_matrix, first.pattern_matrix, atol=1e-12)
    np.testing.assert_allclose(
        first.pattern_matrix @ first.pattern_matrix.T,
        mixed @ mixed.T,
        atol=1e-8,
    )
    assert _cross_loading_energy(first.pattern_matrix) < _cross_loading_energy(mixed)


def test_oblique_quartimin_returns_pattern_structure_and_factor_correlations() -> None:
    """Oblique output obeys the pattern/structure correlation identity."""
    mixed = _mixed_orthogonal()
    solution = rotate_factor_loadings(
        mixed,
        "quartimin",
        n_starts=8,
        seed=19,
        tolerance=1e-7,
        max_threads=1,
    )
    assert solution.mode == "oblique"
    np.testing.assert_allclose(np.diag(solution.factor_correlation), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        solution.structure_matrix,
        solution.pattern_matrix @ solution.factor_correlation,
        atol=1e-10,
    )
    reproduced = (
        solution.pattern_matrix
        @ solution.factor_correlation
        @ solution.pattern_matrix.T
    )
    np.testing.assert_allclose(reproduced, mixed @ mixed.T, atol=2e-7)
    assert 0.0 <= solution.max_factor_correlation < 0.999


def test_partial_target_preserves_labelled_columns_and_accepts_nan_cells() -> None:
    """Target rotation supports evidence-free cells without relabelling columns."""
    mixed = _mixed_orthogonal()
    target = np.asarray(
        [
            [0.8, 0.0],
            [0.7, 0.0],
            [np.nan, np.nan],
            [0.0, 0.8],
            [0.0, 0.7],
            [np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    weights = np.where(np.isnan(target), 0.0, 1.0)
    solution = rotate_factor_loadings(
        mixed,
        "pst",
        target=target,
        weights=weights,
        mode="q",
        n_starts=6,
        seed=5,
    )
    assert solution.criterion == "target"
    assert np.mean(np.abs(solution.pattern_matrix[:2, 0])) > np.mean(
        np.abs(solution.pattern_matrix[:2, 1])
    )
    assert np.mean(np.abs(solution.pattern_matrix[3:5, 1])) > np.mean(
        np.abs(solution.pattern_matrix[3:5, 0])
    )


def test_public_pst_rejects_continuous_weights() -> None:
    """The GPArotation-compatible PST contract accepts binary masks only."""
    mixed = _mixed_orthogonal()
    target = np.zeros_like(mixed)
    weights = np.ones_like(mixed)
    weights[0, 0] = 0.25

    with pytest.raises(ValueError, match="binary zero-or-one"):
        rotation_criterion_value_gradient(
            mixed,
            "pst",
            target=target,
            weights=weights,
        )
    with pytest.raises(ValueError, match="binary zero-or-one"):
        rotate_factor_loadings(
            mixed,
            "pst",
            target=target,
            weights=weights,
            n_starts=2,
        )


def test_bifactor_rotation_keeps_general_factor_column_first() -> None:
    """Bifactor canonicalization never permutes the designated general column."""
    population = np.asarray(
        [
            [0.60, 0.55, 0.05],
            [0.65, 0.50, 0.08],
            [0.58, 0.47, 0.12],
            [0.62, 0.06, 0.56],
            [0.67, 0.08, 0.51],
            [0.59, 0.10, 0.48],
        ],
        dtype=np.float64,
    )
    solution = rotate_factor_loadings(
        population,
        "bigeomin",
        mode="orthogonal",
        n_starts=5,
        seed=101,
        delta=0.02,
    )
    assert solution.pattern_matrix.shape == population.shape
    assert np.mean(np.abs(solution.pattern_matrix[:, 0])) > 0.30
