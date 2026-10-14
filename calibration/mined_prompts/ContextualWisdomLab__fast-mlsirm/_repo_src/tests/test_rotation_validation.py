"""Fail-closed validation and branch coverage for factor rotation."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.rotation import (
    rotate_factor_loadings,
    rotation_criterion_value_gradient,
)


@pytest.fixture
def loadings() -> np.ndarray:
    """Return a valid two-factor loading matrix."""
    return np.asarray(
        [[0.8, 0.2], [0.7, 0.1], [0.1, 0.8], [0.2, 0.7]], dtype=np.float64
    )


def test_public_validation_rejects_invalid_names_modes_and_arrays(loadings: np.ndarray) -> None:
    """Invalid method metadata and arrays never cross the Rust boundary silently."""
    with pytest.raises(ValueError, match="non-empty"):
        rotate_factor_loadings(loadings, "")
    with pytest.raises(ValueError, match="non-empty"):
        rotate_factor_loadings(loadings, 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        rotate_factor_loadings(loadings, "geomin", mode="diagonal")
    with pytest.raises(ValueError, match="orthogonal rotation only"):
        rotate_factor_loadings(loadings, "varimax", mode="oblique")
    with pytest.raises(ValueError, match="oblique rotation only"):
        rotate_factor_loadings(loadings, "quartimin", mode="orthogonal")
    with pytest.raises(ValueError, match="two-dimensional"):
        rotate_factor_loadings(np.asarray([0.8, 0.2]), "varimax")
    invalid = loadings.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rotate_factor_loadings(invalid, "varimax")
    with pytest.raises(ValueError, match="unknown rotation criterion"):
        rotate_factor_loadings(loadings, "not_a_criterion")


def test_target_and_weight_validation_covers_shape_and_finiteness(loadings: np.ndarray) -> None:
    """Target metadata must be shape-compatible, finite-or-NaN, and non-negative."""
    with pytest.raises(ValueError, match="shape"):
        rotate_factor_loadings(
            loadings,
            "target",
            target=np.zeros((2, 2), dtype=np.float64),
        )
    infinite_target = np.zeros_like(loadings)
    infinite_target[0, 0] = np.inf
    with pytest.raises(ValueError, match="not infinity"):
        rotate_factor_loadings(loadings, "target", target=infinite_target)
    nonfinite_weights = np.ones_like(loadings)
    nonfinite_weights[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rotate_factor_loadings(
            loadings,
            "pst",
            target=np.zeros_like(loadings),
            weights=nonfinite_weights,
        )
    negative_weights = np.ones_like(loadings)
    negative_weights[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        rotate_factor_loadings(
            loadings,
            "pst",
            target=np.zeros_like(loadings),
            weights=negative_weights,
        )
    with pytest.raises(ValueError, match="requires target"):
        rotate_factor_loadings(loadings, "target")
    with pytest.raises(ValueError, match="requires weights"):
        rotate_factor_loadings(loadings, "pst", target=np.zeros_like(loadings))


def test_criterion_endpoint_uses_same_validation_contract(loadings: np.ndarray) -> None:
    """The diagnostic gradient endpoint shares target and weight validation."""
    value, gradient = rotation_criterion_value_gradient(loadings, "cf", kappa=0.3)
    assert np.isfinite(value)
    assert gradient.shape == loadings.shape
    with pytest.raises(ValueError, match="shape"):
        rotation_criterion_value_gradient(
            loadings,
            "target",
            target=np.zeros((1, 2), dtype=np.float64),
        )
    negative_weights = -np.ones_like(loadings)
    with pytest.raises(ValueError, match="non-negative"):
        rotation_criterion_value_gradient(
            loadings,
            "pst",
            target=np.zeros_like(loadings),
            weights=negative_weights,
        )
    with pytest.raises(ValueError, match="finite"):
        rotation_criterion_value_gradient(
            loadings,
            "lp_wls",
            weights=np.full_like(loadings, np.nan),
        )


def test_core_numerical_settings_fail_closed(loadings: np.ndarray) -> None:
    """Invalid numerical settings are rejected by the compiled core."""
    invalid_cases = [
        {"n_starts": 0},
        {"max_iter": 0},
        {"function_window": 0},
        {"max_line_search": 0},
        {"tolerance": 0.0},
        {"basin_tolerance": np.nan},
        {"simplimax_zeros": 0, "criterion": "simplimax"},
        {"delta": 0.0, "criterion": "geomin"},
        {"kappa": -0.1, "criterion": "cf"},
    ]
    for kwargs in invalid_cases:
        method = kwargs.pop("criterion", "varimax")
        with pytest.raises(ValueError):
            rotate_factor_loadings(loadings, method, **kwargs)


def test_mode_aliases_and_default_modes_are_stable(loadings: np.ndarray) -> None:
    """Aliases and dual-manifold default resolution remain backward auditable."""
    orthogonal = rotate_factor_loadings(
        loadings,
        "varimax",
        mode="T",
        n_starts=1,
        max_iter=10,
    )
    assert orthogonal.mode == "orthogonal"
    oblique = rotate_factor_loadings(
        loadings,
        "geomin",
        mode="oblq",
        n_starts=1,
        max_iter=10,
    )
    assert oblique.mode == "oblique"
    default_dual = rotate_factor_loadings(
        loadings,
        "geomin",
        n_starts=1,
        max_iter=10,
    )
    assert default_dual.mode == "oblique"
