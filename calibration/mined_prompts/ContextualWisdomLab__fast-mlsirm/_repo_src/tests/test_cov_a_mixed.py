"""Coverage for the mixed-format item-bank calibrator (mixed.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.mixed import (
    MixedFormatFit,
    _categories,
    _normalize_models,
    fit_mixed_items,
)


def _binary(seed=0, n_persons=80, n_items=4):
    rng = np.random.default_rng(seed)
    y = (rng.random((n_persons, n_items)) < 0.5).astype(float)
    y[0, :] = 0.0
    y[1, :] = 1.0
    return y


def _mixed_data(seed=5):
    """5 columns: three binary then two 3-category, all categories present."""
    rng = np.random.default_rng(seed)
    y = np.empty((80, 5))
    y[:, :3] = (rng.random((80, 3)) < 0.5).astype(float)
    y[:, 3:] = rng.integers(0, 3, size=(80, 2)).astype(float)
    y[0, :] = 0.0
    y[1, :3] = 1.0
    y[1, 3:] = 1.0
    y[2, 3:] = 2.0
    return y


def test_normalize_models_broadcast_and_list():
    assert _normalize_models("2pl", 3) == ("2pl", "2pl", "2pl")
    assert _normalize_models(["rasch", "1pl"], 2) == ("rasch", "rasch")


def test_normalize_models_rejects_length_and_unknown():
    with pytest.raises(ValueError):
        _normalize_models(["2pl", "2pl"], 4)
    with pytest.raises(ValueError):
        _normalize_models("totally-bogus", 2)


def test_categories_inference_and_validation():
    y = np.array([[0, 1], [1, 2], [1, 0]], dtype=np.int64)
    observed = np.ones_like(y, dtype=bool)
    inferred = _categories(y, observed, None)
    assert inferred.tolist() == [2, 3]
    # explicit shape mismatch / non-integer / below-two / exceed-declared
    with pytest.raises(ValueError):
        _categories(y, observed, np.array([2]))
    with pytest.raises(ValueError):
        _categories(y, observed, np.array([2.5, 3.0]))
    with pytest.raises(ValueError):
        _categories(y, observed, np.array([1, 3]))
    with pytest.raises(ValueError):
        _categories(y, observed, np.array([2, 2]))


def test_categories_inference_item_without_observations():
    y = np.zeros((3, 2), dtype=np.int64)
    observed = np.ones((3, 2), dtype=bool)
    observed[:, 1] = False
    with pytest.raises(ValueError):
        _categories(y, observed, None)


def test_fit_mixed_converged_infers_categories():
    fit = fit_mixed_items(
        _binary(), "rasch", n_categories=None, max_iter=200,
        q_theta=7, q_xi=7, latent_dim=1, tol=1e-3,
    )
    assert isinstance(fit, MixedFormatFit)
    assert fit.converged
    assert len(fit.items) == 4


def test_fit_mixed_family_marshalling_paths_warn():
    with pytest.warns(RuntimeWarning):
        fit = fit_mixed_items(
            _mixed_data(),
            ["2pl", "rasch", "3pl", "nominal", "grm"],
            n_categories=[2, 2, 2, 3, 3],
            max_iter=5, q_theta=7, q_xi=7, latent_dim=2, tol=1e-5
        )
    models = {it.model for it in fit.items}
    assert "nominal" in models  # slope=None branch
    assert "rasch" in models  # location not-None branch
    assert "3pl" in models  # asymptote not-None branch


def test_fit_mixed_valid_mask_with_missing_cell_warns():
    y = _binary()
    mask = np.isfinite(y)
    mask[10, 0] = False  # forces the observed.all()==False marshalling branch
    with pytest.warns(RuntimeWarning):
        fit_mixed_items(
            y, "2pl", mask=mask, max_iter=5, q_theta=7, q_xi=7, latent_dim=1, tol=1e-5
        )


def test_fit_mixed_require_convergence_raises():
    with pytest.raises(RuntimeError):
        fit_mixed_items(
            _binary(), "rasch", max_iter=1, q_theta=7, q_xi=7,
            latent_dim=1, require_convergence=True, tol=1e-5
        )


def test_fit_mixed_requires_rust_core():
    with patch("fast_mlsirm._core", object()):
        with pytest.raises(RuntimeError):
            fit_mixed_items(_binary(), "2pl", q_theta=7, q_xi=7, max_iter=100, tol=1e-5)


def test_fit_mixed_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_mixed_items(np.zeros(4), "2pl", q_theta=21, q_xi=7, max_iter=100, tol=1e-5)


def test_fit_mixed_rejects_empty():
    with pytest.raises(ValueError):
        fit_mixed_items(np.zeros((0, 4)), "2pl", q_theta=21, q_xi=7, max_iter=100, tol=1e-5)


def test_fit_mixed_rejects_mask_shape_and_non_finite():
    with pytest.raises(ValueError):
        fit_mixed_items(_binary(), "2pl", mask=np.ones((2, 2), dtype=bool), q_theta=21, q_xi=7, max_iter=100, tol=1e-5)
    y = _binary()
    y[5, 0] = np.nan
    with pytest.raises(ValueError):
        fit_mixed_items(y, "2pl", mask=np.ones_like(y, dtype=bool), q_theta=21, q_xi=7, max_iter=100, tol=1e-5)


def test_fit_mixed_rejects_negative_or_non_integer_values():
    y = _binary()
    y[5, 0] = -2.0
    with pytest.raises(ValueError):
        fit_mixed_items(y, "2pl", q_theta=21, q_xi=7, max_iter=100, tol=1e-5)
    y2 = _binary()
    y2[5, 0] = 0.5
    with pytest.raises(ValueError):
        fit_mixed_items(y2, "2pl", q_theta=21, q_xi=7, max_iter=100, tol=1e-5)


def test_fit_mixed_rejects_bad_hyperparameters():
    y = _binary()
    for ld in (0, 4, 1.5):
        with pytest.raises(ValueError):
            fit_mixed_items(y, "2pl", latent_dim=ld, q_theta=21, q_xi=7, max_iter=100, tol=1e-5)
    # #1929: no node-count cap; q_theta/q_xi=8 are now accepted, only < 1 is not.
    with pytest.raises(ValueError):
        fit_mixed_items(y, "2pl", q_theta=0, q_xi=7, max_iter=100, tol=1e-5)
    with pytest.raises(ValueError):
        fit_mixed_items(y, "2pl", q_xi=0, q_theta=21, max_iter=100, tol=1e-5)
    for mi in (0, 1.5):
        with pytest.raises(ValueError):
            fit_mixed_items(y, "2pl", max_iter=mi, q_theta=21, q_xi=7, tol=1e-5)
    for tol in (0.0, np.inf):
        with pytest.raises(ValueError):
            fit_mixed_items(y, "2pl", tol=tol, q_theta=21, q_xi=7, max_iter=100)
    for nt in (-1, 1.5):
        with pytest.raises(ValueError):
            fit_mixed_items(y, "2pl", n_threads=nt, q_theta=21, q_xi=7, max_iter=100, tol=1e-5)
