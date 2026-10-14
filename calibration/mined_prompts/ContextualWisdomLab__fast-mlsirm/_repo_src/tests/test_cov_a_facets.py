"""Coverage for the many-facet Rasch model fitter validation paths (facets.py)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fast_mlsirm.facets import FacetsFit, fit_facets


def _poly3(n_cat=3, seed=0, n_persons=20, n_items=3, n_raters=2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_cat, size=(n_persons, n_items, n_raters)).astype(float)
    for k in range(n_cat):
        y[k] = k
    return y


def test_fit_facets_happy_path_explicit_n_cat():
    fit = fit_facets(_poly3(), n_cat=3, q_theta=7, max_iter=5)
    assert isinstance(fit, FacetsFit)
    assert fit.item_difficulty.shape == (3,)
    assert fit.rater_severity.shape == (2,)
    assert fit.theta.shape == (20,)


def test_fit_facets_happy_path_inferred_n_cat():
    assert isinstance(fit_facets(_poly3(), q_theta=7, max_iter=5), FacetsFit)


def test_fit_facets_requires_rust_core():
    with patch("fast_mlsirm.fitstats._core_module", return_value=None):
        with pytest.raises(RuntimeError):
            fit_facets(_poly3(), n_cat=3, q_theta=7)


def test_fit_facets_rejects_bad_n_cat_type():
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=2.5)
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=True)


def test_fit_facets_rejects_n_cat_range():
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=1)
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=65)


def test_fit_facets_rejects_bad_q_theta():
    # #1929: no node-count cap; q_theta=8 is now accepted, only q_theta < 1 is not.
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=3, q_theta=0)


def test_fit_facets_rejects_bad_max_iter():
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=3, q_theta=7, max_iter=0)
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=3, q_theta=7, max_iter=True)


def test_fit_facets_rejects_bad_tol():
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=3, q_theta=7, tol=0.0)
    with pytest.raises(ValueError):
        fit_facets(_poly3(), n_cat=3, q_theta=7, tol=np.inf)


def test_fit_facets_rejects_non_3d():
    with pytest.raises(ValueError):
        fit_facets(np.zeros((10, 4)), n_cat=3, q_theta=7)


def test_fit_facets_rejects_empty():
    with pytest.raises(ValueError):
        fit_facets(np.zeros((0, 3, 2)), n_cat=3, q_theta=7)


def test_fit_facets_rejects_non_finite_observed():
    y = _poly3()
    y[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        fit_facets(y, n_cat=3, q_theta=7)


def test_fit_facets_rejects_non_integer_or_negative():
    y = _poly3()
    y[5, 0, 0] = 0.5
    with pytest.raises(ValueError):
        fit_facets(y, n_cat=3, q_theta=7)
    y2 = _poly3()
    y2[5, 0, 0] = -1.0
    with pytest.raises(ValueError):
        fit_facets(y2, n_cat=3, q_theta=7)


def test_fit_facets_inference_no_observed_values():
    with pytest.raises(ValueError):
        fit_facets(np.full((10, 3, 2), np.nan), q_theta=7)


def test_fit_facets_inference_single_category():
    with pytest.raises(ValueError):
        fit_facets(np.zeros((10, 3, 2)), q_theta=7)


def test_fit_facets_inference_too_many_categories():
    y = np.zeros((10, 3, 2))
    y[0, 0, 0] = 64.0
    with pytest.raises(ValueError):
        fit_facets(y, q_theta=7)


def test_fit_facets_rejects_category_over_declared_range():
    y = _poly3()
    y[5, 0, 0] = 3.0
    with pytest.raises(ValueError):
        fit_facets(y, n_cat=3, q_theta=7)


def test_fit_facets_rejects_item_with_no_observed():
    y = _poly3()
    y[:, 2, :] = np.nan
    with pytest.raises(ValueError):
        fit_facets(y, n_cat=3, q_theta=7)


def test_fit_facets_rejects_rater_with_no_observed():
    y = _poly3()
    y[:, :, 1] = np.nan
    with pytest.raises(ValueError):
        fit_facets(y, n_cat=3, q_theta=7)
