"""Rectangular-shape preflight regressions for Many-Facet Rasch ratings."""

from __future__ import annotations

import pytest

import fast_mlsirm.facets as facets_module
from fast_mlsirm import fitstats
from fast_mlsirm.facets import fit_facets


def _unexpected_core_discovery():
    """Fail if malformed rating evidence reaches the compiled core."""

    raise AssertionError("compiled core must not be discovered for ragged ratings")


def _unexpected_numpy_materialization(*args, **kwargs):
    """Fail if ragged built-in evidence reaches NumPy materialization."""

    del args, kwargs
    raise AssertionError("ragged rating evidence must fail before np.asarray")


def test_fit_facets_rejects_ragged_builtin_tree_before_numpy(monkeypatch):
    """Ragged persons × items × raters evidence fails during inert preflight."""

    responses = [
        [[0, 1], [1]],
        [[1, 0], [0, 1]],
    ]

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    monkeypatch.setattr(facets_module.np, "asarray", _unexpected_numpy_materialization)

    with pytest.raises(
        ValueError,
        match="responses must be a 3-D persons x items x raters array",
    ):
        fit_facets(responses, n_cat=2, q_theta=41, max_iter=10, tol=1e-6)
