"""Resource-bound regressions for rating-scale-model response admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.rsm as rsm
from fast_mlsirm import fit_rsm


MAX_RSM_RESPONSE_CELLS = 20_000_000


def _unexpected_asarray(*args, **kwargs):
    """Fail if NumPy materialization runs before the RSM resource gate."""

    raise AssertionError("NumPy materialization executed before RSM resource admission")


def test_oversized_exact_rsm_response_array_fails_before_materialization(monkeypatch):
    """A huge broadcast response view is rejected without a dense copy."""

    responses = np.broadcast_to(
        np.array([[0, 1]], dtype=np.int8),
        (MAX_RSM_RESPONSE_CELLS // 2 + 1, 2),
    )
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 20000000-cell RSM evidence budget",
    ):
        fit_rsm(responses, n_cat=2, q_theta=41, max_iter=500, tol=1e-6)


def test_one_dimensional_broadcast_fails_before_dense_materialization(monkeypatch):
    """A large 1-D view fails the public matrix contract before a dense copy."""

    responses = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (MAX_RSM_RESPONSE_CELLS,),
    )
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses must be a 2-D persons x items array",
    ):
        fit_rsm(responses, n_cat=2, q_theta=41, max_iter=500, tol=1e-6)


def test_flat_builtin_response_fails_before_dense_materialization(monkeypatch):
    """A flat built-in carrier fails the matrix contract before NumPy work."""

    responses = [0, 1]
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses must be a 2-D persons x items array",
    ):
        fit_rsm(responses, n_cat=2, q_theta=41, max_iter=500, tol=1e-6)


def test_one_item_broadcast_fails_before_dense_materialization(monkeypatch):
    """A structurally impossible one-item view is rejected before a dense copy."""

    responses = np.broadcast_to(
        np.array([[0]], dtype=np.int8),
        (MAX_RSM_RESPONSE_CELLS, 1),
    )
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses must contain at least one person and at least two item columns",
    ):
        fit_rsm(responses, n_cat=2, q_theta=41, max_iter=500, tol=1e-6)


def test_one_item_builtin_rows_fail_before_dense_materialization(monkeypatch):
    """Built-in one-item rows fail on shape before NumPy materializes them."""

    responses = [[0], [1]]
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses must contain at least one person and at least two item columns",
    ):
        fit_rsm(responses, n_cat=2, q_theta=41, max_iter=500, tol=1e-6)


def test_oversized_exact_numpy_row_fails_before_sequence_materialization(monkeypatch):
    """An oversized trusted NumPy row is bounded before list stacking."""

    row = np.broadcast_to(
        np.array([0], dtype=np.int8),
        (MAX_RSM_RESPONSE_CELLS + 1,),
    )
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 20000000-cell RSM evidence budget",
    ):
        fit_rsm([row], n_cat=2, q_theta=41, max_iter=500, tol=1e-6)


def test_empty_row_fanout_hits_structural_budget_before_materialization(monkeypatch):
    """Zero-cell container fan-out cannot bypass the RSM resource envelope."""

    monkeypatch.setattr(rsm, "_MAX_RSM_RESPONSE_STRUCTURAL_NODES", 4)
    monkeypatch.setattr(rsm.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"responses exceed the 4-node RSM structural evidence budget",
    ):
        fit_rsm([[], [], [], [], []], n_cat=2, q_theta=41, max_iter=500, tol=1e-6)


def test_structural_budget_preserves_valid_small_matrix(monkeypatch):
    """A normal non-empty matrix remains admissible inside the structural bound."""

    responses = [[0, 1]]
    monkeypatch.setattr(rsm, "_MAX_RSM_RESPONSE_STRUCTURAL_NODES", 4)

    assert rsm._trusted_response_source(responses) is responses


def test_rsm_resource_preflight_preserves_small_shared_rows():
    """Repeated inert built-in rows remain valid inside the bounded envelope."""

    shared_row = [0, 1]
    assert rsm._trusted_response_source([shared_row, shared_row]) == [
        shared_row,
        shared_row,
    ]
