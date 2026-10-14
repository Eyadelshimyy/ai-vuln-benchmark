"""Resource-admission regressions for Rust-backed G-theory entry points."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

import fast_mlsirm.gtheory as gtheory


def _unexpected_core_discovery(name: str) -> object:
    """Fail if invalid G-theory evidence reaches native capability lookup."""
    raise AssertionError(f"invalid G-theory evidence reached Rust discovery: {name}")


def test_gtheory_pi_rejects_overrank_sequence_before_numpy_materialization(
    monkeypatch,
) -> None:
    """One-facet score trees fail on rank before dense NumPy materialization."""

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("over-rank G-theory evidence reached np.asarray")

    monkeypatch.setattr(gtheory.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"data must be a 2-D persons x items array"):
        gtheory.gtheory_pi([[[0.0, 1.0]]], n_i_prime=[2])


def test_gtheory_pio_rejects_overrank_sequence_before_numpy_materialization(
    monkeypatch,
) -> None:
    """Two-facet score trees fail on rank before dense NumPy materialization."""

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("over-rank G-theory evidence reached np.asarray")

    monkeypatch.setattr(gtheory.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match=r"data must be a 3-D persons x items x occasions array",
    ):
        gtheory.gtheory_pio([[[[0.0, 1.0]]]], n_prime=[(2, 2)])


def test_phi_lambda_rejects_overrank_sequence_before_numpy_materialization(
    monkeypatch,
) -> None:
    """Mastery G-theory score trees share the two-dimensional rank boundary."""

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("over-rank Phi(lambda) evidence reached np.asarray")

    monkeypatch.setattr(gtheory.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=r"data must be a 2-D persons x items array"):
        gtheory.phi_lambda([[[0.0, 1.0]]], 0.5, n_i_prime=[2])


def test_gtheory_pi_rejects_oversized_exact_view_before_contiguous_copy(
    monkeypatch,
) -> None:
    """Logical cell bounds apply before an oversized view is densely copied."""
    oversized = np.broadcast_to(
        np.array(0.0, dtype=np.float64),
        (1, gtheory.MAX_GTHEORY_SCORE_CELLS + 1),
    )

    def unexpected_contiguous_copy(*args, **kwargs):
        raise AssertionError("oversized G-theory evidence reached contiguous copy")

    monkeypatch.setattr(gtheory.np, "ascontiguousarray", unexpected_contiguous_copy)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match=rf"data exceeds the {gtheory.MAX_GTHEORY_SCORE_CELLS}-cell G-theory limit",
    ):
        gtheory.gtheory_pi(oversized, n_i_prime=[2])


def test_gtheory_pi_rejects_oversized_exact_row_before_numpy_materialization(
    monkeypatch,
) -> None:
    """Trusted sequence rows share the logical-cell bound before np.asarray."""
    oversized_row = np.broadcast_to(
        np.array(0.0, dtype=np.float64),
        (gtheory.MAX_GTHEORY_SCORE_CELLS + 1,),
    )

    def unexpected_asarray(*args, **kwargs):
        raise AssertionError("oversized G-theory row reached np.asarray")

    monkeypatch.setattr(gtheory.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    with pytest.raises(
        ValueError,
        match=rf"data exceeds the {gtheory.MAX_GTHEORY_SCORE_CELLS}-cell G-theory limit",
    ):
        gtheory.gtheory_pi([oversized_row], n_i_prime=[2])


def test_gtheory_sequence_cell_bound_preempts_eager_sibling_scheduling(
    monkeypatch,
) -> None:
    """Flat score rows fail without allocating traversal state per sibling."""
    data = [[0.0] * 100_000]
    monkeypatch.setattr(gtheory, "MAX_GTHEORY_SCORE_CELLS", 1)
    monkeypatch.setattr(gtheory, "_core_or_raise", _unexpected_core_discovery)

    tracemalloc.start()
    try:
        with pytest.raises(
            ValueError,
            match=r"data exceeds the 1-cell G-theory limit",
        ):
            gtheory.gtheory_pi(data, n_i_prime=[2])
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # The wide input exists before tracing. Indexed traversal needs only a
    # depth-sized stack; eagerly scheduling 100k sibling frames requires many
    # megabytes before the second-cell resource failure can fire.
    assert peak_bytes < 2_000_000
