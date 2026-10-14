"""Resource-bound regressions for subscore scientific-evidence admission."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
import fast_mlsirm.subscores as subscores


def _unexpected_core_discovery():
    """Fail if rejected evidence reaches the compiled-core discovery boundary."""

    raise AssertionError("compiled core must not be discovered for invalid evidence")


def _unexpected_asarray(*args, **kwargs):
    """Fail if rejected evidence reaches NumPy materialization."""

    raise AssertionError("rejected evidence must fail before np.asarray")


class _CaptureCore:
    """Capture canonical Rust-boundary payloads without doing numeric work."""

    def __init__(self) -> None:
        self.payload: tuple[np.ndarray, int, int, list[int]] | None = None

    def subscore_analysis(self, values, n_persons, n_items, groups):
        self.payload = (values, n_persons, n_items, groups)
        raise RuntimeError("capture subscore payload")


def test_subscore_rejects_oversized_exact_response_before_numpy(monkeypatch) -> None:
    """Logical response size is bounded before any dense/materializing NumPy call."""

    oversized = np.broadcast_to(
        np.array([[0.0]], dtype=np.float64),
        (3, 6_666_667),
    )
    assert oversized.size == 20_000_001

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    monkeypatch.setattr(subscores.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match="responses must contain at most 20,000,000 logical cells",
    ):
        subscores.subscore_analysis(oversized, [0, 0, 1, 1])


def test_subscore_charges_exact_numpy_leaf_before_parent_materialization(monkeypatch) -> None:
    """A tiny-backing broadcast leaf cannot bypass the logical-cell envelope."""

    oversized_row = np.broadcast_to(
        np.array([0.0], dtype=np.float64),
        (20_000_001,),
    )
    responses = [oversized_row]

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    monkeypatch.setattr(subscores.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match="responses must contain at most 20,000,000 logical cells",
    ):
        subscores.subscore_analysis(responses, [0, 0, 1, 1])


def test_subscore_rejects_cyclic_builtin_evidence_before_numpy(monkeypatch) -> None:
    """Active-path cycle detection prevents nonterminating preflight traversal."""

    responses: list[object] = []
    responses.append(responses)

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    monkeypatch.setattr(subscores.np, "asarray", _unexpected_asarray)

    with pytest.raises(ValueError, match="responses must be acyclic numeric evidence"):
        subscores.subscore_analysis(responses, [0, 0, 1, 1])


def test_subscore_bounds_zero_cell_structural_fanout(monkeypatch) -> None:
    """Malformed container fan-out is bounded independently of scalar-cell count."""

    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    monkeypatch.setattr(subscores, "_MAX_SUBSCORE_STRUCTURE_NODES", 2)
    monkeypatch.setattr(subscores.np, "asarray", _unexpected_asarray)

    with pytest.raises(
        ValueError,
        match="responses exceeded structural traversal budget of 2 nodes",
    ):
        subscores.subscore_analysis([[], [], []], [0, 0, 1, 1])


def test_subscore_preserves_shared_acyclic_builtin_rows(monkeypatch) -> None:
    """Repeated sibling references are valid shared evidence, not cycles."""

    core = _CaptureCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    shared_row = [0.0, 1.0, 2.0, 0.0]
    responses = [shared_row, shared_row, [2.0, 3.0, 4.0, 2.0]]

    with pytest.raises(RuntimeError, match="capture subscore payload"):
        subscores.subscore_analysis(responses, [0, 0, 1, 1])

    assert core.payload is not None
    values, n_persons, n_items, groups = core.payload
    assert type(values) is np.ndarray
    assert values.dtype == np.float64
    assert values.shape == (12,)
    assert (n_persons, n_items) == (3, 4)
    assert groups == [0, 0, 1, 1]
