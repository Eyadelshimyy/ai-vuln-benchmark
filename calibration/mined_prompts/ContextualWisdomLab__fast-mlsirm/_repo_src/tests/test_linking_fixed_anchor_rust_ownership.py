"""Ownership contract for fixed-anchor parameter linking."""

from __future__ import annotations

import numpy as np

import fast_mlsirm._core as core
from fast_mlsirm.linking import link_fixed_item_parameters
from fast_mlsirm.types import MLSIRMParams


def _params(*, theta: list[float], slopes: list[float], intercepts: list[float]) -> MLSIRMParams:
    """Build a tiny one-dimensional parameter set for delegation evidence."""
    return MLSIRMParams(
        theta=np.asarray(theta, dtype=np.float64)[:, None],
        alpha=np.log(np.asarray(slopes, dtype=np.float64)),
        b=np.asarray(intercepts, dtype=np.float64),
        xi=np.zeros((len(theta), 1), dtype=np.float64),
        zeta=np.zeros((len(slopes), 1), dtype=np.float64),
        tau=-30.0,
    )


def test_fixed_anchor_linking_delegates_transformation_to_rust(monkeypatch) -> None:
    """Linked parameters and affine evidence must come from the Rust owner."""
    source = _params(theta=[-0.5, 0.5], slopes=[1.0, 1.5], intercepts=[-0.2, 0.7])
    target = _params(theta=[-0.4, 0.6], slopes=[0.9, 1.4], intercepts=[-0.1, 0.6])
    anchors = np.array([0, 1], dtype=np.int64)
    factors = np.array([0, 0], dtype=np.int64)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_link(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "theta": [[9.0], [8.0]],
            "alpha": [0.125, 0.25],
            "b": [-0.75, 1.25],
            "scale": [1.75],
            "shift": [-0.375],
        }

    monkeypatch.setattr(core, "link_fixed_item_parameters", fake_link, raising=False)

    linked, evidence = link_fixed_item_parameters(source, target, anchors, factors)

    assert len(calls) == 1
    assert np.array_equal(linked.theta, np.array([[9.0], [8.0]], dtype=np.float64))
    assert np.array_equal(linked.alpha, np.array([0.125, 0.25], dtype=np.float64))
    assert np.array_equal(linked.b, np.array([-0.75, 1.25], dtype=np.float64))
    assert np.array_equal(evidence["scale"], np.array([1.75], dtype=np.float64))
    assert np.array_equal(evidence["shift"], np.array([-0.375], dtype=np.float64))
    assert np.array_equal(evidence["anchor_items"], anchors)
