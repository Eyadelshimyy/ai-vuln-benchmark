"""Fail-first ownership contracts for CAT item information and selection."""

from __future__ import annotations

import numpy as np

import fast_mlsirm._core as core
from fast_mlsirm.test_design import item_information, select_cat_item
from fast_mlsirm.types import MLSIRMParams


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small one-dimensional bank for CAT ownership tests."""
    discrimination = np.array([0.8, 1.1, 1.4, 1.7], dtype=np.float64)
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(discrimination),
        b=np.array([-1.0, -0.2, 0.4, 1.2], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_public_item_information_uses_rust_numerical_result(monkeypatch) -> None:
    """Public item information must delegate its numerical result to Rust."""
    bank, factor_id = _bank()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_information(*args, **kwargs):
        calls.append((args, kwargs))
        return [0.125, 0.25, 0.5, 0.75]

    monkeypatch.setattr(core, "cat_item_information", fake_information, raising=False)

    result = item_information(
        bank,
        factor_id,
        theta=np.array([0.25], dtype=np.float64),
        model="MIRT",
    )

    assert len(calls) == 1
    assert np.array_equal(result, np.array([0.125, 0.25, 0.5, 0.75]))


def test_public_cat_selection_uses_rust_ranking_result(monkeypatch) -> None:
    """Next-item argmax/exclusion policy that determines the result must be Rust-owned."""
    bank, factor_id = _bank()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_selection(*args, **kwargs):
        calls.append((args, kwargs))
        return 2

    monkeypatch.setattr(core, "cat_select_item", fake_selection, raising=False)

    selected = select_cat_item(
        bank,
        factor_id,
        theta=np.array([-0.5], dtype=np.float64),
        administered=np.array([0, 1], dtype=np.int64),
        model="MIRT",
    )

    assert len(calls) == 1
    assert selected == 2
