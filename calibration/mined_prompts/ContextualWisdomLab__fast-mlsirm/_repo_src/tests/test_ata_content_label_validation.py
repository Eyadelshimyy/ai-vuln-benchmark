"""Fail-first safety contracts for ATA content-label validation."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ata as ata
from fast_mlsirm.types import MLSIRMParams


class _HostileLabel:
    """Label whose representation callbacks must never run during validation."""

    def __init__(self) -> None:
        self.str_calls = 0
        self.repr_calls = 0

    def __str__(self) -> str:
        self.str_calls += 1
        raise RuntimeError("ATA_CONTENT_STR_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("ATA_CONTENT_REPR_SENTINEL")


class _HostileString(str):
    """String subclass whose conversion callbacks must never run in ATA."""

    def __new__(cls, value: str) -> "_HostileString":
        instance = super().__new__(cls, value)
        instance.str_calls = 0
        instance.repr_calls = 0
        return instance

    def __str__(self) -> str:
        self.str_calls += 1
        raise RuntimeError("ATA_STRING_SUBCLASS_STR_SENTINEL")

    def __repr__(self) -> str:
        self.repr_calls += 1
        raise RuntimeError("ATA_STRING_SUBCLASS_REPR_SENTINEL")


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a tiny valid one-dimensional bank for public-boundary validation."""
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)),
        b=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_content_labels_reject_hostile_objects_before_numeric_work(monkeypatch) -> None:
    """ATA must reject non-string labels without invoking caller callbacks or scoring."""
    bank, factor_id = _bank()
    hostile = _HostileLabel()
    information_calls = 0

    def unexpected_information(*args, **kwargs):
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)

    with pytest.raises(ValueError, match="content labels must be strings"):
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            length=2,
            model="MIRT",
            content=np.array(["A", hostile, "A", "B"], dtype=object),
            seed=0,
        )

    assert information_calls == 0
    assert hostile.str_calls == 0
    assert hostile.repr_calls == 0


def test_content_labels_reject_string_subclasses_before_numeric_work(monkeypatch) -> None:
    """ATA must reject string subclasses before conversion callbacks or scoring."""
    bank, factor_id = _bank()
    hostile = _HostileString("A")
    information_calls = 0

    def unexpected_information(*args, **kwargs):
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)

    with pytest.raises(ValueError, match="content labels must be strings"):
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            length=2,
            model="MIRT",
            content=np.array([hostile, "A", "B", "B"], dtype=object),
            seed=0,
        )

    assert information_calls == 0
    assert hostile.str_calls == 0
    assert hostile.repr_calls == 0


def test_content_constraint_keys_reject_string_subclasses_before_numeric_work(
    monkeypatch,
) -> None:
    """ATA must reject string-subclass map keys without conversion or scoring."""
    bank, factor_id = _bank()
    hostile = _HostileString("A")
    information_calls = 0

    def unexpected_information(*args, **kwargs):
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)

    with pytest.raises(ValueError, match="content constraint keys must be strings"):
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            length=2,
            model="MIRT",
            content=np.array(["A", "A", "B", "B"], dtype=object),
            min_per_content={hostile: 1},
            seed=0,
        )

    assert information_calls == 0
    assert hostile.str_calls == 0
    assert hostile.repr_calls == 0


def test_content_shape_rejected_before_numeric_work(monkeypatch) -> None:
    """ATA must reject malformed label shape before evaluating item information."""
    bank, factor_id = _bank()
    information_calls = 0

    def unexpected_information(*args, **kwargs):
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)

    with pytest.raises(ValueError, match="content length must match the number of items"):
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            length=2,
            model="MIRT",
            content=np.array([["A", "B", "A", "B"]], dtype=object),
            seed=0,
        )

    assert information_calls == 0


def test_numpy_string_scalars_remain_supported() -> None:
    """ATA preserves accepted NumPy string labels and content constraints."""
    bank, factor_id = _bank()
    content = np.array(
        [np.str_("A"), np.str_("A"), np.str_("B"), np.str_("B")],
        dtype=object,
    )

    form = ata.assemble_to_target(
        bank,
        factor_id,
        np.array([0.0], dtype=np.float64),
        np.array([100.0], dtype=np.float64),
        length=2,
        model="MIRT",
        content=content,
        min_per_content={np.str_("B"): 1},
        seed=0,
    )

    assert form.items.shape == (2,)
    assert form.content_counts.get("B", 0) >= 1
