"""Bounded-materialization contracts for ATA target-curve evidence."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ata as ata
from fast_mlsirm.types import MLSIRMParams


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a tiny one-dimensional calibrated bank for resource tests."""
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)),
        b=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_information_grid_limit_precedes_scalar_scan_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provably oversized dense grid must fail before per-cell or scoring work."""
    bank, factor_id = _bank()
    monkeypatch.setattr(ata, "MAX_ATA_INFORMATION_CELLS", 4, raising=False)
    scalar_calls = 0
    information_calls = 0

    def unexpected_scalar_validation(*_args: object, **_kwargs: object) -> None:
        nonlocal scalar_calls
        scalar_calls += 1
        raise AssertionError("oversized target grid reached scalar validation")

    def unexpected_item_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        raise AssertionError("oversized target grid reached psychometric scoring")

    monkeypatch.setattr(ata, "_require_lossless_float64_scalar", unexpected_scalar_validation)
    monkeypatch.setattr(ata, "item_information", unexpected_item_information)
    target_thetas = np.broadcast_to(np.array([0.0], dtype=np.float64), (2,))

    with pytest.raises(
        ValueError,
        match=r"target information matrix exceeds the 4-cell ATA limit",
    ):
        ata.item_information_matrix(bank, factor_id, target_thetas, model="MIRT")

    assert scalar_calls == 0
    assert information_calls == 0


def test_malformed_builtin_fanout_has_bounded_traversal_before_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero-cell container fan-out must exhaust a package work budget first."""
    monkeypatch.setattr(ata, "MAX_ATA_TARGET_CELLS", 1, raising=False)
    numpy_calls = 0
    original_asarray = ata.np.asarray

    def unexpected_asarray(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal numpy_calls
        numpy_calls += 1
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(ata.np, "asarray", unexpected_asarray)

    with pytest.raises(
        ValueError,
        match=r"target_info exceeds the 1-cell ATA evidence limit",
    ):
        ata._trusted_real_array([[], [], []], "target_info")

    assert numpy_calls == 0


def test_nested_exact_numpy_leaf_is_charged_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested inert NumPy leaf must consume its logical size before conversion."""
    monkeypatch.setattr(ata, "MAX_ATA_TARGET_CELLS", 1, raising=False)
    numpy_calls = 0
    scalar_calls = 0
    original_asarray = ata.np.asarray

    def unexpected_asarray(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal numpy_calls
        numpy_calls += 1
        return original_asarray(*args, **kwargs)

    def unexpected_scalar_validation(*_args: object, **_kwargs: object) -> None:
        nonlocal scalar_calls
        scalar_calls += 1
        raise AssertionError("oversized nested NumPy leaf reached scalar validation")

    monkeypatch.setattr(ata.np, "asarray", unexpected_asarray)
    monkeypatch.setattr(ata, "_require_lossless_float64_scalar", unexpected_scalar_validation)
    nested = [np.broadcast_to(np.array([0.0], dtype=np.float64), (2,))]

    with pytest.raises(
        ValueError,
        match=r"target_thetas exceeds the 1-cell ATA evidence limit",
    ):
        ata._trusted_real_array(nested, "target_thetas")

    assert numpy_calls == 0
    assert scalar_calls == 0


def test_excessive_builtin_nesting_fails_before_numpy_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deep one-cell trees must not consume an unbounded active traversal path."""
    monkeypatch.setattr(ata, "MAX_ATA_TARGET_NESTING", 4, raising=False)
    numpy_calls = 0
    original_asarray = ata.np.asarray

    def unexpected_asarray(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal numpy_calls
        numpy_calls += 1
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(ata.np, "asarray", unexpected_asarray)
    target: object = 0.0
    for _ in range(5):
        target = [target]

    with pytest.raises(
        ValueError,
        match=r"target_thetas exceeds the 4-level ATA nesting limit",
    ):
        ata._trusted_real_array(target, "target_thetas")

    assert numpy_calls == 0


def test_assembly_replays_target_theta_scalars_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A materialized float64 grid must not trigger a second Python scalar replay."""
    bank, factor_id = _bank()
    original_scalar_validation = ata._require_lossless_float64_scalar
    theta_scalar_calls = 0

    def counted_scalar_validation(value: object, name: str) -> None:
        nonlocal theta_scalar_calls
        if name == "target_thetas":
            theta_scalar_calls += 1
        original_scalar_validation(value, name)

    def fixed_item_information(
        _bank: MLSIRMParams,
        _factor_id: np.ndarray,
        *,
        theta: np.ndarray,
        model: str,
    ) -> np.ndarray:
        del _bank, _factor_id, theta
        assert model == "MIRT"
        return np.ones(4, dtype=np.float64)

    class _Core:
        @staticmethod
        def target_information_gains(
            matrix: np.ndarray,
            candidates: np.ndarray,
            target_info: np.ndarray,
            accumulated: np.ndarray,
        ) -> np.ndarray:
            del matrix, target_info, accumulated
            return np.arange(candidates.size, 0, -1, dtype=np.float64)

    monkeypatch.setattr(ata, "_require_lossless_float64_scalar", counted_scalar_validation)
    monkeypatch.setattr(ata, "item_information", fixed_item_information)
    monkeypatch.setattr(ata, "ata_core", lambda: _Core())

    form = ata.assemble_to_target(
        bank,
        factor_id,
        [np.float32(-0.5), np.float32(0.5)],
        np.array([1.0, 1.0], dtype=np.float64),
        length=1,
        model="MIRT",
    )

    assert form.items.size == 1
    assert theta_scalar_calls == 2
