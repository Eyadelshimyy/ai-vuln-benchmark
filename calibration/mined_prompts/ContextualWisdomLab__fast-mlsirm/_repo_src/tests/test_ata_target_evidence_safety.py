"""Fail-first trust-boundary contracts for ATA target-curve evidence."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ata as ata
from fast_mlsirm.types import MLSIRMParams


class _HostileArrayProvider:
    """Array-like object whose NumPy callback must never run during admission."""

    def __init__(self) -> None:
        self.array_calls = 0

    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        self.array_calls += 1
        raise RuntimeError("ATA_TARGET_ARRAY_SENTINEL")


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a tiny valid calibrated bank for target-evidence tests."""
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)),
        b=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def test_target_thetas_reject_array_provider_before_information(monkeypatch) -> None:
    """Target-grid admission must not execute caller NumPy protocols or scoring."""
    bank, factor_id = _bank()
    hostile = _HostileArrayProvider()
    item_calls = 0

    def unexpected_item_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal item_calls
        item_calls += 1
        return np.ones(4, dtype=np.float64)

    monkeypatch.setattr(ata, "item_information", unexpected_item_information)

    with pytest.raises(ValueError, match="target_thetas must be real numeric evidence"):
        ata.assemble_to_target(
            bank,
            factor_id,
            hostile,
            np.array([1.0], dtype=np.float64),
            length=2,
            model="MIRT",
        )

    assert hostile.array_calls == 0
    assert item_calls == 0


def test_target_info_rejects_array_provider_before_information(monkeypatch) -> None:
    """Target objective must be admitted before any item-information evaluation."""
    bank, factor_id = _bank()
    hostile = _HostileArrayProvider()
    information_calls = 0

    def unexpected_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)

    with pytest.raises(ValueError, match="target_info must be real numeric evidence"):
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            hostile,
            length=2,
            model="MIRT",
        )

    assert hostile.array_calls == 0
    assert information_calls == 0


def test_trusted_numpy_and_builtin_target_evidence_remains_supported(monkeypatch) -> None:
    """Concrete NumPy scalars and inert built-in containers keep accepted behavior."""
    bank, factor_id = _bank()
    captured: list[tuple[np.ndarray, np.ndarray]] = []

    def fixed_information(
        _bank: MLSIRMParams,
        _factor_id: np.ndarray,
        target_thetas: np.ndarray,
        *,
        model: str,
    ) -> np.ndarray:
        assert model == "MIRT"
        thetas = np.asarray(target_thetas)
        captured.append((thetas.copy(), np.empty(0)))
        return np.ones((thetas.shape[0], 4), dtype=np.float64)

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

    monkeypatch.setattr(ata, "item_information_matrix", fixed_information)
    monkeypatch.setattr(ata, "ata_core", lambda: _Core())

    form = ata.assemble_to_target(
        bank,
        factor_id,
        [np.float32(-0.5), np.int16(0), np.float64(0.5)],
        (np.int16(2), np.float32(2.0), 2.0),
        length=2,
        model="MIRT",
        seed=np.int16(7),
    )

    assert form.items.size == 2
    np.testing.assert_allclose(captured[0][0], np.array([[-0.5], [0.0], [0.5]]))
    np.testing.assert_allclose(form.target_info, np.array([2.0, 2.0, 2.0]))


def test_single_point_scalar_target_info_remains_supported(monkeypatch) -> None:
    """A trusted scalar target keeps the historical one-point ravel behavior."""
    bank, factor_id = _bank()

    def fixed_information(
        _bank: MLSIRMParams,
        _factor_id: np.ndarray,
        target_thetas: np.ndarray,
        *,
        model: str,
    ) -> np.ndarray:
        del _bank, _factor_id, model
        assert np.asarray(target_thetas).shape == (1, 1)
        return np.ones((1, 4), dtype=np.float64)

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

    monkeypatch.setattr(ata, "item_information_matrix", fixed_information)
    monkeypatch.setattr(ata, "ata_core", lambda: _Core())

    form = ata.assemble_to_target(
        bank,
        factor_id,
        np.array([0.0], dtype=np.float64),
        np.float32(2.0),
        length=2,
        model="MIRT",
    )

    np.testing.assert_allclose(form.target_info, np.array([2.0]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_thetas", np.array([0.0 + 1.0j]), "target_thetas must be real numeric evidence"),
        ("target_info", np.array([1.0 + 1.0j]), "target_info must be real numeric evidence"),
        ("target_info", [object()], "target_info must be real numeric evidence"),
        ("target_info", 10**400, "target_info must be real numeric evidence"),
    ],
)
def test_nonreal_or_unrepresentable_target_evidence_fails_before_information(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    """Malformed target evidence must fail before item-information evaluation."""
    bank, factor_id = _bank()
    information_calls = 0

    def unexpected_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)
    target_thetas: object = np.array([0.0], dtype=np.float64)
    target_info: object = np.array([1.0], dtype=np.float64)
    if field == "target_thetas":
        target_thetas = value
    else:
        target_info = value

    with pytest.raises(ValueError, match=message):
        ata.assemble_to_target(
            bank,
            factor_id,
            target_thetas,
            target_info,
            length=2,
            model="MIRT",
        )

    assert information_calls == 0


def test_cyclic_builtin_target_tree_fails_before_information(monkeypatch) -> None:
    """Cyclic trusted-container syntax must fail without NumPy or scoring work."""
    bank, factor_id = _bank()
    cyclic: list[object] = []
    cyclic.append(cyclic)
    information_calls = 0

    def unexpected_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)

    with pytest.raises(ValueError, match="target_info must be real numeric evidence"):
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            cyclic,
            length=2,
            model="MIRT",
        )

    assert information_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_thetas", [2**53 + 1]),
        ("target_info", 2**53 + 1),
        ("target_info", np.uint64(2**53 + 1)),
    ],
)
def test_lossy_integer_target_evidence_fails_before_information(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    """Exact integer target evidence cannot silently change in binary64."""
    bank, factor_id = _bank()
    information_calls = 0

    def unexpected_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        raise AssertionError("lossy target evidence must fail before scoring")

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)
    target_thetas: object = np.array([0.0], dtype=np.float64)
    target_info: object = np.array([1.0], dtype=np.float64)
    if field == "target_thetas":
        target_thetas = value
    else:
        target_info = value

    with pytest.raises(ValueError, match=rf"{field} could not be converted losslessly"):
        ata.assemble_to_target(
            bank,
            factor_id,
            target_thetas,
            target_info,
            length=2,
            model="MIRT",
        )

    assert information_calls == 0


def test_wider_real_target_info_must_not_round_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Higher-precision target evidence must retain identity or fail pre-scoring."""
    wide = np.longdouble(1.0) + np.finfo(np.longdouble).eps
    if np.longdouble(np.float64(wide)) == wide:
        pytest.skip("np.longdouble has no precision beyond float64 on this platform")

    bank, factor_id = _bank()
    information_calls = 0

    def unexpected_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        raise AssertionError("rounded target evidence must fail before scoring")

    monkeypatch.setattr(ata, "item_information_matrix", unexpected_information)

    with pytest.raises(ValueError, match="target_info could not be converted losslessly"):
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            wide,
            length=2,
            model="MIRT",
        )

    assert information_calls == 0


def test_builtin_target_tree_preserves_exact_numpy_row_compatibility(monkeypatch) -> None:
    """Exact NumPy row leaves keep historical inert array-like compatibility."""
    bank, factor_id = _bank()
    captured: list[np.ndarray] = []

    def fixed_information(
        _bank: MLSIRMParams,
        _factor_id: np.ndarray,
        target_thetas: np.ndarray,
        *,
        model: str,
    ) -> np.ndarray:
        del _bank, _factor_id
        assert model == "MIRT"
        captured.append(np.asarray(target_thetas).copy())
        return np.ones((1, 4), dtype=np.float64)

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

    monkeypatch.setattr(ata, "item_information_matrix", fixed_information)
    monkeypatch.setattr(ata, "ata_core", lambda: _Core())

    form = ata.assemble_to_target(
        bank,
        factor_id,
        [np.array([np.float32(0.0)], dtype=np.float32)],
        np.array([2.0], dtype=np.float64),
        length=2,
        model="MIRT",
    )

    assert form.items.size == 2
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], np.array([[0.0]], dtype=np.float64))
