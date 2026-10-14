"""Fail-first ATA semantic-range and exclusion trust-boundary contracts."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.ata as ata
from fast_mlsirm.types import MLSIRMParams


class _HostileInteger:
    """Integer-like object whose conversion hooks must never execute."""

    def __init__(self) -> None:
        self.int_calls = 0
        self.index_calls = 0

    def __int__(self) -> int:
        self.int_calls += 1
        raise RuntimeError("ATA_EXCLUDE_INT_SENTINEL")

    def __index__(self) -> int:
        self.index_calls += 1
        raise RuntimeError("ATA_EXCLUDE_INDEX_SENTINEL")


def _bank() -> tuple[MLSIRMParams, np.ndarray]:
    """Return a small valid calibrated bank for semantic-preflight tests."""
    bank = MLSIRMParams(
        theta=np.array([[0.0]], dtype=np.float64),
        alpha=np.log(np.array([0.8, 1.0, 1.2, 1.4], dtype=np.float64)),
        b=np.array([-1.0, -0.25, 0.5, 1.0], dtype=np.float64),
        xi=np.zeros((1, 1), dtype=np.float64),
        zeta=np.zeros((4, 1), dtype=np.float64),
        tau=-30.0,
    )
    return bank, np.zeros(4, dtype=np.int64)


def _call_with_counted_information(
    monkeypatch: pytest.MonkeyPatch,
    **overrides: object,
) -> tuple[int, BaseException | None]:
    """Call ATA and expose whether invalid controls reached psychometric work."""
    bank, factor_id = _bank()
    information_calls = 0

    def counted_information(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal information_calls
        information_calls += 1
        return np.ones((1, 4), dtype=np.float64)

    monkeypatch.setattr(ata, "item_information_matrix", counted_information)
    kwargs: dict[str, object] = {
        "content": np.array(["A", "A", "B", "B"], dtype=object),
        "seed": 0,
    }
    kwargs.update(overrides)

    failure: BaseException | None = None
    try:
        ata.assemble_to_target(
            bank,
            factor_id,
            np.array([0.0], dtype=np.float64),
            np.array([100.0], dtype=np.float64),
            length=2,
            model="MIRT",
            **kwargs,
        )
    except BaseException as exc:  # noqa: BLE001 - process-control propagation is asserted separately.
        failure = exc
    return information_calls, failure


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": -1}, "seed must be non-negative"),
        ({"min_per_content": {"A": -1}}, "content constraint counts must be non-negative"),
        ({"max_per_content": {"A": -1}}, "content constraint counts must be non-negative"),
        ({"min_per_content": {"A": 2}, "max_per_content": {"A": 1}}, "minimum content constraint cannot exceed maximum"),
        ({"exposure_counts": {0: -1}}, "exposure_counts values must be non-negative"),
        ({"exposure_counts": {-1: 0}}, "exposure_counts keys must identify existing items"),
        ({"exposure_counts": {4: 0}}, "exposure_counts keys must identify existing items"),
    ],
)
def test_invalid_semantic_ranges_fail_before_information(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    """Invalid finite-domain controls must fail before item-information work."""
    calls, failure = _call_with_counted_information(monkeypatch, **overrides)

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == message


@pytest.mark.parametrize("exclude", [[True], [1.5], np.array([False], dtype=np.bool_)])
def test_exclude_requires_exact_integer_item_indices_before_information(
    monkeypatch: pytest.MonkeyPatch,
    exclude: object,
) -> None:
    """Boolean/fractional exclusions must not be coerced into item identities."""
    calls, failure = _call_with_counted_information(monkeypatch, exclude=exclude)

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == "exclude must contain integer item indices"


@pytest.mark.parametrize("exclude", [[-1], [4], np.array([4], dtype=np.int64)])
def test_exclude_rejects_out_of_bank_indices_before_information(
    monkeypatch: pytest.MonkeyPatch,
    exclude: object,
) -> None:
    """Exclusions outside the calibrated bank must fail before scoring."""
    calls, failure = _call_with_counted_information(monkeypatch, exclude=exclude)

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == "exclude item indices must identify existing items"


def test_exclude_rejects_hostile_integer_without_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusion validation must not execute caller integer-conversion hooks."""
    hostile = _HostileInteger()

    calls, failure = _call_with_counted_information(monkeypatch, exclude=[hostile])

    assert calls == 0
    assert isinstance(failure, ValueError)
    assert str(failure) == "exclude must contain integer item indices"
    assert hostile.int_calls == 0
    assert hostile.index_calls == 0


@pytest.mark.parametrize(
    "exclude",
    [
        [np.int64(3)],
        (np.int32(3),),
        np.array([3], dtype=np.int64),
        np.array([], dtype=np.uint64),
    ],
)
def test_valid_integer_exclusion_containers_reach_information(
    monkeypatch: pytest.MonkeyPatch,
    exclude: object,
) -> None:
    """Supported exact-integer exclusion containers preserve accepted behavior."""
    calls, failure = _call_with_counted_information(monkeypatch, exclude=exclude)

    assert calls == 1
    assert failure is None


def test_equal_nonnegative_content_bounds_remain_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feasible equal minimum/maximum count remains an accepted boundary."""
    calls, failure = _call_with_counted_information(
        monkeypatch,
        min_per_content={"A": 1},
        max_per_content={"A": 1},
    )

    assert calls == 1
    assert failure is None


def test_valid_numpy_integer_controls_preserve_preflight_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported NumPy integer controls still reach psychometric work once admitted."""
    calls, failure = _call_with_counted_information(
        monkeypatch,
        seed=np.int64(7),
        min_per_content={np.str_("A"): np.int64(1)},
        max_per_content={np.str_("A"): np.int64(2)},
        exposure_counts={np.int64(0): np.int64(0), np.int64(3): np.int64(0)},
        exposure_max=np.int64(5),
        exclude=np.array([np.int64(3)], dtype=np.int64),
    )

    assert calls == 1
    assert failure is None
