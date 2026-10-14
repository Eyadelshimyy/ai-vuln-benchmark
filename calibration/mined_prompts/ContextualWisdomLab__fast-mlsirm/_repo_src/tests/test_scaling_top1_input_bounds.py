"""Fail-first resource and trust-boundary contracts for top-1 CSR inputs."""

from __future__ import annotations

import numpy as np
import pytest

import fast_mlsirm.scaling as scaling


class _ExplodingLosers:
    """Finite probe that fails if loser consumption is not explicitly bounded."""

    def __init__(self) -> None:
        self.calls = 0

    def __iter__(self) -> "_ExplodingLosers":
        return self

    def __next__(self) -> int:
        self.calls += 1
        if self.calls <= 3:
            return 1
        raise RuntimeError("TOP1_LOSER_SENTINEL")


class _ExplodingObservations:
    """Outer probe whose ordinary iterator failure must be normalized."""

    def __init__(self) -> None:
        self.calls = 0

    def __iter__(self) -> "_ExplodingObservations":
        return self

    def __next__(self) -> tuple[int, tuple[int, ...]]:
        self.calls += 1
        if self.calls == 1:
            return (0, (1,))
        raise RuntimeError("TOP1_OUTER_SENTINEL")


def test_loser_iterable_is_bounded_before_caller_failure() -> None:
    """An impossible loser stream must be rejected before an unbounded next call."""
    losers = _ExplodingLosers()

    with pytest.raises(ValueError, match=r"loser set has more than n - 1 items"):
        scaling._top1_to_csr("probe", [(0, losers)], 3)

    assert losers.calls <= 3


def test_outer_iteration_failure_is_stable_and_non_reflective() -> None:
    """Ordinary outer iterator exceptions must not escape or reflect payload text."""
    data = _ExplodingObservations()

    with pytest.raises(ValueError) as excinfo:
        scaling._top1_to_csr("probe", data, 3)

    assert str(excinfo.value) == "probe: top-1 observation iteration failed"
    assert "TOP1_OUTER_SENTINEL" not in str(excinfo.value)


def test_inner_iteration_failure_is_stable_and_non_reflective() -> None:
    """Ordinary loser iterator exceptions must become package-owned errors."""

    def losers():
        yield 1
        raise RuntimeError("TOP1_INNER_SENTINEL")

    with pytest.raises(ValueError) as excinfo:
        scaling._top1_to_csr("probe", [(0, losers())], 3)

    assert str(excinfo.value) == "probe: loser iteration failed"
    assert "TOP1_INNER_SENTINEL" not in str(excinfo.value)


def test_top1_fixed_width_payload_obeys_shared_csr_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Winner/loser/start uint64 payload must fit the package-owned CSR ceiling."""
    # One observation with one loser needs 8 + 8 + 16 = 32 fixed-width bytes.
    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", 31)

    with pytest.raises(ValueError, match=r"top-1 CSR byte limit exceeded"):
        scaling._top1_to_csr("probe", [(0, (1,))], 2)


def test_top1_fixed_width_payload_accepts_exact_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact shared byte boundary remains accepted without changing transport."""
    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", 32)

    winners, losers, starts, n = scaling._top1_to_csr("probe", [(0, (1,))], 2)

    assert n == 2
    assert winners.dtype == np.uint64 and winners.flags.c_contiguous
    assert losers.dtype == np.uint64 and losers.flags.c_contiguous
    assert starts.dtype == np.uint64 and starts.flags.c_contiguous
    assert np.array_equal(winners, np.array([0], dtype=np.uint64))
    assert np.array_equal(losers, np.array([1], dtype=np.uint64))
    assert np.array_equal(starts, np.array([0, 1], dtype=np.uint64))


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_from_loser_iterator_propagates(signal: type[BaseException]) -> None:
    """Bounded validation must never swallow process-control exceptions."""

    def losers():
        yield 1
        raise signal()

    with pytest.raises(signal):
        scaling._top1_to_csr("probe", [(0, losers())], 3)
