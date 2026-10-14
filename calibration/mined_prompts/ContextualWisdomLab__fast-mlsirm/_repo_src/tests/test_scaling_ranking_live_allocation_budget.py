"""Fail-first allocation contracts for LSR ranking CSR materialization."""

from __future__ import annotations

import math
import weakref

import numpy as np
import pytest

import fast_mlsirm.scaling as scaling


def _element_count(shape: object) -> int:
    """Return the number of elements requested by a NumPy allocation shape."""
    if isinstance(shape, (int, np.integer)):
        return int(shape)
    return math.prod(int(value) for value in shape)  # type: ignore[arg-type]


def test_tiny_budget_does_not_allocate_oversized_uint64_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal uint64 capacities must not exceed the declared CSR payload ceiling."""
    budget = 32  # two item ids + two start offsets at eight bytes each
    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", budget)
    real_empty = np.empty
    requested_uint64_bytes: list[int] = []

    def recording_empty(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        dtype = kwargs.get("dtype", args[0] if args else np.float64)
        if np.dtype(dtype) == np.dtype(np.uint64):
            requested_uint64_bytes.append(_element_count(shape) * np.dtype(np.uint64).itemsize)
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(scaling.np, "empty", recording_empty)

    flat, starts, n = scaling._rankings_to_csr("probe", [[0, 1]], 2)

    assert n == 2
    assert flat.nbytes + starts.nbytes == budget
    assert requested_uint64_bytes
    assert max(requested_uint64_bytes) <= budget


def test_growth_never_exceeds_budget_with_old_and_new_buffers_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reallocation overlap must count both old and replacement uint64 buffers."""
    budget = 48
    monkeypatch.setattr(scaling, "MAX_RANKING_CSR_BYTES", budget)
    real_empty = np.empty
    real_array = np.array
    live_allocations: list[weakref.ReferenceType[np.ndarray]] = []
    peak_live_bytes = 0

    def record(array: np.ndarray) -> np.ndarray:
        nonlocal peak_live_bytes
        if array.dtype == np.dtype(np.uint64):
            alive_bytes = sum(
                live.nbytes
                for ref in live_allocations
                if (live := ref()) is not None
            )
            peak_live_bytes = max(peak_live_bytes, alive_bytes + array.nbytes)
            live_allocations.append(weakref.ref(array))
        return array

    def recording_empty(shape: object, *args: object, **kwargs: object) -> np.ndarray:
        return record(real_empty(shape, *args, **kwargs))

    def recording_array(value: object, *args: object, **kwargs: object) -> np.ndarray:
        return record(real_array(value, *args, **kwargs))

    monkeypatch.setattr(scaling.np, "empty", recording_empty)
    monkeypatch.setattr(scaling.np, "array", recording_array)

    flat, starts, n = scaling._rankings_to_csr("probe", [[0, 1]], 2)

    assert n == 2
    assert np.array_equal(flat, real_array([0, 1], dtype=np.uint64))
    assert np.array_equal(starts, real_array([0, 2], dtype=np.uint64))
    assert peak_live_bytes <= budget


def test_ranking_copy_does_not_create_unbudgeted_uint64_list_temporary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming validation must not add a list→uint64 temporary beside live CSR arrays."""
    real_asarray = np.asarray
    list_uint64_conversions: list[int] = []

    def recording_asarray(value: object, *args: object, **kwargs: object) -> np.ndarray:
        dtype = kwargs.get("dtype", args[0] if args else None)
        if isinstance(value, list) and dtype is not None and np.dtype(dtype) == np.dtype(np.uint64):
            list_uint64_conversions.append(len(value))
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(scaling.np, "asarray", recording_asarray)

    flat, starts, n = scaling._rankings_to_csr("probe", ([0, 1], [1, 2]), 3)

    assert n == 3
    assert np.array_equal(flat, np.array([0, 1, 1, 2], dtype=np.uint64))
    assert np.array_equal(starts, np.array([0, 2, 4], dtype=np.uint64))
    assert list_uint64_conversions == []
