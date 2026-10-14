"""Tests for bounded async helpers."""

from __future__ import annotations

import asyncio

import pytest
from ellements.core.async_utils import parallel_map


@pytest.mark.asyncio
async def test_parallel_map_preserves_order_and_respects_concurrency():
    active = 0
    peak_active = 0
    lock = asyncio.Lock()

    async def worker(item: tuple[int, float]) -> int:
        nonlocal active, peak_active

        value, delay = item
        async with lock:
            active += 1
            peak_active = max(peak_active, active)

        try:
            await asyncio.sleep(delay)
            return value * 10
        finally:
            async with lock:
                active -= 1

    results = await parallel_map(
        [(1, 0.03), (2, 0.01), (3, 0.02)],
        worker,
        max_concurrency=2,
    )

    assert results == [10, 20, 30]
    assert peak_active == 2


@pytest.mark.asyncio
async def test_parallel_map_returns_empty_for_no_items():
    async def worker(item: int) -> int:
        return item * 10

    assert await parallel_map([], worker) == []


@pytest.mark.asyncio
async def test_parallel_map_validates_concurrency():
    async def worker(item: int) -> int:
        return item

    with pytest.raises(ValueError, match="max_concurrency"):
        await parallel_map([1], worker, max_concurrency=0)
