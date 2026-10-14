"""Async helpers for bounded parallel work."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")


async def parallel_map(
    items: Iterable[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    max_concurrency: int = 3,
) -> list[R]:
    """Apply an async worker to items with bounded concurrency.

    Results preserve the input order even when items finish out of order.
    If any worker raises, the remaining in-flight tasks are cancelled.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    item_list = list(items)
    if not item_list:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[R | None] = [None] * len(item_list)

    async def _run(index: int, item: T) -> None:
        async with semaphore:
            results[index] = await worker(item)

    async with asyncio.TaskGroup() as task_group:
        for index, item in enumerate(item_list):
            task_group.create_task(_run(index, item))

    return [cast(R, result) for result in results]
