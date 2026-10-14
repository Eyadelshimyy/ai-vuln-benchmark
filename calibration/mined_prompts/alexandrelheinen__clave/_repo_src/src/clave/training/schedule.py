"""When an optimizer step happens, and which rollouts fill it.

The stored pick decides which frames exist. This module only orders them.
A window is one optimizer step. `accumulation_steps` of 1 is one microbatch
per window, in archive order, which is the loop a run used before windows.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping, Sequence

Microbatch = tuple[str, tuple[int, ...]]
Window = tuple[Microbatch, ...]


def accumulation_schedule(
    picks: Mapping[str, Sequence[int]],
    *,
    rollout_order: Sequence[str],
    batch_size: int,
    accumulation_steps: int,
    seed: int,
    epoch: int,
) -> tuple[Window, ...]:
    """Build the windows for one epoch.

    Args:
        picks: Kept frame indexes per rollout.
        rollout_order: Archive order. Windows of one microbatch follow it.
        batch_size: Frames in one microbatch, from one rollout.
        accumulation_steps: Microbatches in one optimizer step.
        seed: Training seed.
        epoch: Epoch index, starting at zero. It changes the shuffle.

    Returns:
        Windows. Each microbatch names one rollout and the indexes to read.

    Raises:
        ValueError: If `accumulation_steps` or `batch_size` is below one.
    """
    if accumulation_steps < 1:
        raise ValueError(
            f"accumulation_steps is {accumulation_steps}, and a step needs "
            "at least one microbatch"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size is {batch_size}, and a microbatch needs a frame")
    queues = {
        rollout_id: deque(_shuffled(picks.get(rollout_id, ()), seed, epoch, rollout_id))
        for rollout_id in rollout_order
    }
    if accumulation_steps == 1:
        return _one_at_a_time(rollout_order, queues, batch_size)
    return _across_rollouts(rollout_order, queues, batch_size, accumulation_steps)


def _shuffled(
    indexes: Sequence[int], seed: int, epoch: int, rollout_id: str
) -> list[int]:
    """A copy of the kept indexes in this epoch's order."""
    pending = list(indexes)
    random.Random(f"{seed}:{epoch}:{rollout_id}").shuffle(pending)
    return pending


def _take(queue: deque[int], batch_size: int) -> tuple[int, ...]:
    """Pull up to one microbatch from a rollout."""
    chunk: list[int] = []
    while queue and len(chunk) < batch_size:
        chunk.append(queue.popleft())
    return tuple(chunk)


def _one_at_a_time(
    rollout_order: Sequence[str],
    queues: dict[str, deque[int]],
    batch_size: int,
) -> tuple[Window, ...]:
    """One microbatch per window, archives in the order they were recorded."""
    windows: list[Window] = []
    for rollout_id in rollout_order:
        queue = queues[rollout_id]
        while queue:
            windows.append(((rollout_id, _take(queue, batch_size)),))
    return tuple(windows)


def _across_rollouts(
    rollout_order: Sequence[str],
    queues: dict[str, deque[int]],
    batch_size: int,
    accumulation_steps: int,
) -> tuple[Window, ...]:
    """Fill a step from distinct rollouts until only a short tail remains."""
    ring: deque[str] = deque(rollout_order)
    windows: list[Window] = []
    while any(queues[rollout_id] for rollout_id in ring):
        active = [rollout_id for rollout_id in ring if queues[rollout_id]]
        window: list[Microbatch] = []
        if len(active) >= accumulation_steps:
            for rollout_id in active[:accumulation_steps]:
                window.append((rollout_id, _take(queues[rollout_id], batch_size)))
            for rollout_id, _indexes in window:
                ring.remove(rollout_id)
                ring.append(rollout_id)
        else:
            cursor = 0
            while len(window) < accumulation_steps and any(
                queues[rollout_id] for rollout_id in active
            ):
                rollout_id = active[cursor % len(active)]
                if queues[rollout_id]:
                    window.append((rollout_id, _take(queues[rollout_id], batch_size)))
                cursor += 1
        if window:
            windows.append(tuple(window))
    return tuple(windows)
