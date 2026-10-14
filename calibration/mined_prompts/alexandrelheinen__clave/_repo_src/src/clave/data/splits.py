"""Deterministic, leakage-free splits.

The cheapest way to inflate an accuracy number is to test on something seen in
training, and it leaves no trace a reader can spot. So splitting partitions by
rollout rather than by frame, and the result is checked for overlap and fails
rather than being reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clave.errors import ClaveError

PART_NAMES = ("train", "validation", "test")


class SplitError(ClaveError):
    """The requested split is invalid, or the produced split leaks."""


@dataclass(frozen=True)
class SplitPlan:
    """Which rollouts belong to which part.

    Attributes:
        parts: Part name to the rollout identities it holds.
    """

    parts: dict[str, tuple[str, ...]]

    def part_of(self, rollout_id: str) -> str | None:
        """Return the part a rollout belongs to, or None if it is in none."""
        for name, members in self.parts.items():
            if rollout_id in members:
                return name
        return None


def _check_proportions(proportions: dict[str, float]) -> None:
    """Refuse proportions that do not describe a partition.

    Raises:
        SplitError: If a name is unknown or the values do not sum to one.
    """
    unknown = sorted(set(proportions) - set(PART_NAMES))
    if unknown:
        raise SplitError(f"unknown split parts: {', '.join(unknown)}")
    total = sum(proportions.values())
    if abs(total - 1.0) > 1e-9:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(proportions.items()))
        raise SplitError(
            f"split proportions must sum to 1, got {total} from {rendered}"
        )


def verify(plan: SplitPlan) -> None:
    """Check that no rollout appears in more than one part.

    Args:
        plan: The partition to check.

    Raises:
        SplitError: If any rollout appears twice, naming it. A leaking split is
            a failure rather than a finding, because every number computed after
            it is an overstatement nobody can detect downstream.
    """
    seen: dict[str, str] = {}
    for name, members in plan.parts.items():
        for rollout_id in members:
            if rollout_id in seen:
                raise SplitError(
                    f"rollout {rollout_id!r} appears in both {seen[rollout_id]!r} "
                    f"and {name!r}; the split leaks"
                )
            seen[rollout_id] = name


def split(
    rollout_ids: tuple[str, ...], proportions: dict[str, float], seed: int
) -> SplitPlan:
    """Partition rollouts into parts.

    Args:
        rollout_ids: Every rollout in the dataset.
        proportions: Part name to share, summing to one.
        seed: Seed controlling the shuffle.

    Returns:
        The partition, verified free of overlap.

    Raises:
        SplitError: If the proportions are invalid or the result leaks.
    """
    _check_proportions(proportions)
    order = list(rollout_ids)
    np.random.default_rng(seed).shuffle(order)

    parts: dict[str, tuple[str, ...]] = {}
    start = 0
    names = [name for name in PART_NAMES if name in proportions]
    for index, name in enumerate(names):
        if index == len(names) - 1:
            parts[name] = tuple(order[start:])
            continue
        take = int(round(len(order) * proportions[name]))
        parts[name] = tuple(order[start : start + take])
        start += take

    plan = SplitPlan(parts=parts)
    verify(plan)
    return plan
