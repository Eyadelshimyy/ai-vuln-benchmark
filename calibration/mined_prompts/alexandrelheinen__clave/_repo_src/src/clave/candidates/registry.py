"""The shortlisted candidates, one entry per architecture v0.1.2 advanced."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from clave.candidates.base import Candidate, Stage
from clave.candidates.bench import BenchResult, benchmark
from clave.candidates.perception import PERCEPTION_CANDIDATES
from clave.candidates.policy import POLICY_CANDIDATES

REGISTRY: tuple[tuple[Candidate, Callable[[Any], Callable[[], Any]]], ...] = (
    *PERCEPTION_CANDIDATES,
    *POLICY_CANDIDATES,
)


def specs() -> tuple[Any, ...]:
    """List every candidate spec without loading anything.

    Returns:
        The specs, readable on a machine with no optional dependency installed.
    """
    return tuple(candidate.spec for candidate, _ in REGISTRY)


def by_stage(stage: Stage) -> tuple[Any, ...]:
    """List the specs belonging to one learned stage.

    Args:
        stage: Perception or policy.

    Returns:
        The matching specs.
    """
    return tuple(spec for spec in specs() if spec.stage is stage)


def sweep(
    warmup: int = 3, repetitions: int = 10
) -> list[tuple[Any, BenchResult | None, str | None]]:
    """Load and benchmark every candidate.

    A candidate that cannot load is reported with its reason rather than
    dropped, and one broken candidate does not stop the sweep.

    Args:
        warmup: Untimed iterations per candidate.
        repetitions: Timed iterations per candidate.

    Returns:
        One row per candidate: its spec, its result when it ran, and the reason
        it did not when it did not.
    """
    rows: list[tuple[Any, BenchResult | None, str | None]] = []
    for candidate, forward_for in REGISTRY:
        loaded = candidate.load()
        if not loaded.loaded:
            rows.append((candidate.spec, None, loaded.unavailable_reason))
            continue
        result = benchmark(
            candidate.spec,
            loaded.model,
            forward_for(loaded.model),
            warmup=warmup,
            repetitions=repetitions,
        )
        rows.append((candidate.spec, result, result.note))
    return rows
