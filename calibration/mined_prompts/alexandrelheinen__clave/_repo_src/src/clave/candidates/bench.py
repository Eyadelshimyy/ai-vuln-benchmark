"""Measure what a candidate costs on the machine it runs on.

v0.1.2 rejected two architectures on estimated training cost. Everything here
replaces an estimate with a measurement, which is the only way a shortlist
survives contact with hardware.

Latency is measured one frame at a time because that is CLAVE's access pattern:
a conveyor produces frames singly and batching across objects is not available.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from clave.candidates.base import CandidateSpec

WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 10


@dataclass(frozen=True)
class BenchResult:
    """What a candidate cost, and the conditions it was measured under.

    Attributes:
        spec: The candidate measured.
        parameter_count: Total learnable parameters.
        median_latency_seconds: Median single-frame forward-pass time.
        spread_seconds: Difference between the fastest and slowest measured run.
        repetitions: How many timed runs the median came from.
        warmup: How many untimed runs preceded them.
        threads: Compute threads available during measurement.
        machine: Processor and platform the measurement was taken on.
        note: Set when the candidate loaded but could not be benchmarked.
    """

    spec: CandidateSpec
    parameter_count: int
    median_latency_seconds: float | None
    spread_seconds: float | None
    repetitions: int
    warmup: int
    threads: int
    machine: str
    note: str | None = None


def count_parameters(model: object) -> int:
    """Count learnable parameters, tolerating a model with none.

    Args:
        model: An object exposing `parameters()`, as torch modules do.

    Returns:
        The total parameter count, or 0 when the object exposes none.
    """
    params = getattr(model, "parameters", None)
    if params is None:
        return 0
    return sum(int(p.numel()) for p in params())


def _machine() -> tuple[str, int]:
    """Describe the machine and its usable compute threads."""
    try:
        import torch

        threads = int(torch.get_num_threads())
    except ImportError:
        import os

        threads = os.cpu_count() or 1
    return (
        f"{platform.processor() or platform.machine()} on {platform.system()}",
        threads,
    )


def benchmark(
    spec: CandidateSpec,
    model: object,
    forward: Callable[[], object],
    warmup: int = WARMUP_ITERATIONS,
    repetitions: int = MEASURED_ITERATIONS,
) -> BenchResult:
    """Time a single-frame forward pass, discarding warm-up.

    The first call into a freshly constructed model pays for lazy kernel
    selection and allocator warm-up, which has nothing to do with steady-state
    cost, so those iterations run untimed.

    Args:
        spec: The candidate being measured.
        model: The loaded model, used for its parameter count.
        forward: A zero-argument callable performing one forward pass.
        warmup: Untimed iterations run first.
        repetitions: Timed iterations whose median is reported.

    Returns:
        The measurement, or a result carrying a note when the forward pass
        raised. A candidate that loads but cannot run is reported rather than
        dropped.
    """
    machine, threads = _machine()
    parameters = count_parameters(model)

    try:
        for _ in range(warmup):
            forward()
        samples: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter()
            forward()
            samples.append(time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - one broken candidate must not stop the sweep
        return BenchResult(
            spec=spec,
            parameter_count=parameters,
            median_latency_seconds=None,
            spread_seconds=None,
            repetitions=0,
            warmup=warmup,
            threads=threads,
            machine=machine,
            note=f"forward pass failed: {type(exc).__name__}: {exc}",
        )

    return BenchResult(
        spec=spec,
        parameter_count=parameters,
        median_latency_seconds=statistics.median(samples),
        spread_seconds=max(samples) - min(samples),
        repetitions=repetitions,
        warmup=warmup,
        threads=threads,
        machine=machine,
    )
