#!/usr/bin/env python3
"""Measure NumPy fit-statistics fallback latency and Python allocation evidence.

The report is machine- and configuration-specific. It verifies numerical parity
against the former equations and does not claim a universal speed or memory
improvement. The compiled Rust core remains the production backend.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from typing import Any, Callable

import numpy as np

from fast_mlsirm import fitstats
from fast_mlsirm.types import MLSIRMParams


def _positive_integer(value: str) -> int:
    """Return a strictly positive integer for bounded benchmark dimensions."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _arguments() -> argparse.Namespace:
    """Parse bounded dimensions, repetitions, and deterministic seed."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persons", type=_positive_integer, default=2_000)
    parser.add_argument("--items", type=_positive_integer, default=80)
    parser.add_argument("--repetitions", type=_positive_integer, default=7)
    parser.add_argument("--seed", type=int, default=73)
    return parser.parse_args()


def _fixture(
    persons: int,
    items: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MLSIRMParams]:
    """Return deterministic partially observed MLSIRM fit-statistics inputs."""

    if persons * items > 20_000_000:
        raise ValueError("benchmark matrix exceeds the 20,000,000-cell safety bound")
    random_generator = np.random.default_rng(seed)
    responses = random_generator.integers(0, 2, size=(persons, items)).astype(
        np.float64
    )
    observed = random_generator.random((persons, items)) > 0.2
    factor_id = np.zeros(items, dtype=np.int64)
    params = MLSIRMParams(
        theta=random_generator.normal(size=(persons, 1)),
        alpha=random_generator.normal(scale=0.1, size=items),
        b=random_generator.normal(scale=0.4, size=items),
        xi=random_generator.normal(size=(persons, 2)),
        zeta=random_generator.normal(size=(items, 2)),
        tau=-0.5,
    )
    return responses, observed, factor_id, params


def _former_equations(
    responses: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
) -> dict[str, np.ndarray]:
    """Return the former fallback equations for one bounded parity comparison."""

    slopes = np.exp(params.alpha)
    eta = slopes[None, :] * params.theta[:, factor_id] + params.b[None, :]
    difference = params.xi[:, None, :] - params.zeta[None, :, :]
    distance = np.sqrt(1e-8 + np.sum(difference * difference, axis=2))
    eta = eta - np.exp(params.tau) * distance
    probability = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0))),
        1e-12,
        1.0 - 1e-12,
    )
    variance = probability * (1.0 - probability)
    residual = (responses - probability) ** 2 * observed
    count = np.maximum(observed.sum(axis=0), 1)
    return {
        "outfit": (residual / variance * observed).sum(axis=0) / count,
        "infit": residual.sum(axis=0)
        / np.maximum((variance * observed).sum(axis=0), 1e-12),
    }


def _measure(operation: Callable[[], dict[str, np.ndarray]], repetitions: int) -> dict[str, Any]:
    """Return elapsed and traced-memory observations for one operation."""

    operation()
    elapsed: list[float] = []
    traced_peak: list[int] = []
    checksum = 0.0
    for _ in range(repetitions):
        tracemalloc.start()
        started = time.perf_counter()
        result = operation()
        elapsed.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced_peak.append(peak)
        checksum = float(np.sum(result["infit"]) + np.sum(result["outfit"]))
    return {
        "elapsed_seconds": elapsed,
        "elapsed_median_seconds": statistics.median(elapsed),
        "traced_peak_bytes": traced_peak,
        "traced_peak_maximum_bytes": max(traced_peak),
        "result_checksum": checksum,
    }


def main() -> int:
    """Run the fallback, prove former-equation parity, and print JSON evidence."""

    arguments = _arguments()
    responses, observed, factor_id, params = _fixture(
        arguments.persons,
        arguments.items,
        arguments.seed,
    )
    original_core_module = fitstats._core_module
    fitstats._core_module = lambda: None
    try:
        operation = lambda: fitstats.infit_outfit(
            responses,
            factor_id,
            params,
            "mlsirm",
            mask=observed,
        )
        result = operation()
        expected = _former_equations(responses, observed, factor_id, params)
        np.testing.assert_allclose(result["infit"], expected["infit"], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result["outfit"], expected["outfit"], rtol=1e-13, atol=1e-13)
        report = {
            "interpretation": (
                "Environment-specific evidence only; no universal performance claim."
            ),
            "environment": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "configuration": {
                "persons": arguments.persons,
                "items": arguments.items,
                "repetitions": arguments.repetitions,
                "seed": arguments.seed,
                "observed_fraction": float(np.mean(observed)),
            },
            "parity": {"passed": True, "rtol": 1e-13, "atol": 1e-13},
            "fallback": _measure(operation, arguments.repetitions),
        }
    finally:
        fitstats._core_module = original_core_module
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
