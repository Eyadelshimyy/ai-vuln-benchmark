#!/usr/bin/env python3
"""Report environment-specific timing and memory for marginal distances.

The benchmark compares the bounded coordinate-subtraction implementation with
the former three-dimensional broadcast only when the requested broadcast tensor
is below an intentionally conservative safety ceiling. Results characterize the
current environment; they are not universal speed or capacity claims.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
import pprint
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from fast_mlsirm.estimators.marginal import (  # noqa: E402
    MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES,
    _pairwise_euclidean_distances,
)

MAX_SAFE_BROADCAST_ELEMENTS = 5_000_000


def _positive_integer(value: str) -> int:
    """Return one positive command-line integer or raise an argparse error."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def _rss_bytes() -> int:
    """Return process maximum resident-set size in platform-normalized bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _numpy_configuration() -> str:
    """Return the complete NumPy build and linked-library configuration."""
    try:
        build_configuration = np.show_config(mode="dicts")
    except TypeError:
        pass
    else:
        return pprint.pformat(build_configuration, sort_dicts=True)
    configuration_output = io.StringIO()
    with redirect_stdout(configuration_output):
        np.show_config()
    return configuration_output.getvalue().strip()


def _legacy_broadcast(
    left: np.ndarray,
    right: np.ndarray,
    *,
    eps_distance: float,
) -> np.ndarray:
    """Return the former explicit three-dimensional distance equation."""
    difference = left[:, None, :] - right[None, :, :]
    return np.sqrt(eps_distance + np.sum(difference * difference, axis=2))


def _measure(
    callback: Callable[[], np.ndarray],
    *,
    warmups: int,
    repetitions: int,
) -> dict[str, object]:
    """Measure repeated elapsed time and traced/RSS peak deltas for one callback."""
    for _ in range(warmups):
        callback()

    elapsed_seconds: list[float] = []
    traced_peaks: list[int] = []
    rss_before = _rss_bytes()
    checksum = 0.0
    for _ in range(repetitions):
        tracemalloc.start()
        started = time.perf_counter()
        result = callback()
        elapsed_seconds.append(time.perf_counter() - started)
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        traced_peaks.append(int(traced_peak))
        checksum += float(result[0, 0]) if result.size else 0.0
    rss_after = _rss_bytes()
    return {
        "elapsed_seconds": {
            "minimum": min(elapsed_seconds),
            "median": statistics.median(elapsed_seconds),
            "maximum": max(elapsed_seconds),
            "values": elapsed_seconds,
        },
        "tracemalloc_peak_bytes": {
            "minimum": min(traced_peaks),
            "median": statistics.median(traced_peaks),
            "maximum": max(traced_peaks),
        },
        "maximum_rss_before_bytes": rss_before,
        "maximum_rss_after_bytes": rss_after,
        "maximum_rss_growth_bytes": max(rss_after - rss_before, 0),
        "checksum": checksum,
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    """Build one deterministic benchmark report for validated dimensions."""
    rng = np.random.default_rng(args.seed)
    left = np.ascontiguousarray(
        rng.standard_normal((args.n_left, args.latent_dim)),
        dtype=np.float64,
    )
    right = np.ascontiguousarray(
        rng.standard_normal((args.n_right, args.latent_dim)),
        dtype=np.float64,
    )
    eps_distance = 1e-8
    pairwise_output_bytes = args.n_left * args.n_right * np.dtype(np.float64).itemsize
    pairwise_kernel_peak_bytes = 2 * pairwise_output_bytes
    broadcast_elements = args.n_left * args.n_right * args.latent_dim

    bounded = _measure(
        lambda: _pairwise_euclidean_distances(
            left,
            right,
            eps_distance=eps_distance,
        ),
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    report: dict[str, object] = {
        "scope": "environment_specific_marginal_distance_workspace_evidence",
        "claims": [
            "removes_the_named_three_dimensional_broadcast",
            "eliminates_the_identified_high_offset_cancellation_mechanism",
            "enforces_the_private_float64_distance_byte_ceiling",
            "does_not_claim_universal_speedup_or_capacity",
        ],
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "operating_system": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy_configuration": _numpy_configuration(),
        },
        "inputs": {
            "n_left": args.n_left,
            "n_right": args.n_right,
            "latent_dim": args.latent_dim,
            "dtype": "float64",
            "c_contiguous": True,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "eps_distance": eps_distance,
            "pairwise_output_bytes": pairwise_output_bytes,
            "pairwise_kernel_peak_bytes": pairwise_kernel_peak_bytes,
            "private_ceiling_bytes": MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES,
            "broadcast_elements": broadcast_elements,
        },
        "bounded_coordinate_subtraction": bounded,
        "legacy_broadcast": None,
    }

    if broadcast_elements <= MAX_SAFE_BROADCAST_ELEMENTS:
        legacy = _measure(
            lambda: _legacy_broadcast(
                left,
                right,
                eps_distance=eps_distance,
            ),
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        bounded_values = _pairwise_euclidean_distances(
            left,
            right,
            eps_distance=eps_distance,
        )
        legacy_values = _legacy_broadcast(
            left,
            right,
            eps_distance=eps_distance,
        )
        legacy["maximum_absolute_difference"] = float(
            np.max(np.abs(bounded_values - legacy_values), initial=0.0)
        )
        report["legacy_broadcast"] = legacy
    else:
        report["legacy_broadcast_skip_reason"] = (
            f"{broadcast_elements} elements exceed the conservative "
            f"{MAX_SAFE_BROADCAST_ELEMENTS}-element benchmark safety ceiling"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the safe benchmark and emit one machine-readable JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-left", type=_positive_integer, default=256)
    parser.add_argument("--n-right", type=_positive_integer, default=512)
    parser.add_argument("--latent-dim", type=_positive_integer, default=2)
    parser.add_argument("--warmups", type=_positive_integer, default=2)
    parser.add_argument("--repetitions", type=_positive_integer, default=7)
    parser.add_argument("--seed", type=int, default=563)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
