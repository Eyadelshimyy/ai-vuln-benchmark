# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Wall-clock regression: PyO3 ``.detach`` must free the GIL for ThreadPool fits.

``run_bifactor_bootstrap(..., n_jobs>1)`` dispatches replicates on a
``ThreadPoolExecutor``. Without ``Python::detach`` around the Rust EM body,
workers serialize on the GIL and wall-clock speedup collapses to ~1.0×.
With detach, identical tiny bifactor fits must show clear parallel speedup.

Speedup floor (measured, not an unsourced default)
--------------------------------------------------
ADR-0028 forbids shipping unsourced numeric defaults on public study-facing
APIs; the same justification applies to this regression gate: the bound must
be the measured separation between GIL-held (~1.0–1.05×) and GIL-released
parallelism, not a magic constant.

Measured on Air late-life (2026-09-18) with the fixture below (4 identical
``fit_bifactor_grm`` calls, ``max_workers=2`` vs serial)::

    serial_s ≈ 2.4–3.2
    pool_s   ≈ 1.3–1.8
    speedup  ≈ 1.55–1.95×

GIL-held synthetic check (same fixture, temporary no-detach binding) stayed
at ≤1.08×. Floor ``MIN_PARALLEL_SPEEDUP = 1.25`` sits below the measured
detached distribution and above GIL-held noise.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fast_mlsirm.bifactor_grm import fit_bifactor_grm

# Measured floor — see module docstring (ADR-0028 justification).
MIN_PARALLEL_SPEEDUP = 1.25
N_PARALLEL = 4
N_WORKERS = 2


def _tiny_responses() -> tuple[np.ndarray, np.ndarray]:
    n_persons, n_items, n_cat = 40, 4, 3
    smap = np.array([0, 0, 1, 1], dtype=np.int64)
    rng = np.random.default_rng(20260918)
    y = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(np.float64)
    return y, smap


def _one_fit(y: np.ndarray, smap: np.ndarray) -> None:
    fit_bifactor_grm(
        y,
        smap,
        n_cat=3,
        n_specific=2,
        q_general=7,
        q_specific=7,
        max_iter=6,
        tol=1e-3,
        n_starts=1,
        seed=42,
        device="cpu",
    )


def test_threadpool_fit_bifactor_grm_beats_serial_wall_clock() -> None:
    """ThreadPoolExecutor must beat serial wall-clock when Rust detaches the GIL."""
    y, smap = _tiny_responses()
    # Warm-up (exclude first-call import/extension costs from the ratio).
    _one_fit(y, smap)

    t0 = time.perf_counter()
    for _ in range(N_PARALLEL):
        _one_fit(y, smap)
    serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        list(pool.map(lambda _: _one_fit(y, smap), range(N_PARALLEL)))
    pool_s = time.perf_counter() - t0

    speedup = serial_s / pool_s
    assert speedup >= MIN_PARALLEL_SPEEDUP, (
        f"expected ThreadPool speedup >= {MIN_PARALLEL_SPEEDUP:.2f}x "
        f"(ADR-0028 measured floor; GIL-held ≈1.0x); "
        f"got {speedup:.3f}x (serial={serial_s:.3f}s, pool={pool_s:.3f}s). "
        f"Missing py.detach around fit_bifactor_grm would explain this."
    )
