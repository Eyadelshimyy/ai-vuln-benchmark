# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU/GPU E-step parity at study-precision quadrature grids (121/241/481 nodes).

The maintainer standard (issue #1929) requires study settings to use at least
121 Gauss-Hermite nodes per dimension, with larger counts (241, 481, ...)
until estimates stabilize: node count controls the numerical precision of
marginal-likelihood integration, so counts are caller arguments with no
defaults and no caps (arbitrary-``n`` rules generated on demand via Golub &
Welsch, 1969). These tests assert the stage-5 GPU E-step agrees with the
``f64`` CPU reference at those study-precision grids — not just at the small
smoke-test grids of ``tests/test_bifactor_gpu.py`` — and that the CPU fit at
121 nodes agrees with the CPU fit at 241 nodes (marginal-likelihood integral
convergence, mirroring the ``#[ignore]``d
``bifactor_grm_121_vs_241_nodes_agree`` Rust regression).

The q=241 / q=481 legs also exercise the Metal/WebGPU workgroup-dimension
split in ``gpu_bifactor``: Apple Metal rejects 1-D dispatches above 65535
workgroups (AC late-life bootstrap at q=241 hit 141573 on
``reduce_counts_blk``), so the host factors the grid across x/y/z from the
adapter's ``max_compute_workgroups_per_dimension``.

Runtime note: ``q_general * q_specific`` grids (14,641 nodes at 121;
58,081 at 241; 231,361 at 481) make these fits orders of magnitude heavier
than the smoke tests, so they are gated behind ``STAGE5_HIGH_Q=1`` (mirroring
the Rust ``#[ignore]`` convention for slow node-count regressions) and use a
tiny fixture (48 persons, 6 items, 2 specific blocks) with few EM iterations.
Run with ``STAGE5_HIGH_Q=1 pytest tests/test_bifactor_gpu_high_q.py -s`` to
execute and print the measured wall times for the PR record.

Tolerance derivation: same single-precision envelope as
``tests/test_bifactor_gpu.py`` — the WGSL kernels accumulate in ``f32``
(machine epsilon ~1.19e-7) while the CPU reference is ``f64``; fit-level
agreement is asserted at 1e-4 absolute for slopes/thresholds and 1e-3
absolute for the loglik. The 121-vs-241 CPU agreement uses the 5e-3
loglik band of the two-tier node-agreement regression (a numerical
agreement check, not truth recovery).

Implementation basis: Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D.
J., Segawa, E., Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V.
J., & Stover, A. (2007). Full-information item bifactor analysis of graded
response data. *Applied Psychological Measurement, 31*(1), 4-19.
https://doi.org/10.1177/0146621606289485; Bock, R. D., & Aitkin, M. (1981).
Marginal maximum likelihood estimation of item parameters: Application of
an EM algorithm. *Psychometrika, 46*(4), 443-459.
https://doi.org/10.1007/BF02293801; Golub, G. H., & Welsch, J. H. (1969).
Calculation of Gauss quadrature rules. *Mathematics of Computation, 23*,
221-230. https://doi.org/10.1090/S0025-5718-1969-0245201-X
"""

import os
import time

import numpy as np
import pytest

from fast_mlsirm.bifactor_grm import fit_bifactor_grm

SLOPE_ATOL = 1e-4
THRESHOLD_ATOL = 1e-4
LOGLIK_ATOL = 1e-3
# Numerical-agreement band for the 121-vs-241 CPU comparison (same 5e-3 band
# as the two-tier node-agreement regression).
NODE_AGREE_LOGLIK_ATOL = 5e-3
NODE_AGREE_PARAM_ATOL = 5e-3

needs_high_q = pytest.mark.skipif(
    os.environ.get("STAGE5_HIGH_Q") != "1",
    reason="study-precision grids (121/241 nodes); rerun with STAGE5_HIGH_Q=1",
)


def _fixture(n_persons=48, n_items=6, n_cat=3, seed=20260917):
    smap = np.zeros(n_items, dtype=np.int64)
    smap[n_items // 2 :] = 1
    rng = np.random.default_rng(seed)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items)).astype(float)
    # Reverse-keyed items (measurement-relevant fixture asymmetry).
    responses[:, 1] = (n_cat - 1) - responses[:, 1]
    responses[:, n_items // 2 + 1] = (n_cat - 1) - responses[
        :, n_items // 2 + 1
    ]
    return responses, smap, n_cat


def _fit_kwargs(q, max_iter=8):
    return dict(
        n_cat=3,
        n_specific=2,
        q_general=q,
        q_specific=q,
        max_iter=max_iter,
        tol=1e-3,
        n_starts=1,
        seed=7,
    )


def _report(tag, cpu_time, gpu_time, fit_cpu, fit_gpu):
    slope_diff = float(
        max(
            np.max(np.abs(fit_cpu.a_general - fit_gpu.a_general)),
            np.max(np.abs(fit_cpu.a_specific - fit_gpu.a_specific)),
        )
    )
    threshold_diff = float(np.max(np.abs(fit_cpu.threshold - fit_gpu.threshold)))
    loglik_diff = abs(float(fit_cpu.loglik_trace[-1]) - float(fit_gpu.loglik_trace[-1]))
    print(
        f"\n[{tag}] CPU {cpu_time:.3f}s vs GPU {gpu_time:.3f}s; "
        f"max|Δslope|={slope_diff:.3e} max|Δthreshold|={threshold_diff:.3e} "
        f"|Δloglik|={loglik_diff:.3e} "
        f"(iters cpu={fit_cpu.n_iter}/gpu={fit_gpu.n_iter} "
        f"converged cpu={fit_cpu.converged}/gpu={fit_gpu.converged})"
    )
    return slope_diff, threshold_diff, loglik_diff


@needs_high_q
def test_bifactor_gpu_parity_q121(capfd):
    """CPU and GPU E-steps agree within f32 tolerance at q=121 per dim."""
    responses, smap, _ = _fixture()
    kw = _fit_kwargs(121)

    t0 = time.perf_counter()
    fit_cpu = fit_bifactor_grm(responses, smap, **kw, device="cpu")
    t1 = time.perf_counter()
    fit_gpu = fit_bifactor_grm(responses, smap, **kw, device="gpu")
    t2 = time.perf_counter()

    slope_diff, threshold_diff, loglik_diff = _report(
        "single q=121", t1 - t0, t2 - t1, fit_cpu, fit_gpu
    )

    assert fit_cpu.converged == fit_gpu.converged
    assert fit_cpu.n_iter == fit_gpu.n_iter
    assert fit_cpu.best_start == fit_gpu.best_start
    assert slope_diff <= SLOPE_ATOL
    assert threshold_diff <= THRESHOLD_ATOL
    assert loglik_diff <= LOGLIK_ATOL


@needs_high_q
def test_bifactor_gpu_parity_q241(capfd):
    """CPU and GPU E-steps agree within f32 tolerance at q=241 per dim."""
    responses, smap, _ = _fixture()
    kw = _fit_kwargs(241)

    t0 = time.perf_counter()
    fit_cpu = fit_bifactor_grm(responses, smap, **kw, device="cpu")
    t1 = time.perf_counter()
    fit_gpu = fit_bifactor_grm(responses, smap, **kw, device="gpu")
    t2 = time.perf_counter()

    slope_diff, threshold_diff, loglik_diff = _report(
        "single q=241", t1 - t0, t2 - t1, fit_cpu, fit_gpu
    )

    assert fit_cpu.converged == fit_gpu.converged
    assert fit_cpu.n_iter == fit_gpu.n_iter
    assert fit_cpu.best_start == fit_gpu.best_start
    assert slope_diff <= SLOPE_ATOL
    assert threshold_diff <= THRESHOLD_ATOL
    assert loglik_diff <= LOGLIK_ATOL


@needs_high_q
def test_bifactor_gpu_parity_q481(capfd):
    """CPU and GPU E-steps agree within f32 tolerance at q=481 per dim."""
    responses, smap, _ = _fixture()
    kw = _fit_kwargs(481)

    t0 = time.perf_counter()
    fit_cpu = fit_bifactor_grm(responses, smap, **kw, device="cpu")
    t1 = time.perf_counter()
    fit_gpu = fit_bifactor_grm(responses, smap, **kw, device="gpu")
    t2 = time.perf_counter()

    slope_diff, threshold_diff, loglik_diff = _report(
        "single q=481", t1 - t0, t2 - t1, fit_cpu, fit_gpu
    )

    assert fit_cpu.converged == fit_gpu.converged
    assert fit_cpu.n_iter == fit_gpu.n_iter
    assert fit_cpu.best_start == fit_gpu.best_start
    assert slope_diff <= SLOPE_ATOL
    assert threshold_diff <= THRESHOLD_ATOL
    assert loglik_diff <= LOGLIK_ATOL


@needs_high_q
def test_bifactor_gpu_parity_q241_wide_items_metal_workgroups(capfd):
    """q=241 with enough items to force x/y workgroup split on Metal.

    AC late-life multigroup bootstrap failed with dispatch [141573,1,1] on
    ``reduce_counts_blk``. A 26-item × 241×241×3 grid needs ~70815 workgroups
    (>65535), so this leg exercises the runtime 2-D factoring path.
    """
    responses, smap, _ = _fixture(n_persons=32, n_items=26, n_cat=3, seed=20260918)
    kw = _fit_kwargs(241, max_iter=4)

    t0 = time.perf_counter()
    fit_cpu = fit_bifactor_grm(responses, smap, **kw, device="cpu")
    t1 = time.perf_counter()
    fit_gpu = fit_bifactor_grm(responses, smap, **kw, device="gpu")
    t2 = time.perf_counter()

    slope_diff, threshold_diff, loglik_diff = _report(
        "wide q=241 (Metal 2-D dispatch)", t1 - t0, t2 - t1, fit_cpu, fit_gpu
    )

    assert fit_cpu.converged == fit_gpu.converged
    assert fit_cpu.n_iter == fit_gpu.n_iter
    assert fit_cpu.best_start == fit_gpu.best_start
    assert slope_diff <= SLOPE_ATOL
    assert threshold_diff <= THRESHOLD_ATOL
    assert loglik_diff <= LOGLIK_ATOL


@needs_high_q
def test_bifactor_cpu_q121_vs_q241_agree():
    """CPU fits at 121 and 241 nodes agree (integral-convergence check)."""
    responses, smap, _ = _fixture()
    t0 = time.perf_counter()
    fit121 = fit_bifactor_grm(responses, smap, **_fit_kwargs(121), device="cpu")
    t1 = time.perf_counter()
    fit241 = fit_bifactor_grm(responses, smap, **_fit_kwargs(241), device="cpu")
    t2 = time.perf_counter()

    loglik_diff = abs(
        float(fit121.loglik_trace[-1]) - float(fit241.loglik_trace[-1])
    )
    slope_diff = float(
        max(
            np.max(np.abs(fit121.a_general - fit241.a_general)),
            np.max(np.abs(fit121.a_specific - fit241.a_specific)),
        )
    )
    threshold_diff = float(
        np.max(np.abs(fit121.threshold - fit241.threshold))
    )
    print(
        f"\n[q121 vs q241 CPU] q121 {t1 - t0:.3f}s vs q241 {t2 - t1:.3f}s; "
        f"max|Δslope|={slope_diff:.3e} max|Δthreshold|={threshold_diff:.3e} "
        f"|Δloglik|={loglik_diff:.3e}"
    )
    assert loglik_diff <= NODE_AGREE_LOGLIK_ATOL
    assert slope_diff <= NODE_AGREE_PARAM_ATOL
    assert threshold_diff <= NODE_AGREE_PARAM_ATOL
