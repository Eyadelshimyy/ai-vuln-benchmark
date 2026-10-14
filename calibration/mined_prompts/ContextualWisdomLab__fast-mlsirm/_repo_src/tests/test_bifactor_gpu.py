# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""CPU/GPU E-step equivalence for the Bock-Aitkin bifactor GRM.

Tolerance derivation (documented, from float precision)
-------------------------------------------------------
The WGSL kernels accumulate in f32 (machine epsilon ≈ 1.19e-7) while the CPU
reference sweep is f64. Expected counts are reductions over persons of
posterior weights in [0, 1] (worst-case per-entry error ~ ``n_persons`` ×
eps ≈ 1.4e-5 here); the marginal log-likelihood sums per-person terms every
EM iteration. Fit-level agreement is asserted at 1e-4 absolute for slopes
and thresholds (about three orders of magnitude above the measured f32-level
differences of ~2e-7) and 1e-3 absolute for the loglik (measured 4.7e-5);
convergence paths (iterations, flags) must agree exactly.

Quadrature follows the merged estimator contract: Gauss-Hermite counts are
caller arguments (any integer ``>= 1``, generated on demand via Golub &
Welsch, 1969 — no fixed-table cap, issue #1929) with no defaults; smoke
equivalence runs at small grids while study-precision parity runs at the
maintainer-standard 121- and 241-point grids (chosen by precision
convergence). Node counts are never capped.

Implementation basis: Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D.
J., Segawa, E., Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V.
J., & Stover, A. (2007). Full-information item bifactor analysis of graded
response data. *Applied Psychological Measurement, 31*(1), 4–19.
https://doi.org/10.1177/0146621606289485; Bock, R. D., & Aitkin, M. (1981).
Marginal maximum likelihood estimation of item parameters: Application of
an EM algorithm. *Psychometrika, 46*(4), 443–459.
https://doi.org/10.1007/BF02293801
"""

import time

import numpy as np

from fast_mlsirm.bifactor_grm import fit_bifactor_grm
from fast_mlsirm.bifactor_multigroup import fit_bifactor_grm_multigroup

SLOPE_ATOL = 1e-4
THRESHOLD_ATOL = 1e-4
LOGLIK_ATOL = 1e-3


def _fixture(n_persons=120, n_items=8, n_cat=3, seed=42):
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


def test_bifactor_gpu_equivalence_single_group(capfd):
    """CPU and GPU E-steps agree within f32-derived tolerance (single group)."""
    responses, smap, n_cat = _fixture()
    kw = dict(
        n_cat=n_cat,
        n_specific=2,
        q_general=21,
        q_specific=21,
        max_iter=25,
        tol=1e-4,
        n_starts=1,
        seed=1,
    )

    t0 = time.perf_counter()
    fit_cpu = fit_bifactor_grm(responses, smap, **kw, device="cpu")
    t1 = time.perf_counter()
    fit_gpu = fit_bifactor_grm(responses, smap, **kw, device="gpu")
    t2 = time.perf_counter()

    cpu_time, gpu_time = t1 - t0, t2 - t1
    slope_diff = float(
        max(
            np.max(np.abs(fit_cpu.a_general - fit_gpu.a_general)),
            np.max(np.abs(fit_cpu.a_specific - fit_gpu.a_specific)),
        )
    )
    threshold_diff = float(np.max(np.abs(fit_cpu.threshold - fit_gpu.threshold)))
    loglik_diff = abs(float(fit_cpu.loglik_trace[-1]) - float(fit_gpu.loglik_trace[-1]))

    err = capfd.readouterr().err
    gpu_executed = "falling back" not in err
    print(
        f"\n[single q=21] CPU {cpu_time:.3f}s vs GPU {gpu_time:.3f}s "
        f"(gpu_executed={gpu_executed}); max|Δslope|={slope_diff:.3e} "
        f"max|Δthreshold|={threshold_diff:.3e} |Δloglik|={loglik_diff:.3e}"
    )

    # Convergence path must agree exactly (same iterations, same outcome).
    assert fit_cpu.converged == fit_gpu.converged
    assert fit_cpu.n_iter == fit_gpu.n_iter
    assert fit_cpu.best_start == fit_gpu.best_start
    np.testing.assert_allclose(fit_cpu.a_general, fit_gpu.a_general, atol=SLOPE_ATOL)
    np.testing.assert_allclose(
        fit_cpu.a_specific, fit_gpu.a_specific, atol=SLOPE_ATOL
    )
    np.testing.assert_allclose(
        fit_cpu.threshold, fit_gpu.threshold, atol=THRESHOLD_ATOL
    )
    assert loglik_diff <= LOGLIK_ATOL


def test_bifactor_gpu_equivalence_multigroup(capfd):
    """CPU and GPU E-steps agree within f32-derived tolerance (two groups)."""
    responses, smap, n_cat = _fixture()
    n_persons = responses.shape[0]
    group = np.zeros(n_persons, dtype=np.int64)
    group[n_persons // 2 :] = 1
    kw = dict(
        n_cat=n_cat,
        n_specific=2,
        q_general=11,
        q_specific=11,
        max_iter=15,
        tol=1e-4,
        n_starts=1,
        seed=1,
    )

    t0 = time.perf_counter()
    fit_cpu = fit_bifactor_grm_multigroup(responses, group, smap, **kw, device="cpu")
    t1 = time.perf_counter()
    fit_gpu = fit_bifactor_grm_multigroup(responses, group, smap, **kw, device="gpu")
    t2 = time.perf_counter()

    cpu_time, gpu_time = t1 - t0, t2 - t1
    slope_diff = float(np.max(np.abs(fit_cpu.a_general - fit_gpu.a_general)))
    threshold_diff = float(np.max(np.abs(fit_cpu.threshold - fit_gpu.threshold)))
    loglik_diff = abs(float(fit_cpu.loglik_trace[-1]) - float(fit_gpu.loglik_trace[-1]))

    err = capfd.readouterr().err
    gpu_executed = "falling back" not in err
    print(
        f"\n[multi q=11] CPU {cpu_time:.3f}s vs GPU {gpu_time:.3f}s "
        f"(gpu_executed={gpu_executed}); max|Δslope|={slope_diff:.3e} "
        f"max|Δthreshold|={threshold_diff:.3e} |Δloglik|={loglik_diff:.3e}"
    )

    assert fit_cpu.converged == fit_gpu.converged
    assert fit_cpu.n_iter == fit_gpu.n_iter
    np.testing.assert_allclose(fit_cpu.a_general, fit_gpu.a_general, atol=SLOPE_ATOL)
    np.testing.assert_allclose(
        fit_cpu.a_specific, fit_gpu.a_specific, atol=SLOPE_ATOL
    )
    np.testing.assert_allclose(
        fit_cpu.threshold, fit_gpu.threshold, atol=THRESHOLD_ATOL
    )
    assert loglik_diff <= LOGLIK_ATOL
    np.testing.assert_allclose(
        fit_cpu.general_mean, fit_gpu.general_mean, atol=SLOPE_ATOL
    )
