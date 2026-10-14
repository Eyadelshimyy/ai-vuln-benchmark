"""Recovery and interval-coverage evidence for Rust-owned covariance numerics.

The production covariance inverse and diagonal standard-error reduction remain in
Rust.  This module uses a closed-form Gaussian location model only as a
reference/test oracle so bias, MAE, RMSE, and nominal interval coverage can be
checked without duplicating production psychometric arithmetic in Python.
"""

from __future__ import annotations

import math

import numpy as np

from fast_mlsirm.inference import standard_errors_from_vcov, vcov_from_hessian


def test_rust_vcov_standard_error_has_known_normal_mean_recovery_and_coverage() -> None:
    """Rust covariance/SE output supports calibrated Wald intervals in a known model.

    For a Normal location model with known variance, the observed information for
    the mean is exactly ``n / sigma**2``.  The sample mean and Monte Carlo metrics
    below are explicit test/reference formulas; the result-affecting covariance
    inversion and standard-error extraction exercised by the public package are
    delegated to the compiled Rust core.
    """
    rng = np.random.default_rng(20260815)
    replicates = 4096
    sample_size = 64
    true_mean = 0.3
    sigma = 1.2

    samples = rng.normal(true_mean, sigma, size=(replicates, sample_size))
    estimates = np.mean(samples, axis=1)
    errors = estimates - true_mean

    information = np.array([[sample_size / sigma**2]], dtype=np.float64)
    vcov = vcov_from_hessian(information)
    standard_error = float(standard_errors_from_vcov(vcov)[0])

    bias = float(np.mean(errors))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    expected_standard_error = sigma / math.sqrt(sample_size)
    covered = np.logical_and(
        true_mean >= estimates - 1.96 * standard_error,
        true_mean <= estimates + 1.96 * standard_error,
    )
    coverage = float(np.mean(covered))

    assert np.isclose(standard_error, expected_standard_error, rtol=1e-12, atol=1e-12)
    assert abs(bias) < 0.02
    assert 0.10 < mae < 0.14
    assert 0.13 < rmse < 0.17
    assert 0.93 < coverage < 0.97
