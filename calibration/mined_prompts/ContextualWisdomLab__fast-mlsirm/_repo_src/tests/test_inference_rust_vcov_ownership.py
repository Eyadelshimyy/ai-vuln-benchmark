"""Fail-first Rust ownership contracts for covariance and standard errors."""

from __future__ import annotations

import numpy as np

import fast_mlsirm._core as core
from fast_mlsirm.inference import standard_errors_from_vcov, vcov_from_hessian


def test_vcov_from_hessian_delegates_numerical_result_to_rust(monkeypatch) -> None:
    """Matrix inversion/pseudoinversion must be owned by the Rust core."""
    calls: list[tuple[np.ndarray, float]] = []

    def fake_vcov(matrix: np.ndarray, rcond: float) -> list[float]:
        calls.append((np.asarray(matrix, dtype=np.float64).copy(), rcond))
        return [2.0, 0.25, 0.25, 3.0]

    monkeypatch.setattr(core, "vcov_from_hessian", fake_vcov, raising=False)
    hessian = np.array([[1.0, 0.1], [0.1, 1.5]], dtype=np.float64)

    result = vcov_from_hessian(hessian, rcond=1e-9)

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], hessian)
    assert calls[0][1] == 1e-9
    assert np.array_equal(result, np.array([[2.0, 0.25], [0.25, 3.0]]))


def test_standard_errors_from_vcov_delegate_to_rust(monkeypatch) -> None:
    """Covariance-diagonal reduction and square roots must be Rust-owned."""
    calls: list[np.ndarray] = []

    def fake_standard_errors(matrix: np.ndarray) -> list[float]:
        calls.append(np.asarray(matrix, dtype=np.float64).copy())
        return [0.75, 1.25]

    monkeypatch.setattr(core, "standard_errors_from_vcov", fake_standard_errors, raising=False)
    vcov = np.array([[4.0, 0.2], [0.2, 9.0]], dtype=np.float64)

    result = standard_errors_from_vcov(vcov)

    assert len(calls) == 1
    assert np.array_equal(calls[0], vcov)
    assert np.array_equal(result, np.array([0.75, 1.25]))
