"""Fail-first resource contracts for response-time calibration iteration controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.config import MAX_MAX_ITER
from fast_mlsirm.rt import fit_response_times, fit_speed_accuracy


class _BombRtCore:
    """Compiled-core sentinel proving invalid controls fail before numerical work."""

    def fit_rt_lognormal(self, *args, **kwargs):
        raise AssertionError("standalone RT core reached with invalid max_iter")

    def fit_speed_accuracy_covariance(self, *args, **kwargs):
        raise AssertionError("joint RT core reached with invalid max_iter")


class _RecordingRtCore:
    """Return deterministic tiny fits while recording the marshalled iteration count."""

    def __init__(self) -> None:
        self.standalone_max_iter: list[int] = []
        self.joint_max_iter: list[int] = []

    def fit_rt_lognormal(self, *args, **kwargs):
        self.standalone_max_iter.append(args[4])
        return {
            "alpha": [1.0, 1.0],
            "beta": [0.0, 0.0],
            "mu_tau": 0.0,
            "sigma_tau": 1.0,
            "tau_eap": [0.0, 0.0],
            "tau_sd": [1.0, 1.0],
            "loglik": 0.0,
            "loglik_trace": [0.0],
            "n_iter": 1,
            "converged": True,
            "termination_reason": "converged",
            "final_loglik_change": 0.0,
        }

    def fit_speed_accuracy_covariance(self, *args, **kwargs):
        self.joint_max_iter.append(args[10])
        return {
            "rho": 0.0,
            "sigma_tau": 1.0,
            "s_theta2": 1.0,
            "theta_eap": [0.0, 0.0],
            "tau_eap": [0.0, 0.0],
            "loglik": 0.0,
            "loglik_trace": [0.0],
            "n_iter": 1,
            "converged": True,
            "termination_reason": "converged",
            "final_loglik_change": 0.0,
        }


def _install_bomb_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a compiled-core sentinel with both RT entrypoints present."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _BombRtCore())


def _install_recording_core(monkeypatch: pytest.MonkeyPatch) -> _RecordingRtCore:
    """Install a deterministic core that records validated public controls."""
    core = _RecordingRtCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    return core


def _rt_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tiny valid standalone/joint RT inputs."""
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    times = np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64)
    a = np.array([1.0, 1.2], dtype=np.float64)
    b = np.array([0.0, -0.2], dtype=np.float64)
    alpha = np.array([1.1, 1.3], dtype=np.float64)
    beta = np.array([0.1, 0.2], dtype=np.float64)
    return responses, times, a, b, alpha, beta


@pytest.mark.parametrize("bad_max_iter", [False, np.bool_(True), 0, 1.5, MAX_MAX_ITER + 1])
def test_fit_response_times_rejects_invalid_max_iter_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    bad_max_iter: object,
) -> None:
    """Standalone RT calibration must validate exact bounded integers before Rust."""
    _install_bomb_core(monkeypatch)
    _, times, *_ = _rt_inputs()

    with pytest.raises(ValueError, match=rf"max_iter must be an integer in 1\.\.{MAX_MAX_ITER}"):
        fit_response_times(times, max_iter=bad_max_iter)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_max_iter", [False, np.bool_(True), 0, 1.5, MAX_MAX_ITER + 1])
def test_fit_speed_accuracy_rejects_invalid_max_iter_before_rust(
    monkeypatch: pytest.MonkeyPatch,
    bad_max_iter: object,
) -> None:
    """Joint RT calibration must enforce the same exact bounded integer contract."""
    _install_bomb_core(monkeypatch)
    responses, times, a, b, alpha, beta = _rt_inputs()

    with pytest.raises(ValueError, match=rf"max_iter must be an integer in 1\.\.{MAX_MAX_ITER}"):
        fit_speed_accuracy(
            responses,
            times,
            a,
            b,
            alpha,
            beta,
            max_iter=bad_max_iter,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("accepted", [MAX_MAX_ITER - 1, MAX_MAX_ITER, np.int64(MAX_MAX_ITER)])
def test_fit_response_times_preserves_accepted_integer_controls(
    monkeypatch: pytest.MonkeyPatch,
    accepted: object,
) -> None:
    """Accepted Python/NumPy integers must reach Rust unchanged as bounded integers."""
    core = _install_recording_core(monkeypatch)
    _, times, *_ = _rt_inputs()

    fit_response_times(times, max_iter=accepted)  # type: ignore[arg-type]

    assert core.standalone_max_iter == [int(accepted)]


@pytest.mark.parametrize("accepted", [MAX_MAX_ITER - 1, MAX_MAX_ITER, np.int64(MAX_MAX_ITER)])
def test_fit_speed_accuracy_preserves_accepted_integer_controls(
    monkeypatch: pytest.MonkeyPatch,
    accepted: object,
) -> None:
    """The joint fitter must marshal the same bounded integer control without truncation."""
    core = _install_recording_core(monkeypatch)
    responses, times, a, b, alpha, beta = _rt_inputs()

    fit_speed_accuracy(
        responses,
        times,
        a,
        b,
        alpha,
        beta,
        max_iter=accepted,  # type: ignore[arg-type]
    )

    assert core.joint_max_iter == [int(accepted)]
