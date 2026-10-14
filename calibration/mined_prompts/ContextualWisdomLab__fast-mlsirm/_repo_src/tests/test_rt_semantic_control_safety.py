"""Trust-boundary regressions for response-time calibration controls."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.rt import fit_response_times, fit_speed_accuracy


class _BombRtCore:
    """Native-core sentinel proving invalid controls fail before dispatch."""

    def fit_rt_lognormal(self, *args, **kwargs):
        """Fail if standalone RT calibration reaches native code."""
        raise AssertionError("standalone RT core reached before control rejection")

    def fit_speed_accuracy_covariance(self, *args, **kwargs):
        """Fail if joint RT calibration reaches native code."""
        raise AssertionError("joint RT core reached before control rejection")


class _HostileFloat(float):
    """Caller-defined real scalar whose coercion callback must stay dormant."""

    calls = 0

    def __float__(self) -> float:
        """Record forbidden normalization and return an otherwise valid value."""
        type(self).calls += 1
        return float.__float__(self)


class _HostileInt(int):
    """Caller-defined integer scalar whose coercion callback must stay dormant."""

    calls = 0

    def __int__(self) -> int:
        """Record forbidden normalization and return an otherwise valid value."""
        type(self).calls += 1
        return int.__int__(self)


class _HostileBoolProtocol:
    """Arbitrary truth-value provider whose callback must never execute."""

    calls = 0

    def __bool__(self) -> bool:
        """Record forbidden truth-value coercion."""
        type(self).calls += 1
        return True


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the smallest valid arrays needed by both public RT calibrators."""
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    times = np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64)
    a = np.array([1.0, 1.2], dtype=np.float64)
    b = np.array([0.0, -0.2], dtype=np.float64)
    alpha = np.array([1.1, 1.3], dtype=np.float64)
    beta = np.array([0.1, 0.2], dtype=np.float64)
    return responses, times, a, b, alpha, beta


def _install_bomb_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make native dispatch observable without executing numerical work."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _BombRtCore())


@pytest.mark.parametrize("control", ["tol", "var_floor", "sigma_floor", "fix_sigma_tau"])
def test_fit_response_times_rejects_untrusted_real_controls_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    """Standalone RT semantic controls reject caller-defined real subclasses."""
    _install_bomb_core(monkeypatch)
    _HostileFloat.calls = 0
    _, times, *_ = _inputs()

    with pytest.raises(ValueError, match=rf"{control} must be positive and finite"):
        fit_response_times(times, **{control: _HostileFloat(1.0)})

    assert _HostileFloat.calls == 0


def test_fit_response_times_rejects_truth_protocol_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone convergence policy requires an exact Boolean."""
    _install_bomb_core(monkeypatch)
    _HostileBoolProtocol.calls = 0
    _, times, *_ = _inputs()

    with pytest.raises(ValueError, match="require_convergence must be a Boolean"):
        fit_response_times(
            times,
            require_convergence=_HostileBoolProtocol(),  # type: ignore[arg-type]
        )

    assert _HostileBoolProtocol.calls == 0


def test_fit_speed_accuracy_rejects_integer_subclass_q_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint quadrature admission must not coerce caller-defined integers."""
    _install_bomb_core(monkeypatch)
    _HostileInt.calls = 0
    responses, times, a, b, alpha, beta = _inputs()

    with pytest.raises(ValueError, match="q must be an integer"):
        fit_speed_accuracy(
            responses,
            times,
            a,
            b,
            alpha,
            beta,
            q=_HostileInt(21),
        )

    assert _HostileInt.calls == 0


def test_fit_speed_accuracy_rejects_fractional_q_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A floating quadrature count may not silently narrow to a supported rule."""
    _install_bomb_core(monkeypatch)
    responses, times, a, b, alpha, beta = _inputs()

    with pytest.raises(ValueError, match="q must be an integer"):
        fit_speed_accuracy(responses, times, a, b, alpha, beta, q=21.9)  # type: ignore[arg-type]


@pytest.mark.parametrize("control", ["tol", "fix_sigma_tau"])
def test_fit_speed_accuracy_rejects_untrusted_real_controls_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
) -> None:
    """Joint RT semantic controls reject caller-defined real subclasses."""
    _install_bomb_core(monkeypatch)
    _HostileFloat.calls = 0
    responses, times, a, b, alpha, beta = _inputs()

    with pytest.raises(ValueError, match=rf"{control} must be positive and finite"):
        fit_speed_accuracy(
            responses,
            times,
            a,
            b,
            alpha,
            beta,
            **{control: _HostileFloat(1.0)},
        )

    assert _HostileFloat.calls == 0


def test_fit_speed_accuracy_rejects_truth_protocol_before_native_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joint convergence policy requires an exact Boolean."""
    _install_bomb_core(monkeypatch)
    _HostileBoolProtocol.calls = 0
    responses, times, a, b, alpha, beta = _inputs()

    with pytest.raises(ValueError, match="require_convergence must be a Boolean"):
        fit_speed_accuracy(
            responses,
            times,
            a,
            b,
            alpha,
            beta,
            require_convergence=_HostileBoolProtocol(),  # type: ignore[arg-type]
        )

    assert _HostileBoolProtocol.calls == 0


@pytest.mark.parametrize("function", [fit_response_times, fit_speed_accuracy])
def test_rt_public_calibrators_do_not_use_runtime_asserts(function) -> None:
    """Runtime validation must remain active under Python optimization flags."""
    tree = ast.parse(inspect.getsource(function))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
