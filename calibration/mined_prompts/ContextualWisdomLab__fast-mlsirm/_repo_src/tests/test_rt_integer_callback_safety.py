"""Callback-safety regressions for response-time integer controls."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.config import MAX_MAX_ITER
from fast_mlsirm.rt import fit_response_times, fit_speed_accuracy


class _BombRtCore:
    """Native-core sentinel proving invalid controls fail before dispatch."""

    def fit_rt_lognormal(self, *args, **kwargs):
        raise AssertionError("standalone RT core reached with untrusted max_iter")

    def fit_speed_accuracy_covariance(self, *args, **kwargs):
        raise AssertionError("joint RT core reached with untrusted max_iter")


class _HostileIndex:
    """Arbitrary integer protocol provider whose callback must never execute."""

    calls = 0

    def __index__(self) -> int:
        """Record forbidden coercion and return an otherwise valid count."""
        type(self).calls += 1
        return 2


class _HostileInt(int):
    """Caller-defined Python integer subclass outside the trusted boundary."""

    calls = 0

    def __int__(self) -> int:
        """Record forbidden normalization if validation admits this subclass."""
        type(self).calls += 1
        return int.__int__(self)


class _HostileNumpyInt(np.int64):
    """Caller-defined NumPy integer subclass outside the trusted boundary."""

    calls = 0

    def __int__(self) -> int:
        """Record forbidden normalization if validation admits this subclass."""
        type(self).calls += 1
        return np.int64.__int__(self)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the smallest valid inputs needed by both public RT fitters."""
    responses = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    times = np.array([[1.0, 1.5], [1.2, 1.8]], dtype=np.float64)
    a = np.array([1.0, 1.2], dtype=np.float64)
    b = np.array([0.0, -0.2], dtype=np.float64)
    alpha = np.array([1.1, 1.3], dtype=np.float64)
    beta = np.array([0.1, 0.2], dtype=np.float64)
    return responses, times, a, b, alpha, beta


def _install_bomb_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install both RT entrypoints so ordering is observable."""
    monkeypatch.setattr(fitstats, "_core_module", lambda: _BombRtCore())


def _assert_standalone_rejects_without_callback(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    callback_owner: type,
) -> None:
    """Require standalone validation to reject before callbacks/native work."""
    _install_bomb_core(monkeypatch)
    callback_owner.calls = 0
    _, times, *_ = _inputs()
    with pytest.raises(
        ValueError,
        match=rf"max_iter must be an integer in 1\.\.{MAX_MAX_ITER}",
    ):
        fit_response_times(times, max_iter=value)  # type: ignore[arg-type]
    assert callback_owner.calls == 0


def _assert_joint_rejects_without_callback(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    callback_owner: type,
) -> None:
    """Require joint validation to reject before callbacks/native work."""
    _install_bomb_core(monkeypatch)
    callback_owner.calls = 0
    responses, times, a, b, alpha, beta = _inputs()
    with pytest.raises(
        ValueError,
        match=rf"max_iter must be an integer in 1\.\.{MAX_MAX_ITER}",
    ):
        fit_speed_accuracy(
            responses,
            times,
            a,
            b,
            alpha,
            beta,
            max_iter=value,  # type: ignore[arg-type]
        )
    assert callback_owner.calls == 0


@pytest.mark.parametrize(
    ("value", "callback_owner"),
    [
        (_HostileIndex(), _HostileIndex),
        (_HostileInt(2), _HostileInt),
        (_HostileNumpyInt(2), _HostileNumpyInt),
    ],
)
def test_fit_response_times_rejects_untrusted_integer_controls_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    callback_owner: type,
) -> None:
    """Standalone RT max-iteration validation is callback-inert."""
    _assert_standalone_rejects_without_callback(monkeypatch, value, callback_owner)


@pytest.mark.parametrize(
    ("value", "callback_owner"),
    [
        (_HostileIndex(), _HostileIndex),
        (_HostileInt(2), _HostileInt),
        (_HostileNumpyInt(2), _HostileNumpyInt),
    ],
)
def test_fit_speed_accuracy_rejects_untrusted_integer_controls_before_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    callback_owner: type,
) -> None:
    """Joint RT max-iteration validation is callback-inert."""
    _assert_joint_rejects_without_callback(monkeypatch, value, callback_owner)
