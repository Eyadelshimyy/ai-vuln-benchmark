"""Callback-free trust-boundary regressions for factor-rotation controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.rotation as rotation


class _HostileString(str):
    """String subclass whose normalization callback must remain unreachable."""

    calls = 0

    def strip(self, chars: str | None = None) -> str:
        type(self).calls += 1
        raise AssertionError("caller string callback must not execute")


class _HostileInteger:
    """Integer-protocol object whose conversion callbacks must remain unreachable."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller integer callback must not execute")

    def __index__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller index callback must not execute")


class _HostileReal:
    """Real-protocol object whose conversion callback must remain unreachable."""

    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller float callback must not execute")


class _HostileBoolean:
    """Boolean-protocol object whose conversion callback must remain unreachable."""

    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("caller bool callback must not execute")


@pytest.fixture
def loadings() -> np.ndarray:
    """Return a small valid loading matrix for public-boundary tests."""

    return np.asarray([[0.8, 0.2], [0.7, 0.1], [0.1, 0.8], [0.2, 0.7]])


def _deny_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace native-core discovery with a probe that must remain untouched."""

    calls: list[int] = []

    def _probe() -> object:
        calls.append(1)
        raise AssertionError("rotation core must not be discovered for invalid controls")

    monkeypatch.setattr(rotation, "rotation_core", _probe)
    return calls


def _assert_rejected_without_callback(
    monkeypatch: pytest.MonkeyPatch,
    hostile_type: type,
    call: Callable[[object], object],
    *,
    value: object | None = None,
) -> None:
    """Assert package-owned rejection precedes caller callbacks and Rust discovery."""

    hostile_type.calls = 0
    core_calls = _deny_core_discovery(monkeypatch)
    candidate = hostile_type() if value is None else value
    with pytest.raises(ValueError):
        call(candidate)
    assert hostile_type.calls == 0
    assert core_calls == []


@pytest.mark.parametrize(
    "call",
    (
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, value),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(loadings, value),
    ),
)
def test_rotation_rejects_criterion_subclass_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    loadings: np.ndarray,
    call: Callable[[np.ndarray, object], object],
) -> None:
    """Criterion normalization rejects string subclasses before ``strip`` or Rust."""

    _HostileString.calls = 0
    core_calls = _deny_core_discovery(monkeypatch)
    with pytest.raises(ValueError, match="criterion must be a non-empty string"):
        call(loadings, _HostileString("varimax"))
    assert _HostileString.calls == 0
    assert core_calls == []


def test_rotation_rejects_boolean_protocol_normalize_before_core(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray
) -> None:
    """The boolean normalize control rejects arbitrary protocols before native discovery."""

    _assert_rejected_without_callback(
        monkeypatch,
        _HostileBoolean,
        lambda value: rotation.rotate_factor_loadings(loadings, normalize=value),
    )


@pytest.mark.parametrize(
    "call",
    (
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, n_starts=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, seed=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, max_iter=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, function_window=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, max_line_search=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, max_threads=value),
    ),
)
def test_rotation_rejects_integer_protocol_controls_before_core(
    monkeypatch: pytest.MonkeyPatch,
    loadings: np.ndarray,
    call: Callable[[np.ndarray, object], object],
) -> None:
    """Integer controls reject arbitrary conversion protocols before native discovery."""

    _assert_rejected_without_callback(
        monkeypatch, _HostileInteger, lambda value: call(loadings, value)
    )


@pytest.mark.parametrize(
    "call",
    (
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, tolerance=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, basin_tolerance=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, criterion="cf", kappa=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, criterion="oblimin", gamma=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, criterion="geomin", delta=value),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(loadings, "cf", kappa=value),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(loadings, "oblimin", gamma=value),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(loadings, "geomin", delta=value),
    ),
)
def test_rotation_rejects_real_protocol_controls_before_core(
    monkeypatch: pytest.MonkeyPatch,
    loadings: np.ndarray,
    call: Callable[[np.ndarray, object], object],
) -> None:
    """Real-valued controls reject arbitrary float protocols before native discovery."""

    _assert_rejected_without_callback(
        monkeypatch, _HostileReal, lambda value: call(loadings, value)
    )


@pytest.mark.parametrize(
    "call",
    (
        lambda loadings, value: rotation.rotate_factor_loadings(
            loadings, criterion="simplimax", simplimax_zeros=value
        ),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(
            loadings, "simplimax", simplimax_zeros=value
        ),
    ),
)
def test_rotation_rejects_optional_integer_protocol_controls_before_core(
    monkeypatch: pytest.MonkeyPatch,
    loadings: np.ndarray,
    call: Callable[[np.ndarray, object], object],
) -> None:
    """Optional integer controls reject arbitrary protocols before native discovery."""

    _assert_rejected_without_callback(
        monkeypatch, _HostileInteger, lambda value: call(loadings, value)
    )


@pytest.mark.parametrize("value", (True, False))
@pytest.mark.parametrize(
    "call",
    (
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, n_starts=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, seed=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, max_iter=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, tolerance=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, basin_tolerance=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, function_window=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, max_line_search=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, max_threads=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, criterion="cf", kappa=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, criterion="oblimin", gamma=value),
        lambda loadings, value: rotation.rotate_factor_loadings(loadings, criterion="geomin", delta=value),
        lambda loadings, value: rotation.rotate_factor_loadings(
            loadings, criterion="simplimax", simplimax_zeros=value
        ),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(loadings, "cf", kappa=value),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(loadings, "oblimin", gamma=value),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(loadings, "geomin", delta=value),
        lambda loadings, value: rotation.rotation_criterion_value_gradient(
            loadings, "simplimax", simplimax_zeros=value
        ),
    ),
)
def test_rotation_rejects_boolean_numeric_controls_before_core(
    monkeypatch: pytest.MonkeyPatch,
    loadings: np.ndarray,
    call: Callable[[np.ndarray, object], object],
    value: bool,
) -> None:
    """Boolean values never masquerade as numeric rotation controls."""

    core_calls = _deny_core_discovery(monkeypatch)
    with pytest.raises(ValueError):
        call(loadings, value)
    assert core_calls == []
