"""Callback-free trust-boundary regressions for rotation selection controls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fast_mlsirm.rotation_selection as selection


class _HostileString(str):
    """String subclass whose normalization callback must stay unreachable."""

    calls = 0

    def strip(self, chars: str | None = None) -> str:
        type(self).calls += 1
        raise AssertionError("caller string callback must not execute")


class _HostileInteger:
    """Integer-protocol object whose callbacks must stay unreachable."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller integer callback must not execute")

    def __index__(self) -> int:
        type(self).calls += 1
        raise AssertionError("caller index callback must not execute")


class _HostileReal:
    """Real-protocol object whose callback must stay unreachable."""

    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("caller float callback must not execute")


class _HostileBoolean:
    """Boolean-protocol object whose callback must stay unreachable."""

    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("caller bool callback must not execute")


@pytest.fixture
def loadings() -> np.ndarray:
    """Return a small valid loading matrix."""

    return np.asarray([[0.8, 0.2], [0.7, 0.1], [0.1, 0.8], [0.2, 0.7]])


def _deny_core_discovery(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Install a native-discovery probe that invalid controls must not reach."""

    calls: list[int] = []

    def _probe() -> object:
        calls.append(1)
        raise AssertionError("rotation core must not be discovered for invalid controls")

    monkeypatch.setattr(selection, "rotation_core", _probe)
    return calls


def _assert_rejected_without_callback(
    monkeypatch: pytest.MonkeyPatch,
    hostile_type: type,
    call: Callable[[object], object],
) -> None:
    """Assert package-owned rejection precedes both callback and Rust discovery."""

    hostile_type.calls = 0
    core_calls = _deny_core_discovery(monkeypatch)
    with pytest.raises(ValueError):
        call(hostile_type())
    assert hostile_type.calls == 0
    assert core_calls == []


def test_selection_rejects_policy_subclass_without_callbacks(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray
) -> None:
    """Policy normalization rejects string subclasses before ``strip`` or Rust."""

    _HostileString.calls = 0
    core_calls = _deny_core_discovery(monkeypatch)
    with pytest.raises(ValueError, match="policy must be a non-empty string"):
        selection.select_rotation_criterion(
            loadings,
            ("varimax", "geomin"),
            policy=_HostileString("fully_exploratory"),
        )
    assert _HostileString.calls == 0
    assert core_calls == []


def test_selection_rejects_candidate_subclass_without_callbacks(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray
) -> None:
    """Candidate names share the exact-string criterion boundary."""

    _HostileString.calls = 0
    core_calls = _deny_core_discovery(monkeypatch)
    with pytest.raises(ValueError, match="criterion must be a non-empty string"):
        selection.select_rotation_criterion(
            loadings,
            (_HostileString("varimax"), "geomin"),
        )
    assert _HostileString.calls == 0
    assert core_calls == []


def test_selection_rejects_boolean_protocol_normalize_before_core(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray
) -> None:
    """Normalize rejects arbitrary truth-value providers before native discovery."""

    _assert_rejected_without_callback(
        monkeypatch,
        _HostileBoolean,
        lambda value: selection.select_rotation_criterion(
            loadings, ("varimax", "geomin"), normalize=value
        ),
    )


@pytest.mark.parametrize(
    "keyword",
    ("n_starts", "seed", "max_iter", "function_window", "max_line_search", "max_threads"),
)
def test_selection_rejects_integer_protocol_controls_before_core(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray, keyword: str
) -> None:
    """Every integer selector control rejects caller conversion protocols."""

    _assert_rejected_without_callback(
        monkeypatch,
        _HostileInteger,
        lambda value: selection.select_rotation_criterion(
            loadings, ("varimax", "geomin"), **{keyword: value}
        ),
    )


@pytest.mark.parametrize("keyword", ("tolerance", "basin_tolerance"))
def test_selection_rejects_real_protocol_controls_before_core(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray, keyword: str
) -> None:
    """Every real selector control rejects caller float protocols."""

    _assert_rejected_without_callback(
        monkeypatch,
        _HostileReal,
        lambda value: selection.select_rotation_criterion(
            loadings, ("varimax", "geomin"), **{keyword: value}
        ),
    )


@pytest.mark.parametrize("keyword", ("kappa", "gamma", "delta"))
def test_selection_rejects_optional_real_protocol_controls_before_core(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray, keyword: str
) -> None:
    """Optional criterion reals reject caller float protocols before native discovery."""

    _assert_rejected_without_callback(
        monkeypatch,
        _HostileReal,
        lambda value: selection.select_rotation_criterion(
            loadings, ("cf", "geomin"), **{keyword: value}
        ),
    )


def test_selection_rejects_optional_integer_protocol_before_core(
    monkeypatch: pytest.MonkeyPatch, loadings: np.ndarray
) -> None:
    """Simplimax zero-count rejects caller integer protocols before native discovery."""

    _assert_rejected_without_callback(
        monkeypatch,
        _HostileInteger,
        lambda value: selection.select_rotation_criterion(
            loadings,
            ("simplimax", "geomin"),
            simplimax_zeros=value,
        ),
    )
