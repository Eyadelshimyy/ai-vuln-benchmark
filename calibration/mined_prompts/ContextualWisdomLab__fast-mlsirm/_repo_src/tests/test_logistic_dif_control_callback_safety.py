"""Trust-boundary regressions for observed-score logistic DIF controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm._dif_control_safety as dif_control_safety
import fast_mlsirm.dif as dif
import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.dif import logistic_dif, logistic_dif_purified, mantel_haenszel_dif_purified


class _DataSentinel:
    """Fail if rejected controls permit caller data materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise on any attempt to materialize rejected caller data."""
        raise AssertionError("data must not be materialized for invalid controls")


def _unexpected_core_discovery():
    """Fail if rejected controls permit compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


class _HostileFloat(float):
    """Float subclass whose conversion hooks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset callback accounting."""
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        return float.__float__(self)


class _HostileFloatProvider:
    """Arbitrary float protocol provider whose callback must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset callback accounting."""
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        return 0.05


class _HostileBool(int):
    """Integer disguised as a boolean control."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset callback accounting."""
        cls.calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        return True


class _HostileInt(int):
    """Integer subclass whose conversion hooks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset callback accounting."""
        cls.calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return int.__int__(self)

    def __index__(self) -> int:
        type(self).calls += 1
        return int.__index__(self)


@pytest.mark.parametrize(
    ("kwargs", "value", "message"),
    (
        ({"fdr_q": _HostileFloat(0.05)}, _HostileFloat, "fdr_q must be a real number"),
        ({"fdr_q": _HostileFloatProvider()}, _HostileFloatProvider, "fdr_q must be a real number"),
        ({"exclude_studied_item": _HostileBool(1)}, _HostileBool, "exclude_studied_item must be a bool"),
        ({"max_iter": _HostileInt(50)}, _HostileInt, "max_iter must be an integer"),
    ),
)
def test_logistic_rejects_executable_controls_before_callbacks_data_or_core(
    monkeypatch, kwargs, value, message
):
    """Executable logistic controls fail before callbacks, data, and Rust discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    value.reset()

    with pytest.raises(ValueError, match=message):
        logistic_dif(_DataSentinel(), _DataSentinel(), **kwargs)

    assert value.calls == 0


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    (
        (logistic_dif, {"fdr_q": 0.0}, r"fdr_q must be finite and in \(0, 1\]"),
        (logistic_dif, {"fdr_q": True}, "fdr_q must be a real number"),
        (logistic_dif, {"max_iter": -1}, "max_iter must be >= 1"),
        (logistic_dif, {"max_iter": 0}, "max_iter must be >= 1"),
        (logistic_dif, {"max_iter": True}, "max_iter must be an integer"),
        (mantel_haenszel_dif_purified, {"max_rounds": 0}, "max_rounds must be >= 1"),
        (mantel_haenszel_dif_purified, {"min_anchor_items": -1}, "min_anchor_items must be >= 0"),
        (logistic_dif_purified, {"max_iter": -1}, "max_iter must be >= 1"),
        (logistic_dif_purified, {"max_iter": 0}, "max_iter must be >= 1"),
        (logistic_dif_purified, {"max_rounds": 0}, "max_rounds must be >= 1"),
        (logistic_dif_purified, {"min_anchor_items": -1}, "min_anchor_items must be >= 0"),
    ),
)
def test_invalid_exact_controls_fail_before_data_and_core(monkeypatch, function, kwargs, message):
    """Exact but invalid controls fail before caller data and native discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=message):
        function(_DataSentinel(), _DataSentinel(), **kwargs)


@pytest.mark.parametrize(
    ("function", "kwargs", "control_type", "message"),
    (
        (mantel_haenszel_dif_purified, {"max_rounds": _HostileInt(3)}, _HostileInt, "max_rounds must be an integer"),
        (mantel_haenszel_dif_purified, {"min_anchor_items": _HostileInt(4)}, _HostileInt, "min_anchor_items must be an integer"),
        (logistic_dif_purified, {"max_iter": _HostileInt(50)}, _HostileInt, "max_iter must be an integer"),
        (logistic_dif_purified, {"max_rounds": _HostileInt(3)}, _HostileInt, "max_rounds must be an integer"),
        (logistic_dif_purified, {"min_anchor_items": _HostileInt(4)}, _HostileInt, "min_anchor_items must be an integer"),
    ),
)
def test_purified_integer_controls_reject_subclasses_without_callbacks(
    monkeypatch, function, kwargs, control_type, message
):
    """Purification integer controls reject executable subclasses at the first boundary."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    control_type.reset()

    with pytest.raises(ValueError, match=message):
        function(_DataSentinel(), _DataSentinel(), **kwargs)

    assert control_type.calls == 0


class _FakeCore:
    """Capture one valid native logistic dispatch."""

    def __init__(self) -> None:
        """Initialize call accounting."""
        self.calls: list[tuple[Any, ...]] = []

    def logistic_dif(self, *args: Any) -> dict[str, Any]:
        """Return a structurally valid Rust-like result."""
        self.calls.append(args)
        return {
            "item": [0, 1],
            "chi2_uniform": [0.0, 0.0],
            "p_uniform": [1.0, 1.0],
            "chi2_nonuniform": [0.0, 0.0],
            "p_nonuniform": [1.0, 1.0],
            "chi2_total": [0.0, 0.0],
            "p_total": [1.0, 1.0],
            "delta_r2": [0.0, 0.0],
            "delta_r2_uniform": [0.0, 0.0],
            "jg_class": ["A", "A"],
            "flagged_bh": [False, False],
            "converged": [True, True],
        }


def test_genuine_numpy_controls_dispatch_as_exact_builtins(monkeypatch):
    """Supported NumPy scalar controls normalize once before the PyO3 boundary."""
    core = _FakeCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)
    responses = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int64)
    group = np.array([0, 1, 0, 1], dtype=np.int64)

    result = logistic_dif(
        responses,
        group,
        exclude_studied_item=np.bool_(True),
        fdr_q=np.float32(0.05),
        max_iter=np.int64(25),
    )

    assert list(result["jg_class"]) == ["A", "A"]
    assert len(core.calls) == 1
    (_y, _g, n_persons, n_items, exclude_studied_item, fdr_q, max_iter) = core.calls[0]
    assert type(n_persons) is int
    assert type(n_items) is int
    assert type(exclude_studied_item) is bool
    assert type(fdr_q) is float
    assert type(max_iter) is int
    assert (n_persons, n_items, exclude_studied_item, max_iter) == (4, 2, True, 25)
    assert fdr_q == pytest.approx(0.05, rel=0, abs=1e-6)


def test_dif_installer_is_idempotent():
    """Repeated package installation must not stack another validation layer."""
    first = (
        dif.logistic_dif,
        dif.mantel_haenszel_dif_purified,
        dif.logistic_dif_purified,
    )

    dif_control_safety.install(dif)

    assert (
        dif.logistic_dif,
        dif.mantel_haenszel_dif_purified,
        dif.logistic_dif_purified,
    ) == first
