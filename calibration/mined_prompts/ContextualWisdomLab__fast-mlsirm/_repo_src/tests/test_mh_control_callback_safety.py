"""Trust-boundary regressions for public Mantel-Haenszel DIF controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.dif import mantel_haenszel_dif


class _DataSentinel:
    """Fail if rejected controls permit response/group materialization."""

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Raise on any attempt to materialize caller-owned data."""
        raise AssertionError("data must not be materialized for invalid controls")


def _unexpected_core_discovery():
    """Fail if rejected controls permit compiled-core discovery."""
    raise AssertionError("compiled core must not be discovered for invalid controls")


class _HostileFloat(float):
    """Float subclass whose conversion/comparison callbacks must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self):
        type(self).calls += 1
        return float.__float__(self)

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        return float.__lt__(self, other)


class _HostileFloatProvider:
    """Arbitrary float protocol provider whose callback must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        return 0.05


class _HostileBool(int):
    """Integer disguised as a matching-score flag; ``bool()`` must stay inert."""

    calls = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the callback counter."""
        cls.calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        return int.__bool__(self)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("fdr_q", _HostileFloat(0.05), "fdr_q must be a real number"),
        ("fdr_q", _HostileFloatProvider(), "fdr_q must be a real number"),
        ("exclude_studied_item", _HostileBool(1), "exclude_studied_item must be an exact bool"),
    ),
)
def test_rejects_executable_controls_before_callbacks_data_or_core(
    monkeypatch, field, value, message
):
    """Executable control providers fail before callbacks, data, and Rust discovery."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)
    type(value).reset()

    with pytest.raises(ValueError, match=message):
        mantel_haenszel_dif(_DataSentinel(), _DataSentinel(), **{field: value})

    assert type(value).calls == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"fdr_q": 0.0}, r"fdr_q must be finite and in \(0, 1\]"),
        ({"fdr_q": 1.5}, r"fdr_q must be finite and in \(0, 1\]"),
        ({"fdr_q": float("nan")}, r"fdr_q must be finite and in \(0, 1\]"),
        ({"fdr_q": float("inf")}, r"fdr_q must be finite and in \(0, 1\]"),
        ({"fdr_q": True}, "fdr_q must be a real number"),
        ({"exclude_studied_item": 1}, "exclude_studied_item must be an exact bool"),
        ({"exclude_studied_item": np.bool_(True)}, "exclude_studied_item must be an exact bool"),
    ),
)
def test_invalid_exact_controls_fail_before_data_and_core(monkeypatch, kwargs, message):
    """Type-correct invalid controls fail before any data/native side effect."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match=message):
        mantel_haenszel_dif(_DataSentinel(), _DataSentinel(), **kwargs)


def test_huge_builtin_integer_fdr_q_fails_as_value_error(monkeypatch):
    """Overflowing exact integers become package ValueError before data or Rust."""
    monkeypatch.setattr(fitstats, "_core_module", _unexpected_core_discovery)

    with pytest.raises(ValueError, match="fdr_q must be a real number"):
        mantel_haenszel_dif(_DataSentinel(), _DataSentinel(), fdr_q=10**10000)


class _FakeCore:
    """Capture the trusted PyO3 payload for one valid Mantel-Haenszel call."""

    def __init__(self) -> None:
        """Initialize an empty call ledger."""
        self.calls: list[tuple[Any, ...]] = []

    def mantel_haenszel_dif(self, *args: Any) -> dict[str, Any]:
        """Record dispatch and return a structurally valid Rust-like result."""
        self.calls.append(args)
        return {
            "item": [0, 1],
            "alpha_mh": [1.0, 1.0],
            "chi2_mh": [0.0, 0.0],
            "p_value": [1.0, 1.0],
            "mh_d_dif": [0.0, 0.0],
            "se_d_dif": [1.0, 1.0],
            "std_p_dif": [0.0, 0.0],
            "ets_class": ["A", "A"],
            "flagged_bh": [False, False],
        }


def _responses() -> np.ndarray:
    """Return a small valid dichotomous response matrix."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.int64)


def _group() -> np.ndarray:
    """Return a valid reference/focal group vector."""
    return np.array([0, 1, 0, 1], dtype=np.int64)


def test_genuine_numpy_fdr_q_dispatches_as_exact_builtins(monkeypatch):
    """Supported NumPy scalars normalize once before the PyO3 call."""
    core = _FakeCore()
    monkeypatch.setattr(fitstats, "_core_module", lambda: core)

    result = mantel_haenszel_dif(
        _responses(),
        _group(),
        exclude_studied_item=True,
        fdr_q=np.float32(0.05),
    )

    assert list(result["ets_class"]) == ["A", "A"]
    assert len(core.calls) == 1
    (_y, _g, n_persons, n_items, exclude_studied_item, fdr_q) = core.calls[0]
    assert type(n_persons) is int
    assert type(n_items) is int
    assert type(exclude_studied_item) is bool
    assert type(fdr_q) is float
    assert (n_persons, n_items, exclude_studied_item) == (4, 2, True)
    assert fdr_q == pytest.approx(0.05, rel=0, abs=1e-6)
