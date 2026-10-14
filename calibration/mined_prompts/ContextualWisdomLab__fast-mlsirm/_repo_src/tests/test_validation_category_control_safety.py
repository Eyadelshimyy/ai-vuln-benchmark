"""Security regressions for judge category-count validation."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import validation


class _HostileInt(int):
    """Integer subclass whose conversion callback records execution."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return 2


class _HostileNumpyInt(np.int64):
    """NumPy integer subclass whose conversion callback records execution."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return 2


class _HostileScalarMeta(type):
    """Metaclass proving scalar-type admission cannot hash caller types."""

    calls = 0

    def __hash__(cls) -> int:
        type(cls).calls += 1
        return type.__hash__(np.int64)

    def __eq__(cls, other: object) -> bool:
        type(cls).calls += 1
        return type.__eq__(cls, other)


class _MetaclassHostileNumpyInt(np.int64, metaclass=_HostileScalarMeta):
    """NumPy scalar subclass with caller-controlled type hash/equality hooks."""


class _IntegerProtocolProvider:
    """Arbitrary integer protocol provider that must never be invoked."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return 2


class _IndexOnlyProvider:
    """``__index__``-only provider that must never be invoked."""

    calls = 0

    def __index__(self) -> int:
        type(self).calls += 1
        return 2


class _ComparisonHookProvider:
    """Non-integer control whose repr/eq/hash/order hooks must stay inert."""

    calls = 0

    def __repr__(self) -> str:
        type(self).calls += 1
        return "2"

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        return False

    def __hash__(self) -> int:
        type(self).calls += 1
        return 2

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        return False

    def __gt__(self, other: object) -> bool:
        type(self).calls += 1
        return False


def _fake_core(calls: list[tuple[int, type[int]]]) -> SimpleNamespace:
    """Return a deterministic Rust-boundary stand-in for marshalling tests."""

    def validate_scoring(judge, human, k, **kwargs):
        del judge, human, kwargs
        calls.append((k, type(k)))
        return {
            "gates": [],
            "exact_agreement": 1.0,
            "adjacent_agreement": 1.0,
            "pass": True,
        }

    return SimpleNamespace(validate_scoring=validate_scoring)


@pytest.mark.parametrize(
    "control_type",
    [
        _HostileInt,
        _HostileNumpyInt,
        _IntegerProtocolProvider,
        _IndexOnlyProvider,
        _ComparisonHookProvider,
    ],
)
def test_validate_judge_rejects_executable_integer_controls_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    control_type: type,
) -> None:
    """Rejected category controls cannot run conversion code or Rust dispatch."""
    control_type.calls = 0
    rust_calls: list[tuple[int, type[int]]] = []
    monkeypatch.setattr(fast_mlsirm, "_core", _fake_core(rust_calls), raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", _fake_core(rust_calls))

    value = (
        control_type(2)
        if control_type in {_HostileInt, _HostileNumpyInt}
        else control_type()
    )
    with pytest.raises(ValueError, match="must be an integer"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=value)

    assert control_type.calls == 0
    assert rust_calls == []


@pytest.mark.parametrize("k", [True, False, np.bool_(True), np.array(2)])
def test_validate_judge_rejects_bool_and_zero_dim_array_controls(
    monkeypatch: pytest.MonkeyPatch,
    k: object,
) -> None:
    """Booleans and 0-d arrays cannot satisfy the exact integer-scalar contract."""
    rust_calls: list[tuple[int, type[int]]] = []
    monkeypatch.setattr(fast_mlsirm, "_core", _fake_core(rust_calls), raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", _fake_core(rust_calls))

    with pytest.raises(ValueError, match="must be an integer"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=k)

    assert rust_calls == []


def test_validate_judge_rejects_hostile_scalar_metaclass_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalar-type admission cannot dispatch caller metaclass hash/equality hooks."""
    _HostileScalarMeta.calls = 0
    rust_calls: list[tuple[int, type[int]]] = []
    fake = _fake_core(rust_calls)
    monkeypatch.setattr(fast_mlsirm, "_core", fake, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake)

    with pytest.raises(ValueError):
        validation.validate_judge(
            np.array([0, 1]),
            np.array([0, 1]),
            k=_MetaclassHostileNumpyInt(2),
        )

    assert _HostileScalarMeta.calls == 0
    assert rust_calls == []


def test_validate_judge_rejects_invalid_k_before_core_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid exact category count fails before compiled-core discovery."""
    real_import = builtins.__import__
    core_import_calls = 0

    def guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        nonlocal core_import_calls
        if "_core" in fromlist:
            core_import_calls += 1
            raise AssertionError("compiled core discovered before k validation")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="must be >= 2"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=1)

    assert core_import_calls == 0


@pytest.mark.parametrize("k", [True, _IndexOnlyProvider(), np.array(2)])
def test_validate_judge_rejects_type_invalid_k_before_core_import(
    monkeypatch: pytest.MonkeyPatch,
    k: object,
) -> None:
    """Type-invalid category counts fail before compiled-core discovery."""
    real_import = builtins.__import__
    core_import_calls = 0
    _IndexOnlyProvider.calls = 0

    def guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        nonlocal core_import_calls
        if "_core" in fromlist:
            core_import_calls += 1
            raise AssertionError("compiled core discovered before k validation")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="must be an integer"):
        validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=k)

    assert core_import_calls == 0
    if isinstance(k, _IndexOnlyProvider):
        assert _IndexOnlyProvider.calls == 0


@pytest.mark.parametrize(
    "k",
    [2, np.int8(2), np.uint16(2), np.int32(2), np.uint64(2)],
)
def test_validate_judge_normalizes_genuine_numpy_category_counts(
    monkeypatch: pytest.MonkeyPatch,
    k: int | np.integer,
) -> None:
    """Supported exact integers marshal as exact Python ints."""
    rust_calls: list[tuple[int, type[int]]] = []
    fake = _fake_core(rust_calls)
    monkeypatch.setattr(fast_mlsirm, "_core", fake, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake)

    verdict = validation.validate_judge(np.array([0, 1]), np.array([0, 1]), k=k)

    assert verdict.passed is True
    assert rust_calls == [(2, int)]


def test_validate_judge_rejects_category_count_above_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing 2..=1000 domain still rejects an exact in-type overflow."""
    rust_calls: list[tuple[int, type[int]]] = []
    fake = _fake_core(rust_calls)
    monkeypatch.setattr(fast_mlsirm, "_core", fake, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake)

    with pytest.raises(ValueError, match="must be <="):
        validation.validate_judge(
            np.array([0, 1]),
            np.array([0, 1]),
            k=validation.MAX_JUDGE_CATEGORIES + 1,
        )

    assert rust_calls == []
