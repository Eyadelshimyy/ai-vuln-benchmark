"""Security regressions for Fleiss-kappa public control validation."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import fast_mlsirm
from fast_mlsirm import validation


class _HostileInt(int):
    """Integer subclass whose conversion callback must remain inert."""

    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        return 2


class _IndexOnlyProvider:
    """Integer-protocol provider that must not be executed."""

    calls = 0

    def __index__(self) -> int:
        type(self).calls += 1
        return 2


class _TruthinessProvider:
    """Boolean-protocol provider that must not be executed."""

    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        return False


class _RatingsArrayProvider:
    """Array-protocol provider that records forbidden early materialization."""

    def __init__(self) -> None:
        self.calls = 0

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        self.calls += 1
        raise AssertionError("ratings materialized before Fleiss control validation")


def _fake_core(calls: list[tuple[int, type[int], bool, type[bool]]]) -> SimpleNamespace:
    """Return a deterministic Rust-boundary stand-in for marshalling tests."""

    def fleiss_kappa(ratings, ns, nr, k, exact):
        del ratings, ns, nr
        calls.append((k, type(k), exact, type(exact)))
        return {
            "kappa": 0.5,
            "subjects_used": 2,
            "z": 0.0,
            "p_value": 1.0,
            "category_kappa": [0.5, 0.5],
            "category_z": [0.0, 0.0],
            "category_p": [1.0, 1.0],
        }

    return SimpleNamespace(fleiss_kappa=fleiss_kappa)


def _ratings() -> np.ndarray:
    """Return a small valid two-category, two-rater fixture."""
    return np.array([[0, 1], [1, 0]], dtype=np.int64)


@pytest.mark.parametrize("k", [_HostileInt(2), _IndexOnlyProvider(), True, np.array(2)])
def test_fleiss_rejects_executable_or_untrusted_k_without_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    k: object,
) -> None:
    """Rejected category controls cannot run callbacks or native dispatch."""
    _HostileInt.calls = 0
    _IndexOnlyProvider.calls = 0
    rust_calls: list[tuple[int, type[int], bool, type[bool]]] = []
    fake = _fake_core(rust_calls)
    monkeypatch.setattr(fast_mlsirm, "_core", fake, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake)

    with pytest.raises(ValueError, match="k must be an integer"):
        validation.fleiss_kappa(_ratings(), k=k)

    assert _HostileInt.calls == 0
    assert _IndexOnlyProvider.calls == 0
    assert rust_calls == []


def test_fleiss_rejects_truthiness_provider_without_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-mode control cannot invoke caller-defined ``__bool__``."""
    _TruthinessProvider.calls = 0
    rust_calls: list[tuple[int, type[int], bool, type[bool]]] = []
    fake = _fake_core(rust_calls)
    monkeypatch.setattr(fast_mlsirm, "_core", fake, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake)

    with pytest.raises(ValueError, match="exact must be a boolean"):
        validation.fleiss_kappa(_ratings(), k=2, exact=_TruthinessProvider())

    assert _TruthinessProvider.calls == 0
    assert rust_calls == []


@pytest.mark.parametrize("k", [1, 10_001, _IndexOnlyProvider()])
def test_fleiss_rejects_invalid_k_before_core_import(
    monkeypatch: pytest.MonkeyPatch,
    k: object,
) -> None:
    """Invalid category controls fail before data materialization or core discovery."""
    real_import = builtins.__import__
    core_import_calls = 0
    _IndexOnlyProvider.calls = 0
    ratings = _RatingsArrayProvider()

    def guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        nonlocal core_import_calls
        if "_core" in fromlist:
            core_import_calls += 1
            raise AssertionError("compiled core discovered before Fleiss control validation")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError):
        validation.fleiss_kappa(ratings, k=k)

    assert ratings.calls == 0
    assert core_import_calls == 0
    assert _IndexOnlyProvider.calls == 0


def test_fleiss_rejects_invalid_exact_before_core_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Type-invalid exact controls fail before data materialization or core discovery."""
    real_import = builtins.__import__
    core_import_calls = 0
    ratings = _RatingsArrayProvider()

    def guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        nonlocal core_import_calls
        if "_core" in fromlist:
            core_import_calls += 1
            raise AssertionError("compiled core discovered before exact validation")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="exact must be a boolean"):
        validation.fleiss_kappa(ratings, k=2, exact=1)

    assert ratings.calls == 0
    assert core_import_calls == 0


@pytest.mark.parametrize("k", [2, np.int8(2), np.uint16(2), np.int64(2), np.uint64(2)])
@pytest.mark.parametrize("exact", [False, True, np.bool_(False), np.bool_(True)])
def test_fleiss_normalizes_trusted_controls_to_builtins(
    monkeypatch: pytest.MonkeyPatch,
    k: int | np.integer,
    exact: bool | np.bool_,
) -> None:
    """Supported Python/NumPy controls marshal as exact built-in scalars."""
    rust_calls: list[tuple[int, type[int], bool, type[bool]]] = []
    fake = _fake_core(rust_calls)
    monkeypatch.setattr(fast_mlsirm, "_core", fake, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake)

    validation.fleiss_kappa(_ratings(), k=k, exact=exact)

    assert rust_calls == [(2, int, bool(exact), bool)]


def test_fleiss_rejects_inferred_category_count_above_rust_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inferred categories cannot exceed the native dense-allocation contract."""
    rust_calls: list[tuple[int, type[int], bool, type[bool]]] = []
    fake = _fake_core(rust_calls)
    monkeypatch.setattr(fast_mlsirm, "_core", fake, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake)
    ratings = np.array([[10_000, 0], [1, 0]], dtype=np.int64)

    with pytest.raises(ValueError, match="k must be <= 10000"):
        validation.fleiss_kappa(ratings)

    assert rust_calls == []
