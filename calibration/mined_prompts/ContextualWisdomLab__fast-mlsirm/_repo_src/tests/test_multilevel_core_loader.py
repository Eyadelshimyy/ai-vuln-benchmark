"""Tests for the secondary multilevel Rust extension loader.

The happy path (successful load, then cache hit) is already exercised
indirectly by ``tests/test_multilevel_estimation.py``. This file covers the
loader's own fail-closed edge cases via monkeypatching, since nothing else
in the suite exercises a missing extension path, a failed spec, a failed
``exec_module``, or the double-checked-locking cache-hit branch.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from fast_mlsirm import _multilevel_core_loader as loader


@pytest.fixture(autouse=True)
def _clear_cached_module():
    """Ensure the secondary module cache does not leak between tests."""
    sys.modules.pop(loader._MODULE_NAME, None)
    yield
    sys.modules.pop(loader._MODULE_NAME, None)


def test_multilevel_core_loads_and_caches_the_real_extension() -> None:
    """The happy path returns the same cached module on a second call."""
    first = loader.multilevel_core()
    assert hasattr(first, "weighted_contextual_effect")
    assert hasattr(first, "fit_hierarchical_ctar_rasch")
    assert hasattr(first, "simulate_hierarchical_ctar_rasch")
    assert hasattr(first, "fit_longitudinal_state")
    second = loader.multilevel_core()
    assert second is first


def test_multilevel_core_returns_the_cache_hit_without_reloading(monkeypatch) -> None:
    """A module already present in sys.modules is returned without re-init."""
    sentinel = ModuleType(loader._MODULE_NAME)
    sys.modules[loader._MODULE_NAME] = sentinel

    def _fail_if_called() -> ModuleType:
        raise AssertionError("_initialize_module must not run on a cache hit")

    monkeypatch.setattr(loader, "_initialize_module", _fail_if_called)

    assert loader.multilevel_core() is sentinel


def test_multilevel_core_double_checked_lock_returns_the_racing_winner(
    monkeypatch,
) -> None:
    """A module cached between the outer check and acquiring the lock is used.

    Simulates the race the double-checked-locking pattern in
    ``multilevel_core`` guards against: another thread finished
    ``_initialize_module`` after this call's first ``sys.modules`` check but
    before it acquired ``_LOAD_LOCK``.
    """
    sentinel = ModuleType(loader._MODULE_NAME)
    real_lock = loader._LOAD_LOCK

    class _RacingLock:
        def __enter__(self):
            real_lock.acquire()
            sys.modules[loader._MODULE_NAME] = sentinel
            return self

        def __exit__(self, *exc_info):
            real_lock.release()
            return False

    monkeypatch.setattr(loader, "_LOAD_LOCK", _RacingLock())

    def _fail_if_called() -> ModuleType:
        raise AssertionError("_initialize_module must not run once the race winner is cached")

    monkeypatch.setattr(loader, "_initialize_module", _fail_if_called)

    assert loader.multilevel_core() is sentinel


def test_initialize_module_rejects_a_core_extension_without_a_file_path(
    monkeypatch,
) -> None:
    """A ``_core`` module with no ``__file__`` cannot be used as the binary source."""
    fake_core = ModuleType("fast_mlsirm._core")
    monkeypatch.setattr("fast_mlsirm._core", fake_core, raising=False)
    monkeypatch.setitem(sys.modules, "fast_mlsirm._core", fake_core)

    with pytest.raises(ImportError, match="does not expose an extension path"):
        loader._initialize_module()


def test_initialize_module_rejects_a_failed_spec(monkeypatch) -> None:
    """A loader that cannot produce a spec raises rather than returning None."""
    monkeypatch.setattr(loader, "spec_from_loader", lambda *args, **kwargs: None)

    with pytest.raises(ImportError, match="could not create"):
        loader._initialize_module()


def test_initialize_module_evicts_the_cache_entry_when_exec_module_fails(
    monkeypatch,
) -> None:
    """A failed exec_module removes the half-initialized module from sys.modules."""

    class _ExplodingLoader:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create_module(self, spec):
            """Defer to the default module creation, matching ExtensionFileLoader."""
            return None

        def exec_module(self, module: ModuleType) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(loader, "ExtensionFileLoader", _ExplodingLoader)

    with pytest.raises(RuntimeError, match="boom"):
        loader._initialize_module()

    assert loader._MODULE_NAME not in sys.modules
