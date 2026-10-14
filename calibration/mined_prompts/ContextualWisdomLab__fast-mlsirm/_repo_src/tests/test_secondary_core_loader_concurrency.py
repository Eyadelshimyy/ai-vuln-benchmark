"""Concurrency regressions for secondary native-extension loaders."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
import sys
import threading

import pytest

import fast_mlsirm
import fast_mlsirm._ata_core_loader as ata_loader
import fast_mlsirm._bifactor_core_loader as bifactor_loader
import fast_mlsirm._multilevel_core_loader as multilevel_loader
import fast_mlsirm._rating_range_core_loader as rating_range_loader
import fast_mlsirm._regression_core_loader as regression_loader
import fast_mlsirm._rotation_core_loader as rotation_loader


@pytest.mark.parametrize(
    ("loader", "function_name"),
    [
        (ata_loader, "ata_core"),
        (bifactor_loader, "bifactor_core"),
        (multilevel_loader, "multilevel_core"),
        (rating_range_loader, "rating_range_core"),
        (regression_loader, "regression_core"),
        (rotation_loader, "rotation_core"),
    ],
)
def test_secondary_loader_cache_does_not_bypass_initialization_lock(
    monkeypatch,
    tmp_path,
    loader,
    function_name: str,
) -> None:
    """A published in-progress module must not escape through the cache fast path."""
    target = loader._MODULE_NAME
    previous = sys.modules.pop(target, None)

    fake_core = ModuleType("fast_mlsirm._core")
    fake_core.__file__ = str(tmp_path / "fake-fast-mlsirm-core.so")
    monkeypatch.setattr(fast_mlsirm, "_core", fake_core)

    exec_started = threading.Event()
    release_exec = threading.Event()
    exec_calls = 0

    class FakeExtensionLoader:
        """Expose the publication window without loading a native library."""

        def __init__(self, name: str, path: str) -> None:
            self.name = name
            self.path = path

        def exec_module(self, module: ModuleType) -> None:
            nonlocal exec_calls
            exec_calls += 1
            module.initialized = False
            exec_started.set()
            if not release_exec.wait(timeout=2.0):
                raise AssertionError("test did not release simulated extension initialization")
            module.initialized = True

    monkeypatch.setattr(loader, "ExtensionFileLoader", FakeExtensionLoader)
    monkeypatch.setattr(loader, "spec_from_loader", lambda _name, _loader: object())
    monkeypatch.setattr(loader, "module_from_spec", lambda _spec: ModuleType(target))
    load = getattr(loader, function_name)
    second_started = threading.Event()

    def load_second() -> ModuleType:
        """Signal that the second worker is active before entering the loader."""
        second_started.set()
        return load()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(load)
            assert exec_started.wait(timeout=1.0)
            second = pool.submit(load_second)
            assert second_started.wait(timeout=1.0)
            assert not second.done(), (
                f"{function_name} returned the sys.modules entry while the first "
                "caller was still initializing it"
            )
            release_exec.set()
            first_module = first.result(timeout=1.0)
            second_module = second.result(timeout=1.0)
    finally:
        release_exec.set()
        sys.modules.pop(target, None)
        if previous is not None:
            sys.modules[target] = previous

    assert first_module is second_module
    assert first_module.initialized is True
    assert exec_calls == 1
