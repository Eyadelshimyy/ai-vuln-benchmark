"""Fail-closed contracts for an installed but unloadable Rust extension."""

from __future__ import annotations

import importlib.machinery
from collections.abc import Iterator

import pytest

import fast_mlsirm.backend as backend


@pytest.fixture(params=[ImportError, OSError])
def _installed_core_that_fails_to_import(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[ImportError | OSError]:
    """Make discovery succeed while the compiled extension import fails."""
    spec = importlib.machinery.ModuleSpec(backend.CORE_MODULE, loader=None)
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda _name: spec)
    exception_type = request.param
    failure = exception_type("dlopen failed: incompatible compiled extension")

    def fail_import(_name: str):
        """Raise the controlled native-loader failure for the fixture."""
        raise failure

    monkeypatch.setattr(backend.importlib, "import_module", fail_import)
    yield failure


@pytest.mark.parametrize("requested", ["auto", "rust"])
def test_resolve_backend_normalizes_compiled_core_import_failure(
    _installed_core_that_fails_to_import: ImportError | OSError,
    requested: str,
) -> None:
    """Backend resolution must fail closed with a package-owned RuntimeError."""
    failure = _installed_core_that_fails_to_import

    with pytest.raises(
        RuntimeError,
        match="compiled Rust core is present but could not be imported",
    ) as caught:
        backend.resolve_backend(requested)

    assert caught.value.__cause__ is failure


def test_load_rust_core_normalizes_compiled_core_import_failure(
    _installed_core_that_fails_to_import: ImportError | OSError,
) -> None:
    """Direct core loading must expose the same stable fail-closed contract."""
    failure = _installed_core_that_fails_to_import

    with pytest.raises(
        RuntimeError,
        match="compiled Rust core is present but could not be imported",
    ) as caught:
        backend.load_rust_core()

    assert caught.value.__cause__ is failure
