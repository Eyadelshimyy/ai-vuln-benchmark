"""Import-boundary regressions for PR queue bounded JSON parsing."""

from __future__ import annotations

import builtins
from pathlib import Path
import runpy

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_pr_queue_snapshot.py"


def _missing_module(name: str) -> ModuleNotFoundError:
    """Return a ModuleNotFoundError carrying the import name Python exposes."""
    error = ModuleNotFoundError(f"No module named {name!r}")
    error.name = name
    return error


def test_capture_import_fails_closed_when_bounded_parser_is_unavailable(monkeypatch):
    """Isolation must not replace repository JSON bounds with plain json.loads."""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "scripts._bounded_json":
            raise _missing_module("scripts._bounded_json")
        if name == "_bounded_json":
            raise _missing_module("_bounded_json")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(RuntimeError, match="bounded JSON parser is unavailable"):
        runpy.run_path(str(_SCRIPT), run_name="isolated_pr_queue_capture")


def test_capture_import_does_not_mask_helper_internal_module_failures(monkeypatch):
    """A broken real helper is not misclassified as an alternate-layout import."""
    real_import = builtins.__import__
    sibling_attempted = False

    def guarded_import(name, *args, **kwargs):
        nonlocal sibling_attempted
        if name == "scripts._bounded_json":
            raise _missing_module("bounded_json_internal_dependency")
        if name == "_bounded_json":
            sibling_attempted = True
            raise AssertionError("sibling fallback must not mask helper-internal failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        runpy.run_path(str(_SCRIPT), run_name="broken_bounded_json_dependency")

    assert exc_info.value.name == "bounded_json_internal_dependency"
    assert sibling_attempted is False
