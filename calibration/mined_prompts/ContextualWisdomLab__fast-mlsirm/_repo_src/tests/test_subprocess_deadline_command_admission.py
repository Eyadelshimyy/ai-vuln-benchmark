"""Callback-safety regressions for bounded subprocess command admission."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE = Path(__file__).parents[1] / "scripts" / "_subprocess_deadlines.py"
_SPEC = importlib.util.spec_from_file_location("subprocess_deadlines_command_admission", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
deadlines = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deadlines
_SPEC.loader.exec_module(deadlines)


class _HostileCommandList(list[str]):
    """List subclass that records unsafe pre-admission iteration."""

    callbacks = 0

    def __iter__(self):  # type: ignore[override]
        """Fail if validation iterates a caller-defined command container."""
        type(self).callbacks += 1
        raise AssertionError("hostile command-list iteration executed")


class _HostileCommandText(str):
    """Text subclass that records unsafe truthiness evaluation."""

    callbacks = 0

    def __len__(self) -> int:
        """Fail if validation evaluates caller-defined token truthiness."""
        type(self).callbacks += 1
        raise AssertionError("hostile command-token length executed")


def test_command_container_subclass_is_rejected_before_iteration() -> None:
    """Only inert built-in command containers reach argument materialization."""
    _HostileCommandList.callbacks = 0
    with pytest.raises(ValueError, match="command"):
        deadlines._validate_command(_HostileCommandList(["cargo", "metadata"]))
    assert _HostileCommandList.callbacks == 0


def test_command_token_subclass_is_rejected_before_truthiness() -> None:
    """Only inert built-in text tokens reach non-empty validation."""
    _HostileCommandText.callbacks = 0
    with pytest.raises(ValueError, match="command"):
        deadlines._validate_command(["cargo", _HostileCommandText("metadata")])
    assert _HostileCommandText.callbacks == 0


def test_exact_builtin_command_vectors_remain_supported() -> None:
    """Ordinary list and tuple argument vectors preserve existing behavior."""
    assert deadlines._validate_command(["cargo", "metadata"]) == ["cargo", "metadata"]
    assert deadlines._validate_command(("cargo", "metadata")) == ["cargo", "metadata"]
