"""Regression for descendants surviving a timed-out POSIX process-group leader."""

from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
from pathlib import Path

import pytest


_MODULE = Path(__file__).parents[1] / "scripts" / "_subprocess_deadlines.py"
_SPEC = importlib.util.spec_from_file_location("subprocess_deadlines_group_leak", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
deadlines = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deadlines
_SPEC.loader.exec_module(deadlines)


class _LeaderExitsProcess:
    """Timed-out leader that exits while a same-group descendant remains alive."""

    def __init__(self) -> None:
        self.pid = 6262
        self.returncode = 0
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout=None):
        """Timeout once, then model the leader being reapable after group cleanup."""
        self.communicate_calls.append(timeout)
        if len(self.communicate_calls) == 1:
            raise subprocess.TimeoutExpired(["opaque-child"], timeout)
        return (None, None)


def test_posix_timeout_kills_surviving_group_after_leader_exits(monkeypatch) -> None:
    """Cleanup verifies the group after grace and kills a surviving descendant."""
    fake = _LeaderExitsProcess()
    signals: list[tuple[int, int]] = []
    sleeps: list[float] = []

    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(
        deadlines.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )
    monkeypatch.setattr(deadlines.time, "sleep", sleeps.append)

    with pytest.raises(deadlines.BoundedSubprocessTimeout):
        deadlines.run_bounded(
            ["cargo", "test"],
            operation=deadlines.SubprocessOperation.STATISTICAL_TEST,
        )

    assert signals == [
        (fake.pid, signal.SIGTERM),
        (fake.pid, 0),
        (fake.pid, signal.SIGKILL),
    ]
    assert sleeps == [deadlines.PROCESS_GROUP_GRACE_SECONDS]
    assert fake.communicate_calls == [1800.0, deadlines.PROCESS_REAP_TIMEOUT_SECONDS]
