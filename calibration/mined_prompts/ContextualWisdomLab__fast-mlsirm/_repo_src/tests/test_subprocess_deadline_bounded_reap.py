"""Fail-first contract for bounded timeout cleanup reaping."""

from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
from pathlib import Path

import pytest


_MODULE = Path(__file__).parents[1] / "scripts" / "_subprocess_deadlines.py"
_SPEC = importlib.util.spec_from_file_location("subprocess_deadlines_bounded_reap", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
deadlines = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deadlines
_SPEC.loader.exec_module(deadlines)


class _StallingProcess:
    """Popen stand-in whose initial operation and first reap both time out."""

    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = 0
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout=None):
        """Timeout twice so the cleanup path must bound its own final reap."""
        self.communicate_calls.append(timeout)
        if len(self.communicate_calls) <= 2:
            raise subprocess.TimeoutExpired(["opaque-child"], timeout)
        return ("", "")

    def terminate(self) -> None:
        """Record non-POSIX fallback termination."""
        self.returncode = -15

    def kill(self) -> None:
        """Record non-POSIX fallback kill."""
        self.returncode = -9


def test_posix_timeout_cleanup_bounds_final_reap(monkeypatch) -> None:
    """A timed-out reap must still surface stable package timeout evidence."""
    fake = _StallingProcess()
    signals: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        signals.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(deadlines.os, "killpg", fake_killpg)
    monkeypatch.setattr(deadlines.time, "sleep", lambda _seconds: None)

    with pytest.raises(deadlines.BoundedSubprocessTimeout):
        deadlines.run_bounded(
            ["cargo", "test", "--opaque-value=do-not-echo"],
            operation=deadlines.SubprocessOperation.STATISTICAL_TEST,
            capture_output=True,
            text=True,
        )

    assert signals == [(fake.pid, signal.SIGTERM), (fake.pid, 0)]
    assert fake.communicate_calls == [
        1800.0,
        deadlines.PROCESS_REAP_TIMEOUT_SECONDS,
    ]
