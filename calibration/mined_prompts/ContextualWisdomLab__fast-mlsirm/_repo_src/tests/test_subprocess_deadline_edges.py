"""Edge contracts for bounded subprocess deadline execution."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


_MODULE = Path(__file__).parents[1] / "scripts" / "_subprocess_deadlines.py"
_SPEC = importlib.util.spec_from_file_location("subprocess_deadlines_edges", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
deadlines = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deadlines
_SPEC.loader.exec_module(deadlines)


class _FakePipe:
    """Closable inherited pipe stand-in for final-reap cleanup evidence."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        """Record that the parent released its inherited pipe handle."""
        self.closed = True


class _FakeProcess:
    """Small timeout-aware process stand-in for platform cleanup branches."""

    def __init__(self, *, timeouts: int, wait_times_out: bool = False) -> None:
        self.pid = 5252
        self.returncode = 0
        self._timeouts = timeouts
        self._wait_times_out = wait_times_out
        self.communicate_calls: list[float | None] = []
        self.wait_calls: list[float | None] = []
        self.terminated = 0
        self.killed = 0
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self.stderr = _FakePipe()

    def communicate(self, timeout=None):
        """Raise the configured number of TimeoutExpired exceptions, then finish."""
        self.communicate_calls.append(timeout)
        if len(self.communicate_calls) <= self._timeouts:
            raise subprocess.TimeoutExpired(["opaque"], timeout)
        return (None, None)

    def wait(self, timeout=None):
        """Model bounded final wait behavior after inherited pipes are closed."""
        self.wait_calls.append(timeout)
        if self._wait_times_out:
            raise subprocess.TimeoutExpired(["opaque"], timeout)
        self.returncode = -9
        return self.returncode

    def terminate(self):
        """Record graceful direct-process termination."""
        self.terminated += 1

    def kill(self):
        """Record forced direct-process termination."""
        self.killed += 1


def test_default_resolution_reads_environment_only_when_mapping_is_omitted(monkeypatch) -> None:
    """The process environment is default configuration, not an implicit merge."""
    key = "FAST_MLSIRM_CARGO_TEST_LIST_TIMEOUT_SECONDS"
    monkeypatch.setenv(key, "240")
    assert deadlines.resolve_timeout_seconds(
        deadlines.SubprocessOperation.CARGO_TEST_LIST
    ) == 240.0
    assert deadlines.resolve_timeout_seconds(
        deadlines.SubprocessOperation.CARGO_TEST_LIST,
        {},
    ) == 120.0


def test_invalid_operation_and_command_vectors_fail_before_process_creation(monkeypatch) -> None:
    """Malformed operation and argument-vector inputs never reach Popen."""
    with pytest.raises(ValueError, match="SubprocessOperation"):
        deadlines.resolve_timeout_seconds("cargo_metadata", {})

    monkeypatch.setattr(
        deadlines.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not be called"),
    )
    for command in ([], "cargo metadata", ["cargo", ""], ["cargo", 3]):
        with pytest.raises(ValueError, match="command"):
            deadlines.run_bounded(
                command,
                operation=deadlines.SubprocessOperation.CARGO_METADATA,
            )


def test_posix_timeout_handles_process_group_that_already_exited(monkeypatch) -> None:
    """A disappeared group receives only a bounded final reap attempt."""
    fake = _FakeProcess(timeouts=1)

    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(
        deadlines.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )

    with pytest.raises(deadlines.BoundedSubprocessTimeout):
        deadlines.run_bounded(
            ["cargo", "metadata"],
            operation=deadlines.SubprocessOperation.CARGO_METADATA,
        )
    assert fake.communicate_calls == [30.0, deadlines.PROCESS_REAP_TIMEOUT_SECONDS]


def test_non_posix_timeout_terminates_then_escalates_direct_process(monkeypatch) -> None:
    """Non-POSIX fallback uses bounded terminate, kill, and final reap cleanup."""
    fake = _FakeProcess(timeouts=2)
    observed: dict[str, object] = {}

    def fake_popen(_command, **kwargs):
        observed.update(kwargs)
        return fake

    monkeypatch.setattr(deadlines.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        deadlines.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        512,
        raising=False,
    )
    monkeypatch.setattr(deadlines.subprocess, "Popen", fake_popen)

    with pytest.raises(deadlines.BoundedSubprocessTimeout):
        deadlines.run_bounded(
            ["cargo", "test"],
            operation=deadlines.SubprocessOperation.CARGO_TEST_LIST,
        )

    assert observed["creationflags"] == 512
    assert "start_new_session" not in observed
    assert fake.terminated == 1
    assert fake.killed == 1
    assert fake.communicate_calls == [
        120.0,
        deadlines.PROCESS_GROUP_GRACE_SECONDS,
        deadlines.PROCESS_REAP_TIMEOUT_SECONDS,
    ]


def test_non_posix_timeout_stops_after_graceful_termination(monkeypatch) -> None:
    """A direct process that exits after terminate is not killed unnecessarily."""
    fake = _FakeProcess(timeouts=1)
    monkeypatch.setattr(deadlines.os, "name", "other", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)

    with pytest.raises(deadlines.BoundedSubprocessTimeout):
        deadlines.run_bounded(
            ["cargo", "metadata"],
            operation=deadlines.SubprocessOperation.CARGO_METADATA,
        )

    assert fake.terminated == 1
    assert fake.killed == 0
    assert fake.communicate_calls == [30.0, deadlines.PROCESS_GROUP_GRACE_SECONDS]


def test_final_reap_timeout_closes_inherited_pipes_and_remains_bounded(monkeypatch) -> None:
    """A stuck post-kill communicate cannot make the timeout cleanup unbounded."""
    fake = _FakeProcess(timeouts=3, wait_times_out=True)
    monkeypatch.setattr(deadlines.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        deadlines.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        512,
        raising=False,
    )
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)

    with pytest.raises(deadlines.BoundedSubprocessTimeout) as caught:
        deadlines.run_bounded(
            ["cargo", "test", "--opaque-value=do-not-echo"],
            operation=deadlines.SubprocessOperation.CARGO_TEST_LIST,
            capture_output=True,
            text=True,
        )

    assert fake.terminated == 1
    assert fake.killed == 1
    assert fake.communicate_calls == [
        120.0,
        deadlines.PROCESS_GROUP_GRACE_SECONDS,
        deadlines.PROCESS_REAP_TIMEOUT_SECONDS,
    ]
    assert fake.wait_calls == [deadlines.PROCESS_REAP_TIMEOUT_SECONDS]
    assert fake.stdin.closed is True
    assert fake.stdout.closed is True
    assert fake.stderr.closed is True
    assert "do-not-echo" not in str(caught.value)
