"""RED contracts for operation-specific bounded subprocess execution."""

from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
from pathlib import Path

import pytest


_MODULE = Path(__file__).parents[1] / "scripts" / "_subprocess_deadlines.py"
_SPEC = importlib.util.spec_from_file_location("subprocess_deadlines", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
deadlines = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = deadlines
_SPEC.loader.exec_module(deadlines)


def test_operation_defaults_distinguish_metadata_inventory_and_long_studies() -> None:
    """Scientific studies receive more time than metadata or inventory commands."""
    metadata = deadlines.resolve_timeout_seconds(deadlines.SubprocessOperation.CARGO_METADATA, {})
    inventory = deadlines.resolve_timeout_seconds(deadlines.SubprocessOperation.CARGO_TEST_LIST, {})
    study = deadlines.resolve_timeout_seconds(deadlines.SubprocessOperation.STATISTICAL_TEST, {})

    assert metadata == 30.0
    assert inventory == 120.0
    assert study == 1800.0
    assert metadata < inventory < study


def test_bounded_environment_override_is_integer_and_within_operation_range() -> None:
    """Operators may tune deadlines only inside the declared safe range."""
    key = "FAST_MLSIRM_CARGO_METADATA_TIMEOUT_SECONDS"
    assert deadlines.resolve_timeout_seconds(
        deadlines.SubprocessOperation.CARGO_METADATA,
        {key: "45"},
    ) == 45.0

    for invalid in ("", "4", "121", "30.5", "true", "1e2"):
        with pytest.raises(ValueError, match="CARGO_METADATA"):
            deadlines.resolve_timeout_seconds(
                deadlines.SubprocessOperation.CARGO_METADATA,
                {key: invalid},
            )


def test_timeout_error_is_machine_readable_and_omits_child_controlled_text() -> None:
    """Timeout evidence exposes the operation and deadline but no child-controlled data."""
    error = deadlines.BoundedSubprocessTimeout(
        operation=deadlines.SubprocessOperation.STATISTICAL_TEST,
        timeout_seconds=1800.0,
    )
    payload = error.as_dict()

    assert payload == {
        "status": "timeout",
        "operation": "statistical_test",
        "timeout_seconds": 1800.0,
    }
    rendered = str(error)
    assert "statistical_test" in rendered
    assert "1800" in rendered
    assert "command" not in rendered.lower()


class _FakeProcess:
    """Minimal deterministic Popen stand-in for bounded-runner tests."""

    def __init__(
        self,
        *,
        pid: int = 4242,
        returncode: int = 0,
        timeout_once: bool = False,
        timeout_twice: bool = False,
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self._timeout_once = timeout_once
        self._timeout_twice = timeout_twice
        self.communicate_calls: list[float | None] = []

    def communicate(self, timeout=None):
        """Return deterministic output or raise the requested timeout sequence."""
        self.communicate_calls.append(timeout)
        call = len(self.communicate_calls)
        if self._timeout_once and call == 1:
            raise subprocess.TimeoutExpired(["opaque-child"], timeout)
        if self._timeout_twice and call in {1, 2}:
            raise subprocess.TimeoutExpired(["opaque-child"], timeout)
        self.returncode = 0
        return ("opaque-stdout", "opaque-stderr")

    def terminate(self):
        """Record direct-process termination on non-POSIX fallback paths."""
        self.returncode = -15

    def kill(self):
        """Record direct-process kill on non-POSIX fallback paths."""
        self.returncode = -9


def test_successful_run_uses_isolated_process_group_and_returns_completed_process(monkeypatch) -> None:
    """POSIX commands run in a new session so timeout cleanup can address descendants."""
    fake = _FakeProcess(returncode=0)
    observed: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return fake

    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", fake_popen)

    completed = deadlines.run_bounded(
        ["cargo", "metadata"],
        operation=deadlines.SubprocessOperation.CARGO_METADATA,
        capture_output=True,
        text=True,
    )

    assert completed.args == ["cargo", "metadata"]
    assert completed.returncode == 0
    assert completed.stdout == "opaque-stdout"
    assert completed.stderr == "opaque-stderr"
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    assert fake.communicate_calls == [30.0]


def test_posix_timeout_terminates_process_group_and_omits_child_output(monkeypatch) -> None:
    """A terminated group is probed after the full grace period before reaping."""
    fake = _FakeProcess(timeout_once=True)
    signals: list[tuple[int, int]] = []
    sleeps: list[float] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        signals.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(deadlines.os, "killpg", fake_killpg)
    monkeypatch.setattr(deadlines.time, "sleep", sleeps.append)

    with pytest.raises(deadlines.BoundedSubprocessTimeout) as caught:
        deadlines.run_bounded(
            ["cargo", "test", "--opaque-value=do-not-echo"],
            operation=deadlines.SubprocessOperation.STATISTICAL_TEST,
            capture_output=True,
            text=True,
        )

    assert signals == [(fake.pid, signal.SIGTERM), (fake.pid, 0)]
    assert sleeps == [deadlines.PROCESS_GROUP_GRACE_SECONDS]
    assert fake.communicate_calls == [1800.0, deadlines.PROCESS_REAP_TIMEOUT_SECONDS]
    assert "do-not-echo" not in str(caught.value)
    assert "opaque-stdout" not in str(caught.value)


def test_posix_timeout_escalates_to_sigkill_when_group_ignores_sigterm(monkeypatch) -> None:
    """A group that survives the grace period receives SIGKILL before leader reap."""
    fake = _FakeProcess(timeout_once=True)
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


def test_posix_timeout_reaps_when_group_exits_between_probe_and_sigkill(monkeypatch) -> None:
    """A normal exit after liveness probing must not turn timeout cleanup into failure."""
    fake = _FakeProcess(timeout_once=True)
    signals: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        signals.append((pgid, sig))
        if sig == signal.SIGKILL:
            raise ProcessLookupError

    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(deadlines.os, "killpg", fake_killpg)
    monkeypatch.setattr(deadlines.time, "sleep", lambda _seconds: None)

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
    assert fake.communicate_calls == [1800.0, deadlines.PROCESS_REAP_TIMEOUT_SECONDS]


def test_process_group_probe_treats_permission_denial_as_still_alive(monkeypatch) -> None:
    """EPERM must fail closed rather than being mistaken for group disappearance."""
    monkeypatch.setattr(
        deadlines.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(PermissionError()),
    )
    assert deadlines._posix_process_group_exists(4242) is True


def test_nonzero_exit_preserves_check_semantics(monkeypatch) -> None:
    """The wrapper still raises CalledProcessError when callers request check=True."""
    fake = _FakeProcess(returncode=7)

    def communicate(timeout=None):
        fake.communicate_calls.append(timeout)
        fake.returncode = 7
        return ("", "")

    fake.communicate = communicate
    monkeypatch.setattr(deadlines.os, "name", "posix", raising=False)
    monkeypatch.setattr(deadlines.subprocess, "Popen", lambda *_args, **_kwargs: fake)

    with pytest.raises(subprocess.CalledProcessError) as caught:
        deadlines.run_bounded(
            ["cargo", "metadata"],
            operation=deadlines.SubprocessOperation.CARGO_METADATA,
            check=True,
        )
    assert caught.value.returncode == 7
