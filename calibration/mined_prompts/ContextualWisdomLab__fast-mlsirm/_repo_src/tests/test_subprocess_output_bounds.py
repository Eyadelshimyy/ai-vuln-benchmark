"""Regression coverage for bounded subprocess capture in operator scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from scripts import build_pr_queue_governance as governance
from scripts import build_procurement_due_diligence as procurement
from scripts._bounded_subprocess import BoundedSubprocessOutputError


def test_bounded_capture_rejects_stdout_overflow() -> None:
    """The shared runner must cap stdout while the child is still running."""
    from scripts._bounded_subprocess import run_bounded_capture

    with pytest.raises(BoundedSubprocessOutputError, match="stdout"):
        run_bounded_capture(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 4096)"],
            timeout_seconds=5,
            max_stdout_bytes=64,
            max_stderr_bytes=64,
        )


def test_bounded_capture_rejects_missing_capture_pipes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipe setup failures remain explicit when Python assertions are optimized away."""
    class MissingPipes:
        stdout = None
        stderr = None

        def kill(self) -> None:
            """Record the required cleanup operation."""

        def wait(self) -> None:
            """Record the required reap operation."""

    monkeypatch.setattr(
        "scripts._bounded_subprocess.subprocess.Popen",
        lambda *args, **kwargs: MissingPipes(),
    )

    from scripts._bounded_subprocess import run_bounded_capture

    with pytest.raises(RuntimeError, match="requires stdout and stderr pipes"):
        run_bounded_capture(
            [sys.executable, "-c", ""],
            timeout_seconds=5,
            max_stdout_bytes=64,
            max_stderr_bytes=64,
        )


def test_bounded_capture_rejects_stderr_overflow() -> None:
    """The shared runner must cap stderr independently from stdout."""
    from scripts._bounded_subprocess import run_bounded_capture

    with pytest.raises(BoundedSubprocessOutputError, match="stderr"):
        run_bounded_capture(
            [sys.executable, "-c", "import sys; sys.stderr.write('x' * 4096)"],
            timeout_seconds=5,
            max_stdout_bytes=64,
            max_stderr_bytes=64,
        )


def test_bounded_capture_invalid_utf8_is_data_error() -> None:
    """Machine-readable stdout must never be silently replacement-decoded."""
    from scripts._bounded_subprocess import run_bounded_capture

    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'{\"record\":\"ok' + bytes([255]) + b'\"}')",
    ]
    completed = run_bounded_capture(
        command,
        timeout_seconds=5,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )
    assert completed.returncode == 65
    assert completed.stdout == ""
    assert "not valid UTF-8" in completed.stderr
    assert "�" not in completed.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership contract")
def test_bounded_capture_deadline_kills_pipe_inheriting_descendants() -> None:
    """A descendant holding captured pipes cannot extend the configured deadline."""
    from scripts._bounded_subprocess import run_bounded_capture

    grandchild = "import time; time.sleep(2)"
    child = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "sys.exit(0)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded_capture(
            [sys.executable, "-c", child],
            timeout_seconds=0.2,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )
    assert time.monotonic() - started < 1.0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership contract")
def test_bounded_capture_reports_descendant_overflow_before_timeout() -> None:
    """A descendant-held pipe must preserve overflow precedence over timeout."""
    from scripts._bounded_subprocess import run_bounded_capture

    grandchild = "import sys, time; sys.stdout.write('x' * 4096); sys.stdout.flush(); time.sleep(2)"
    child = (
        "import subprocess, sys; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "sys.exit(0)"
    )
    with pytest.raises(BoundedSubprocessOutputError, match="stdout"):
        run_bounded_capture(
            [sys.executable, "-c", child],
            timeout_seconds=1.0,
            max_stdout_bytes=64,
            max_stderr_bytes=1024,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership contract")
def test_process_tree_termination_reaps_the_owned_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kill path must reap the direct child within its bounded cleanup window."""
    from scripts import _bounded_subprocess as bounded

    class FakeProcess:
        pid = 4242

        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            self.wait_timeouts.append(timeout)

    process = FakeProcess()
    monkeypatch.setattr(bounded.os, "killpg", lambda *_args: None)

    bounded._terminate_process_tree(process)  # type: ignore[arg-type]

    assert process.wait_timeouts == [bounded._PROCESS_REAP_TIMEOUT_SECONDS]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership contract")
def test_process_tree_termination_does_not_resignal_reaped_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated cleanup must never signal a process group after the child is reaped."""
    from scripts import _bounded_subprocess as bounded

    class ReapedProcess:
        pid = 4242

        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> None:
            self.wait_timeouts.append(timeout)

    process = ReapedProcess()

    def fail_if_signalled(*_args: object) -> None:
        raise AssertionError("reaped process group must not be signalled")

    monkeypatch.setattr(bounded.os, "killpg", fail_if_signalled)

    bounded._terminate_process_tree(process)  # type: ignore[arg-type]

    assert process.wait_timeouts == [bounded._PROCESS_REAP_TIMEOUT_SECONDS]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group ownership contract")
def test_process_tree_cleanup_ignores_signal_permission_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup signal failures must not replace the bounded subprocess error."""
    from scripts import _bounded_subprocess as bounded

    class LiveProcess:
        pid = 4242

        def __init__(self) -> None:
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            self.wait_timeouts.append(timeout)

    process = LiveProcess()

    def deny_signal(*_args: object) -> None:
        raise PermissionError("signal denied")

    monkeypatch.setattr(bounded.os, "killpg", deny_signal)
    bounded._terminate_process_tree(process)  # type: ignore[arg-type]

    assert process.wait_timeouts == [bounded._PROCESS_REAP_TIMEOUT_SECONDS]


def test_governance_parse_failure_is_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed successful gh output must not crash the governance builder."""
    monkeypatch.setattr(
        governance,
        "run_bounded_capture",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "{", ""),
    )

    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 65
    assert "JSON" in error["stderr"]


def test_governance_decode_failure_is_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid UTF-8 from gh must retain the helper's data-error status."""
    monkeypatch.setattr(
        governance,
        "run_bounded_capture",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 65, "", "stdout was not valid UTF-8"
        ),
    )
    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )
    assert payload is None
    assert error is not None
    assert error["returncode"] == 65
    assert "UTF-8" in error["stderr"]


def test_governance_timeout_remains_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bounded-output successor must preserve the parent timeout contract."""
    def raise_timeout(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(governance, "run_bounded_capture", raise_timeout)

    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 124
    assert "timed out" in error["stderr"]


def test_governance_overflow_is_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized gh output must fail closed through the governance error schema."""
    def raise_overflow(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise BoundedSubprocessOutputError("stdout", 64)

    monkeypatch.setattr(governance, "run_bounded_capture", raise_overflow)

    payload, error = governance._run_gh_json(
        ["gh", "api", "repos/example/project"],
        max_attempts=1,
        retry_sleep_seconds=0,
    )

    assert payload is None
    assert error is not None
    assert error["returncode"] == 75
    assert "stdout" in error["stderr"]


def test_procurement_parse_failure_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed gh JSON must be recorded as evidence failure, not raised."""
    monkeypatch.setattr(
        procurement,
        "run_bounded_capture",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "{", ""),
    )

    snapshot = procurement._github_snapshot("example/project", offline=False)

    assert snapshot["repo"]["ok"] is False
    assert snapshot["repo"]["returncode"] == 65
    assert snapshot["repo"]["data"] is None
    assert "JSON" in snapshot["repo"]["stderr"]


def test_procurement_decode_failure_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid UTF-8 must remain a stable procurement evidence failure."""
    monkeypatch.setattr(
        procurement,
        "run_bounded_capture",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 65, "", "stdout was not valid UTF-8"
        ),
    )
    snapshot = procurement._github_snapshot("example/project", offline=False)
    assert snapshot["repo"]["ok"] is False
    assert snapshot["repo"]["returncode"] == 65
    assert snapshot["repo"]["data"] is None
    assert "UTF-8" in snapshot["repo"]["stderr"]


def test_procurement_overflow_is_snapshot_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized gh output must be recorded in procurement evidence, not raised."""
    def raise_overflow(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise BoundedSubprocessOutputError("stdout", 64)

    monkeypatch.setattr(procurement, "run_bounded_capture", raise_overflow)

    snapshot = procurement._github_snapshot("example/project", offline=False)

    assert snapshot["repo"]["ok"] is False
    assert snapshot["repo"]["returncode"] == 75
    assert snapshot["repo"]["data"] is None
    assert "stdout" in snapshot["repo"]["stderr"]


def test_procurement_lines_snapshot_success_is_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful text output becomes non-empty, whitespace-trimmed evidence lines."""
    monkeypatch.setattr(
        procurement,
        "run_bounded_capture",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "v1\n\n v2 \n", " warning \n"
        ),
    )

    snapshot = procurement._bounded_lines_snapshot(["gh", "release", "list"])

    assert snapshot == {
        "ok": True,
        "returncode": 0,
        "lines": ["v1", " v2 "],
        "stderr": "warning",
    }


def test_procurement_lines_snapshot_timeout_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timed-out text commands produce the stable empty-lines error schema."""
    monkeypatch.setattr(
        procurement,
        "run_bounded_capture",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], 1)
        ),
    )

    snapshot = procurement._bounded_lines_snapshot(["gh", "release", "list"])

    assert snapshot["ok"] is False
    assert snapshot["returncode"] == 124
    assert snapshot["lines"] == []
    assert "timed out" in snapshot["stderr"]
