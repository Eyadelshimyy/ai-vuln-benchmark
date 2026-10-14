"""Operation-specific bounded subprocess execution for repository automation.

The helper deliberately separates short metadata and inventory commands from
long-running statistical studies.  A timeout is treated as operational evidence,
not scientific evidence: the raised error exposes only the operation class and
the configured deadline, never the child command or captured output.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

PROCESS_GROUP_GRACE_SECONDS = 5.0
# Bound every post-timeout communicate/reap so a stuck child cannot hang the
# parent after SIGTERM/SIGKILL (fail-closed evidence still returns promptly).
PROCESS_REAP_TIMEOUT_SECONDS = 5.0

_Command = list[str] | tuple[str, ...]


class SubprocessOperation(str, Enum):
    """Supported subprocess operation classes with independent deadline policies."""

    CARGO_METADATA = "cargo_metadata"
    CARGO_TEST_LIST = "cargo_test_list"
    STATISTICAL_TEST = "statistical_test"


@dataclass(frozen=True)
class _DeadlinePolicy:
    """Internal immutable deadline configuration for one operation class."""

    env_key: str
    default_seconds: int
    minimum_seconds: int
    maximum_seconds: int


_POLICIES = {
    SubprocessOperation.CARGO_METADATA: _DeadlinePolicy(
        "FAST_MLSIRM_CARGO_METADATA_TIMEOUT_SECONDS",
        30,
        5,
        120,
    ),
    SubprocessOperation.CARGO_TEST_LIST: _DeadlinePolicy(
        "FAST_MLSIRM_CARGO_TEST_LIST_TIMEOUT_SECONDS",
        120,
        30,
        600,
    ),
    SubprocessOperation.STATISTICAL_TEST: _DeadlinePolicy(
        "FAST_MLSIRM_STATISTICAL_TEST_TIMEOUT_SECONDS",
        1800,
        60,
        7200,
    ),
}


class BoundedSubprocessTimeout(RuntimeError):
    """Redacted timeout error for one operation class."""

    def __init__(
        self,
        *,
        operation: SubprocessOperation,
        timeout_seconds: float,
    ) -> None:
        """Create timeout evidence without retaining child-controlled text."""
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"{operation.value} exceeded bounded timeout "
            f"({timeout_seconds:g} seconds)"
        )

    def as_dict(self) -> dict[str, object]:
        """Return machine-readable timeout evidence without child-controlled data."""
        return {
            "status": "timeout",
            "operation": self.operation.value,
            "timeout_seconds": self.timeout_seconds,
        }


def resolve_timeout_seconds(
    operation: SubprocessOperation,
    environ: Mapping[str, str] | None = None,
) -> float:
    """Return the bounded timeout for one operation and optional environment."""
    if not isinstance(operation, SubprocessOperation):
        raise ValueError("operation must be a SubprocessOperation")
    policy = _POLICIES[operation]
    source = os.environ if environ is None else environ
    raw = source.get(policy.env_key)
    if raw is None:
        return float(policy.default_seconds)
    if not raw.isdecimal():
        raise ValueError(
            f"{operation.name} timeout must be an integer number of seconds "
            f"between {policy.minimum_seconds} and {policy.maximum_seconds}"
        )
    seconds = int(raw)
    if not policy.minimum_seconds <= seconds <= policy.maximum_seconds:
        raise ValueError(
            f"{operation.name} timeout must be between "
            f"{policy.minimum_seconds} and {policy.maximum_seconds} seconds"
        )
    return float(seconds)


def _validate_command(command: _Command) -> list[str]:
    """Return a plain argument vector after rejecting malformed command fields."""
    if type(command) not in {list, tuple} or not command:
        raise ValueError("command must be a non-empty sequence of strings")
    if not all(type(part) is str and part for part in command):
        raise ValueError(
            "command must be a non-empty sequence of non-empty strings"
        )
    return list(command)


def _posix_process_group_exists(process_group_id: int) -> bool:
    """Return whether a POSIX process group still exists without changing it."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Fail closed: inability to signal does not prove that the group is gone.
        return True
    return True


def _close_process_pipes(process: subprocess.Popen) -> None:
    """Close inherited process pipe handles without inspecting child output."""
    for name in ("stdin", "stdout", "stderr"):
        pipe = getattr(process, name, None)
        if pipe is None:
            continue
        try:
            pipe.close()
        except (OSError, ValueError):
            # Cleanup is best-effort after the child has already exceeded both
            # its operation deadline and the bounded reap interval.
            pass


def _bounded_reap(process: subprocess.Popen) -> None:
    """Bound final child reaping and release inherited pipes if communicate stalls."""
    try:
        process.communicate(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        _close_process_pipes(process)

    wait = getattr(process, "wait", None)
    if wait is None:
        return
    try:
        wait(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # The timeout path itself must remain bounded. The caller will surface
        # BoundedSubprocessTimeout; no child-controlled text is retained here.
        return


def _terminate_after_timeout(process: subprocess.Popen) -> None:
    """Terminate and boundedly reap one timed-out process or POSIX process group."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            _bounded_reap(process)
            return

        # Keep the group leader unreaped during the grace period so its numeric
        # process-group identifier cannot be recycled before the final liveness
        # check. A leader may exit while a descendant in the same group survives.
        time.sleep(PROCESS_GROUP_GRACE_SECONDS)
        if _posix_process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _bounded_reap(process)
        return

    process.terminate()
    try:
        process.communicate(timeout=PROCESS_GROUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        _bounded_reap(process)


def run_bounded(
    command: _Command,
    *,
    operation: SubprocessOperation,
    check: bool = False,
    capture_output: bool = False,
    text: bool = False,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run one command with an operation-specific timeout and bounded cleanup."""
    argv = _validate_command(command)
    timeout_seconds = resolve_timeout_seconds(operation)
    popen_kwargs: dict[str, object] = {
        "text": text,
        "env": env,
    }
    if capture_output:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(argv, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_after_timeout(process)
        raise BoundedSubprocessTimeout(
            operation=operation,
            timeout_seconds=timeout_seconds,
        ) from None

    completed = subprocess.CompletedProcess(
        argv,
        process.returncode,
        stdout,
        stderr,
    )
    if check:
        completed.check_returncode()
    return completed
