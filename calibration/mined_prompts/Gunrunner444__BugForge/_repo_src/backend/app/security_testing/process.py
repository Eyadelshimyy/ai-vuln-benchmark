"""Injectable subprocess runner for external scanners.

Production uses :class:`SubprocessRunner`. Tests inject a fake so CI never
depends on ZAP, Nuclei, or Playwright being installed.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.security_testing.failures import ToolExecutionState


@dataclass(frozen=True)
class ProcessOutcome:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float = 0.0
    unavailable: bool = False

    @property
    def state(self) -> ToolExecutionState:
        if self.unavailable:
            return ToolExecutionState.TOOL_UNAVAILABLE
        if self.timed_out:
            return ToolExecutionState.TIMEOUT
        if self.returncode == 0:
            return ToolExecutionState.COMPLETED
        return ToolExecutionState.EXECUTION_ERROR


class ProcessRunner:
    """Protocol-like base so tests can substitute a fake."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> ProcessOutcome:
        raise NotImplementedError


class SubprocessRunner(ProcessRunner):
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> ProcessOutcome:
        command = tuple(str(part) for part in argv)
        if not command:
            return ProcessOutcome(
                argv=command,
                returncode=None,
                stdout="",
                stderr="empty command",
                timed_out=False,
                unavailable=True,
            )
        if shutil.which(command[0]) is None and not Path(command[0]).exists():
            return ProcessOutcome(
                argv=command,
                returncode=None,
                stdout="",
                stderr=f"{command[0]} not found",
                timed_out=False,
                unavailable=True,
            )
        try:
            completed = subprocess.run(  # noqa: S603 — argv is constructed by adapters
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd) if cwd is not None else None,
                env=dict(env) if env is not None else None,
                input=input_text,
                check=False,
            )
        except FileNotFoundError:
            return ProcessOutcome(
                argv=command,
                returncode=None,
                stdout="",
                stderr=f"{command[0]} not found",
                timed_out=False,
                unavailable=True,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode("utf-8", "replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode("utf-8", "replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return ProcessOutcome(
                argv=command,
                returncode=None,
                stdout=stdout,
                stderr=stderr or "timed out",
                timed_out=True,
                duration_seconds=timeout,
            )
        return ProcessOutcome(
            argv=command,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            timed_out=False,
        )


@dataclass
class FakeProcessRunner(ProcessRunner):
    """Test double. Never executes a real scanner binary."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False
    unavailable: bool = False
    last_argv: tuple[str, ...] = ()
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> ProcessOutcome:
        command = tuple(str(part) for part in argv)
        self.last_argv = command
        self.calls.append(command)
        return ProcessOutcome(
            argv=command,
            returncode=None if self.unavailable or self.timed_out else self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
            timed_out=self.timed_out,
            unavailable=self.unavailable,
            duration_seconds=0.0 if not self.timed_out else timeout,
        )
