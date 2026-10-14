"""Controlled local process execution for optional discovery tools.

Commands are argument lists. BugForge secrets are stripped. Output is redacted.
Nothing is downloaded. Startup, timeout, and a real exit are distinct states.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.security_testing.secrets import redact_text

_STRIP = frozenset(
    {
        "SECRET_KEY",
        "DATABASE_URL",
        "POSTGRES_PASSWORD",
        "REDIS_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "BUGFORGE_OPERATOR_TOKEN",
    }
)


@dataclass(frozen=True)
class ProcessResult:
    """What actually happened when BugForge tried to run a local executable.

    ``available`` means the bare command name resolved on PATH.
    ``started`` means the operating system spawned the process.
    A missing binary, a spawn error, a timeout, and a non-zero exit are different.
    """

    started: bool
    timed_out: bool
    return_code: int | None
    stdout: str
    stderr: str
    available: bool


def tool_path(name: str) -> str | None:
    return shutil.which(name)


def tool_version(name: str, args: tuple[str, ...] = ("--version",)) -> str:
    path = tool_path(name)
    if path is None:
        return ""
    try:
        completed = subprocess.run(
            [path, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0][:120] if text else ""


def run_command(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float = 30,
    stdin: str = "",
    env: dict[str, str] | None = None,
) -> ProcessResult:
    if not argv or "/" in argv[0] or argv[0].startswith("."):
        name = argv[0] if argv else "command"
        return ProcessResult(
            started=False,
            timed_out=False,
            return_code=None,
            stdout="",
            stderr=f"refusing path-qualified command {name}",
            available=False,
        )
    if tool_path(argv[0]) is None:
        return ProcessResult(
            started=False,
            timed_out=False,
            return_code=None,
            stdout="",
            stderr=f"{argv[0]} is not installed",
            available=False,
        )
    executable = tool_path(argv[0]) or argv[0]
    try:
        completed = subprocess.run(
            [executable, *argv[1:]],
            cwd=cwd,
            input=stdin,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_env() if env is None else env,
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
        return ProcessResult(
            started=True,
            timed_out=True,
            return_code=None,
            stdout=redact_text(stdout),
            stderr=redact_text(stderr or "timed out"),
            available=True,
        )
    except OSError as exc:
        return ProcessResult(
            started=False,
            timed_out=False,
            return_code=None,
            stdout="",
            stderr=redact_text(str(exc)),
            available=True,
        )
    return ProcessResult(
        started=True,
        timed_out=False,
        return_code=completed.returncode,
        stdout=redact_text(completed.stdout),
        stderr=redact_text(completed.stderr),
        available=True,
    )


def _env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _STRIP:
        env.pop(key, None)
    return env
