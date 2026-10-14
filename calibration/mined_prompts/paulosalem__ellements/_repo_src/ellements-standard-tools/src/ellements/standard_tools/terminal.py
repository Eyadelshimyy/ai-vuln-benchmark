"""Reusable terminal CLI tool for ellements applications."""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ellements.core import ToolSpec
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TerminalExecutionSpec(BaseModel):
    """Trusted local terminal execution request."""

    model_config = ConfigDict(extra="forbid")

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    shell: bool = False
    script: str | None = None
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    stdin: str | None = None
    timeout_seconds: float = Field(default=300.0, gt=0)
    max_output_bytes: int = Field(default=100_000, gt=0)

    @model_validator(mode="after")
    def _validate_command(self) -> TerminalExecutionSpec:
        if self.shell:
            if not self.script:
                raise ValueError("shell terminal execution requires script")
        elif not self.command:
            raise ValueError("argv terminal execution requires command")
        return self


class TerminalExecutionResult(BaseModel):
    """Captured result of a terminal execution."""

    model_config = ConfigDict(extra="forbid")

    command_line: list[str]
    cwd: str | None
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False


async def run_terminal_execution(
    spec: TerminalExecutionSpec,
) -> TerminalExecutionResult:
    """Run a local terminal command with bounded output."""
    started_at = datetime.now(UTC)
    start = time.monotonic()
    env = {**os.environ, **spec.env}
    stdin_bytes = spec.stdin.encode("utf-8") if spec.stdin is not None else None
    command_line = (
        [spec.script or ""] if spec.shell else [spec.command or "", *spec.args]
    )
    process = await _start_process(spec, env)
    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(input=stdin_bytes),
            timeout=spec.timeout_seconds,
        )
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
    completed_at = datetime.now(UTC)
    stdout, stdout_truncated = _decode_bounded(stdout_bytes, spec.max_output_bytes)
    stderr, stderr_truncated = _decode_bounded(stderr_bytes, spec.max_output_bytes)
    return TerminalExecutionResult(
        command_line=command_line,
        cwd=spec.cwd,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=time.monotonic() - start,
        exit_code=process.returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def terminal_cli_tool(
    *,
    name: str = "terminal.run",
    description: str = "Run a local terminal command with bounded output.",
    timeout_seconds: float = 300.0,
    max_output_bytes: int = 100_000,
) -> ToolSpec:
    """Return an ellements `ToolSpec` for local CLI execution."""

    async def invoke(
        command: str | None = None,
        args: list[str] | None = None,
        shell: bool = False,
        script: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: str | None = None,
        timeout_seconds: float | None = None,
        max_output_bytes: int | None = None,
    ) -> dict[str, Any]:
        spec = TerminalExecutionSpec(
            command=command,
            args=args or [],
            shell=shell,
            script=script,
            cwd=str(Path(cwd)) if cwd is not None else None,
            env=env or {},
            stdin=stdin,
            timeout_seconds=timeout_seconds or terminal_cli_tool_timeout,
            max_output_bytes=max_output_bytes or terminal_cli_tool_max_output,
        )
        return (await run_terminal_execution(spec)).model_dump(mode="json")

    terminal_cli_tool_timeout = timeout_seconds
    terminal_cli_tool_max_output = max_output_bytes
    return ToolSpec(
        name=name,
        description=description,
        params_json_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "shell": {"type": "boolean", "default": False},
                "script": {"type": "string"},
                "cwd": {"type": "string"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "stdin": {"type": "string"},
                "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
                "max_output_bytes": {"type": "integer", "exclusiveMinimum": 0},
            },
        },
        invoke=invoke,
    )


async def _start_process(
    spec: TerminalExecutionSpec,
    env: dict[str, str],
) -> asyncio.subprocess.Process:
    if spec.shell:
        assert spec.script is not None
        return await asyncio.create_subprocess_shell(
            spec.script,
            cwd=spec.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE if spec.stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    assert spec.command is not None
    return await asyncio.create_subprocess_exec(
        spec.command,
        *spec.args,
        cwd=spec.cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE if spec.stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


def _decode_bounded(data: bytes, max_bytes: int) -> tuple[str, bool]:
    truncated = len(data) > max_bytes
    if truncated:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace"), truncated


def terminal_command_text(spec: TerminalExecutionSpec) -> str:
    """Return a display-safe command string."""
    if spec.shell:
        return spec.script or ""
    return " ".join(shlex.quote(part) for part in [spec.command or "", *spec.args])


__all__ = [
    "TerminalExecutionResult",
    "TerminalExecutionSpec",
    "run_terminal_execution",
    "terminal_cli_tool",
    "terminal_command_text",
]
