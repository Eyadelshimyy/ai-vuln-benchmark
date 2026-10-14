"""
Local subprocess executor — for development and CI where Docker is unavailable.

WARNING: This executor does NOT sandbox the executed code.
         Only use it with trusted repositories in development environments.
         Never run untrusted repository code on a production BugForge host.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from app.execution.base import ExecutionConfig, ExecutionResult, TestExecutor

logger = logging.getLogger(__name__)

# BugForge server secrets that must never leak into analyzed project processes.
_STRIP_FROM_ENV: frozenset[str] = frozenset(
    {
        "SECRET_KEY",
        "DATABASE_URL",
        "POSTGRES_PASSWORD",
        "REDIS_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
    }
)


def _safe_env(overrides: dict[str, str]) -> dict[str, str]:
    """Build a safe environment: inherit the host env, strip secrets, apply overrides."""
    env = os.environ.copy()
    for key in _STRIP_FROM_ENV:
        env.pop(key, None)
    env.update(overrides)
    return env


class LocalTestExecutor(TestExecutor):
    """Runs commands directly on the host via asyncio subprocess.

    Only suitable for trusted repositories in development environments.
    Inherits the host PATH and Python environment so project tooling works,
    but strips known BugForge secrets before passing the environment to the process.
    """

    async def is_available(self) -> bool:
        return True

    async def execute(self, config: ExecutionConfig) -> ExecutionResult:
        start = time.monotonic()
        env = _safe_env(config.environment)

        try:
            proc = await asyncio.create_subprocess_exec(
                *config.command,
                cwd=config.working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except (OSError, FileNotFoundError) as exc:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_seconds=time.monotonic() - start,
                error_message=str(exc),
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(config.timeout_seconds)
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="Execution timed out",
                duration_seconds=time.monotonic() - start,
                timed_out=True,
            )

        duration = time.monotonic() - start

        # Read any artifact files produced into output_dir
        artifacts: dict[str, str] = {}
        if config.output_dir:
            output_path = Path(config.output_dir)
            if output_path.is_dir():
                for artifact in output_path.iterdir():
                    if artifact.is_file():
                        try:
                            artifacts[artifact.name] = artifact.read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except OSError:
                            pass

        return ExecutionResult(
            exit_code=proc.returncode or 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_seconds=duration,
            artifact_contents=artifacts,
        )
