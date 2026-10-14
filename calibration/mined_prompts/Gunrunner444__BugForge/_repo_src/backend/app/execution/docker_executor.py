"""
Docker-backed test executor.

Runs test commands inside disposable, resource-limited containers.
Secrets from the BugForge host environment are NEVER forwarded into containers.

Security model:
  - Network disabled by default (--network none)
  - Read-only root filesystem; only /tmp and the output directory are writable
  - All Linux capabilities dropped
  - No new privileges flag set
  - Repository source is mounted read-only
  - Only explicitly listed environment variables are passed

Limitations (document honestly):
  - Does not use seccomp / AppArmor profiles beyond Docker's defaults
  - The host Docker daemon socket is not mounted so containers cannot spawn sibling containers
  - Resource limits (CPU/memory) are best-effort on non-cgroups-v2 hosts
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
import time
from pathlib import Path

from app.execution.base import (
    OUTPUT_CONTAINER_PATH,
    ExecutionConfig,
    ExecutionResult,
    TestExecutor,
)

logger = logging.getLogger(__name__)

DEFAULT_PYTHON_IMAGE = "python:3.12-slim"


def local_image_present(image: str) -> bool:
    """True when Docker already has this image. This does not pull and does not run a target."""
    cleaned = image.strip()
    if not cleaned or any(mark in cleaned for mark in ("\n", "\x00", ";")):
        return False
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", cleaned],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


_MAX_ARTIFACT_NAME = 128
_MAX_ARTIFACT_BYTES = 65_536


def collect_output_artifacts(output_dir: str) -> dict[str, str]:
    """Read regular files that stay inside the output directory.

    Symlinks are not followed. A path that resolves outside the output root
    is ignored. This is the host-side boundary for container-writable mounts.
    """
    root = Path(output_dir).resolve()
    if not root.is_dir():
        return {}
    artifacts: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if len(path.name) > _MAX_ARTIFACT_NAME:
            continue
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if len(relative.as_posix()) > _MAX_ARTIFACT_NAME:
            continue
        try:
            data = resolved.read_bytes()
        except OSError:
            continue
        if len(data) > _MAX_ARTIFACT_BYTES:
            data = data[:_MAX_ARTIFACT_BYTES]
        artifacts[relative.as_posix()] = data.decode("utf-8", errors="replace")
    return artifacts


class DockerTestExecutor(TestExecutor):
    """Executes commands inside short-lived Docker containers."""

    def __init__(self, image: str = DEFAULT_PYTHON_IMAGE) -> None:
        self._image = image

    # ------------------------------------------------------------------
    # TestExecutor interface
    # ------------------------------------------------------------------

    def get_artifact_write_path(self, config: ExecutionConfig, filename: str) -> str:
        """Return the container-side path the command should write artifacts to."""
        return f"{OUTPUT_CONTAINER_PATH}/{filename}"

    async def is_available(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except (OSError, FileNotFoundError):
            return False

    async def execute(self, config: ExecutionConfig) -> ExecutionResult:
        docker_args = self._build_docker_args(config)
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=float(config.timeout_seconds) + 15
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
        except Exception as exc:
            logger.exception("Docker execution failed: %s", exc)
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                duration_seconds=time.monotonic() - start,
                error_message=str(exc),
            )

        duration = time.monotonic() - start

        artifacts = collect_output_artifacts(config.output_dir) if config.output_dir else {}

        return ExecutionResult(
            exit_code=proc.returncode or 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_seconds=duration,
            artifact_contents=artifacts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_docker_args(self, config: ExecutionConfig) -> list[str]:
        args = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none" if not config.allow_network else "bridge",
            "--memory",
            f"{config.memory_limit_mb}m",
            "--cpus",
            str(config.cpu_limit),
            "--stop-timeout",
            str(config.timeout_seconds),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=128m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-w",
            config.working_directory,
            # Working directory (test files) — read-only inside container
            "-v",
            f"{config.working_directory}:{config.working_directory}:ro",
        ]

        # Mount any additional read-only volumes (e.g. the analyzed repository)
        for host_path, container_path in config.read_only_volumes.items():
            args += ["-v", f"{host_path}:{container_path}:ro"]

        # Mount output_dir for artifact capture (read-write)
        if config.output_dir:
            args += ["-v", f"{config.output_dir}:{OUTPUT_CONTAINER_PATH}:rw"]

        # Only forward explicitly whitelisted environment variables
        for key, value in config.environment.items():
            args += ["-e", f"{key}={value}"]

        args.append(self._image)
        args += ["sh", "-c", shlex.join(config.command)]
        return args
