from __future__ import annotations

import logging
import shutil

from app.execution.base import ExecutionConfig, ExecutionResult, TestExecutor
from app.execution.docker_executor import DockerTestExecutor
from app.execution.local_executor import LocalTestExecutor

logger = logging.getLogger(__name__)

__all__ = [
    "TestExecutor",
    "ExecutionConfig",
    "ExecutionResult",
    "DockerTestExecutor",
    "LocalTestExecutor",
    "ExecutorFactory",
]


class ExecutorFactory:
    """Select the best available executor at runtime.

    Preference order:
      1. DockerTestExecutor — when Docker is installed and reachable.
      2. LocalTestExecutor  — fallback for *development* environments only.

    The LocalTestExecutor runs code directly on the host and is NOT safe for
    production use with untrusted repositories.  In production, Docker is
    required; the factory raises RuntimeError rather than silently falling back.
    """

    @staticmethod
    def create(prefer_docker: bool = True) -> TestExecutor:
        """Return the best synchronously-detectable executor.

        Raises RuntimeError in production when Docker is unavailable to prevent
        silent execution of untrusted code on the BugForge host.
        """
        from app.core.config import settings

        if prefer_docker and shutil.which("docker") is not None:
            return DockerTestExecutor()

        if settings.environment == "production":
            raise RuntimeError(
                "Docker is required for sandboxed execution in production but was not found. "
                "Install Docker or set ENVIRONMENT=development to allow local execution "
                "(development-only — not safe for untrusted repositories)."
            )

        if prefer_docker:
            logger.warning(
                "Docker not found; falling back to LocalTestExecutor "
                "(development only — do not use with untrusted code)"
            )
        return LocalTestExecutor()
