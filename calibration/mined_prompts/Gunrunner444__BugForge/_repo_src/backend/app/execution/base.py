"""
Execution abstraction — all test execution goes through this interface.

Never call subprocess or docker directly from outside this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExecutionConfig:
    """Parameters that control a single sandboxed execution."""

    command: list[str]
    working_directory: str
    timeout_seconds: int = 120
    memory_limit_mb: int = 512
    cpu_limit: float = 1.0
    environment: dict[str, str] = field(default_factory=dict)
    allow_network: bool = False
    # Host-side directory for artifacts produced by the command.
    # For Docker: mounted into the container at OUTPUT_CONTAINER_PATH.
    # For Local: the command path and host path are identical.
    output_dir: str | None = None
    # Additional host paths to mount read-only inside the container.
    # Keys are absolute host paths; values are absolute container paths.
    # Ignored by LocalTestExecutor (the host paths are already accessible).
    read_only_volumes: dict[str, str] = field(default_factory=dict)


# Container-side mount path for the output directory
OUTPUT_CONTAINER_PATH = "/bugforge-output"


@dataclass
class ExecutionResult:
    """Result returned by every executor implementation."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    error_message: str | None = None
    # Contents of files captured from output_dir after execution.
    # Key is the relative filename (e.g., "report.json").
    artifact_contents: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class TestExecutor(ABC):
    """Abstract interface for sandboxed test execution.

    Concrete implementations:
      - DockerTestExecutor  (Docker containers — use in production / CI)
      - LocalTestExecutor   (direct subprocess — development / trusted repos only)
    """

    @abstractmethod
    async def execute(self, config: ExecutionConfig) -> ExecutionResult:
        """Run the command and return the result, including any captured artifacts."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if this executor backend is usable in the current environment."""
        ...

    def get_artifact_write_path(self, config: ExecutionConfig, filename: str) -> str:
        """Return the path the *command* should write artifacts to.

        For LocalTestExecutor this equals the host path.
        For DockerTestExecutor this is the container-side path.
        Subclasses override when they need a different mapping.
        """
        if config.output_dir is None:
            raise ValueError("output_dir must be set before calling get_artifact_write_path")
        return f"{config.output_dir}/{filename}"
