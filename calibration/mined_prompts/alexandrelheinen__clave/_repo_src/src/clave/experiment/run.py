"""The run record.

A number nobody can trace to its inputs is an anecdote. A run record carries
what produced a result: the seed, a digest of the configuration, the artifacts
consumed with the digests they had, and the environment that executed it.

It serializes as JSON so a later tool can read it without importing this
package, which matters because the tool comparing two runs may outlive the code
that wrote them.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

from clave.corpus.manifest import Manifest
from clave.errors import UnknownArtifactError

_TRACKED = ("numpy", "torch")


def config_digest(config: dict[str, Any]) -> str:
    """Digest a configuration so two identical ones are identifiable as such.

    The serialization sorts keys, so key order cannot make two equivalent
    configurations look different.

    Args:
        config: The configuration mapping.

    Returns:
        A lowercase hexadecimal SHA-256 digest.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode()).hexdigest()


def environment() -> dict[str, str]:
    """Describe the interpreter and the declared dependencies present now.

    Returns:
        A mapping of component name to version. A dependency that is not
        installed is reported as absent rather than omitted, so a reader can
        tell the difference between "not installed" and "not recorded".
    """
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": sys.implementation.name,
    }
    for name in _TRACKED:
        try:
            env[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            env[name] = "absent"
    return env


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to trace a result back to what produced it."""

    seed: int
    config_sha256: str
    artifacts: dict[str, str | None]
    environment: dict[str, str]
    started_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    @classmethod
    def create(
        cls,
        seed: int,
        config: dict[str, Any],
        artifact_names: list[str],
        manifest: Manifest,
    ) -> RunRecord:
        """Build a record, refusing inputs the manifest cannot identify.

        Args:
            seed: The seed the run was executed with.
            config: The run configuration.
            artifact_names: Names of the corpus artifacts consumed.
            manifest: The manifest those names must appear in.

        Returns:
            The run record.

        Raises:
            UnknownArtifactError: If any name is absent from the manifest. An
                unidentifiable input makes the whole record untrustworthy, so
                this fails before anything is written.
        """
        unknown = [name for name in artifact_names if name not in manifest]
        if unknown:
            raise UnknownArtifactError(
                f"not described by the manifest: {', '.join(sorted(unknown))}"
            )
        return cls(
            seed=seed,
            config_sha256=config_digest(config),
            artifacts={name: manifest[name].sha256 for name in artifact_names},
            environment=environment(),
        )

    def write(self, path: Path) -> None:
        """Write the record as JSON.

        Args:
            path: Destination file.
        """
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
