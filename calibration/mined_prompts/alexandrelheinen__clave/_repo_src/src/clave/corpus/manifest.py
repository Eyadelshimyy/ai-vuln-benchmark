"""The committed corpus manifest.

Every corpus artifact CLAVE uses is described here by name, source, and
expected digest. No corpus binary is committed to the repository; the manifest
is what makes two runs comparable, because it says which bytes each one saw.

An entry whose digest has not been recorded is *unverified*. That is a state the
type carries rather than a convention a caller has to remember, so code cannot
treat an unrecorded digest as a passing one by accident.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from clave.errors import ManifestError

_REQUIRED = ("name", "source")


@dataclass(frozen=True)
class Artifact:
    """One corpus artifact described by the manifest.

    Attributes:
        name: Stable identifier used by run records to name an input.
        source: Where an operator fetches the artifact from.
        sha256: Expected digest, or None when it has never been recorded.
    """

    name: str
    source: str
    sha256: str | None = None

    @property
    def is_unverified(self) -> bool:
        """Whether this entry carries no recorded digest."""
        return self.sha256 is None


@dataclass(frozen=True)
class Manifest:
    """A parsed corpus manifest, keyed by artifact name."""

    artifacts: dict[str, Artifact]

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Read and validate a manifest file.

        Args:
            path: Path to the TOML manifest.

        Returns:
            The parsed manifest.

        Raises:
            ManifestError: If the file is not valid TOML, an entry is missing a
                required field, or an artifact name appears more than once. The
                message names the offending entry.
        """
        try:
            raw = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as exc:
            raise ManifestError(f"{path} is not valid TOML: {exc}") from exc

        artifacts: dict[str, Artifact] = {}
        for index, entry in enumerate(raw.get("artifact", [])):
            name = entry.get("name", f"<entry {index}>")
            missing = [field for field in _REQUIRED if field not in entry]
            if missing:
                raise ManifestError(
                    f"artifact {name!r} is missing {', '.join(missing)}"
                )
            if name in artifacts:
                raise ManifestError(f"artifact {name!r} is declared more than once")
            artifacts[name] = Artifact(
                name=name, source=entry["source"], sha256=entry.get("sha256")
            )
        return cls(artifacts=artifacts)

    def __getitem__(self, name: str) -> Artifact:
        """Look up an artifact by name.

        Raises:
            KeyError: If the manifest does not describe that artifact.
        """
        return self.artifacts[name]

    def __contains__(self, name: object) -> bool:
        """Whether the manifest describes an artifact of this name."""
        return name in self.artifacts
