"""Local artifact verification.

A fetch that silently accepts changed bytes gives no guarantee at all, so
verification reports three distinct outcomes rather than a boolean plus a
message. Only :attr:`ArtifactStatus.VERIFIED` makes an artifact available.

Recording a digest deliberately does not write the manifest. A tool that
updates the file it checks against has verified nothing; the observed digest is
returned so a human can review it and commit it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from clave.corpus.manifest import Artifact

_CHUNK = 1 << 20


class ArtifactStatus(Enum):
    """Outcome of verifying a local artifact against its manifest entry."""

    VERIFIED = "verified"
    DIGEST_MISMATCH = "digest_mismatch"
    UNVERIFIED_ENTRY = "unverified_entry"
    MISSING = "missing"


@dataclass(frozen=True)
class Verification:
    """The result of checking one artifact.

    Attributes:
        artifact: The manifest entry that was checked.
        status: Which outcome applies.
        observed_sha256: Digest computed from local bytes, when they existed.
    """

    artifact: Artifact
    status: ArtifactStatus
    observed_sha256: str | None = None

    @property
    def available(self) -> bool:
        """Whether the artifact may be used.

        True only for a verified artifact. An unverified manifest entry and a
        digest mismatch are both unavailable, so neither can be trained on by
        accident.
        """
        return self.status is ArtifactStatus.VERIFIED


def digest_of(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path: File to read.

    Returns:
        The digest as a lowercase hexadecimal string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(artifact: Artifact, path: Path) -> Verification:
    """Check a local file against its manifest entry.

    Args:
        artifact: The manifest entry describing the expected bytes.
        path: Where the artifact is expected locally.

    Returns:
        A verification carrying the status and, when the file existed, the
        digest observed.
    """
    if not path.is_file():
        return Verification(artifact, ArtifactStatus.MISSING)
    observed = digest_of(path)
    if artifact.is_unverified:
        return Verification(artifact, ArtifactStatus.UNVERIFIED_ENTRY, observed)
    if observed != artifact.sha256:
        return Verification(artifact, ArtifactStatus.DIGEST_MISMATCH, observed)
    return Verification(artifact, ArtifactStatus.VERIFIED, observed)


def record_digest(artifact: Artifact, path: Path) -> str:
    """Compute the digest a manifest entry is missing, for review.

    This does not modify the manifest. The caller reviews the returned digest
    and commits it, which is what keeps the manifest a record of intent rather
    than a mirror of whatever happens to be on disk.

    Args:
        artifact: The unverified manifest entry.
        path: Where the artifact is stored locally.

    Returns:
        The observed digest, as a lowercase hexadecimal string.

    Raises:
        FileNotFoundError: If the artifact is not present locally.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{artifact.name!r} is not present at {path}")
    return digest_of(path)
