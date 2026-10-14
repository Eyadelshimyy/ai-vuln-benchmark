"""Tests for local artifact verification."""

from pathlib import Path

import pytest

from clave.corpus.artifacts import ArtifactStatus, record_digest, verify
from clave.corpus.manifest import Artifact

CONTENT = b"material,class\nPET bottle,M-01\n"
DIGEST = "b1c4e6f9c98a0e4b3d3d7e2f3c6a0a5f2f3b8e5c1c8b0e0d1a2f3c4b5a697887"


@pytest.fixture
def stored(tmp_path: Path) -> tuple[Path, str]:
    """Write the fixture bytes and return their path and true digest."""

    path = tmp_path / "smoke.csv"
    path.write_bytes(CONTENT)
    return path, record_digest(Artifact("smoke", "x"), path)


def test_matching_bytes_are_verified_and_available(stored: tuple[Path, str]) -> None:
    """A digest that matches makes the artifact available."""
    path, true_digest = stored
    result = verify(Artifact("smoke", "x", true_digest), path)
    assert result.status is ArtifactStatus.VERIFIED
    assert result.available


def test_mutated_bytes_report_mismatch_and_are_not_available(
    stored: tuple[Path, str],
) -> None:
    """One changed byte is a mismatch, not a warning."""
    path, true_digest = stored
    path.write_bytes(CONTENT.replace(b"M-01", b"M-02"))
    result = verify(Artifact("smoke", "x", true_digest), path)
    assert result.status is ArtifactStatus.DIGEST_MISMATCH
    assert not result.available


def test_unverified_entry_is_not_available(stored: tuple[Path, str]) -> None:
    """An unrecorded digest never passes as a recorded one."""
    path, _ = stored
    result = verify(Artifact("smoke", "x"), path)
    assert result.status is ArtifactStatus.UNVERIFIED_ENTRY
    assert not result.available
    assert result.observed_sha256 is not None


def test_absent_artifact_is_missing_and_not_available(tmp_path: Path) -> None:
    """A file that is not there is not available."""
    result = verify(Artifact("smoke", "x", DIGEST), tmp_path / "absent.csv")
    assert result.status is ArtifactStatus.MISSING
    assert not result.available


def test_recording_a_digest_returns_it_and_leaves_the_manifest_alone(
    tmp_path: Path, stored: tuple[Path, str]
) -> None:
    """Recording reports for review, it does not write."""
    path, true_digest = stored
    manifest = tmp_path / "manifest.toml"
    manifest.write_text('[[artifact]]\nname = "smoke"\nsource = "x"\n')
    before = manifest.read_text()
    assert record_digest(Artifact("smoke", "x"), path) == true_digest
    assert manifest.read_text() == before


def test_recording_a_digest_for_an_absent_file_raises(tmp_path: Path) -> None:
    """There is nothing to record when the bytes are absent."""
    with pytest.raises(FileNotFoundError, match="smoke"):
        record_digest(Artifact("smoke", "x"), tmp_path / "absent.csv")
