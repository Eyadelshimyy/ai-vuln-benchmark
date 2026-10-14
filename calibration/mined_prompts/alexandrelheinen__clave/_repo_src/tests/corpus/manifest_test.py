"""Tests for the corpus manifest.

Each test names the acceptance criterion it guards, so the mapping from
requirement to proof is greppable in both directions.
"""

from pathlib import Path

import pytest

from clave.corpus.manifest import Manifest
from clave.errors import ManifestError


def write(tmp_path: Path, body: str) -> Path:
    """Write a manifest body to a temporary file and return its path."""

    path = tmp_path / "manifest.toml"
    path.write_text(body)
    return path


def test_valid_manifest_round_trips_every_artifact(tmp_path: Path) -> None:
    """Name, source and digest are read back."""
    path = write(
        tmp_path,
        """
        [[artifact]]
        name = "smoke"
        source = "https://example.invalid/smoke.csv"
        sha256 = "abc123"
        """,
    )
    manifest = Manifest.load(path)
    entry = manifest["smoke"]
    assert entry.source == "https://example.invalid/smoke.csv"
    assert entry.sha256 == "abc123"


def test_entry_without_digest_reports_unverified(tmp_path: Path) -> None:
    """An absent digest is a state, not a convention."""
    path = write(
        tmp_path,
        """
        [[artifact]]
        name = "spectralwaste"
        source = "https://example.invalid/sw.zip"
        """,
    )
    assert Manifest.load(path)["spectralwaste"].is_unverified


def test_duplicate_artifact_name_is_rejected_by_name(tmp_path: Path) -> None:
    """The error names the offending entry."""
    path = write(
        tmp_path,
        """
        [[artifact]]
        name = "twice"
        source = "https://example.invalid/a"

        [[artifact]]
        name = "twice"
        source = "https://example.invalid/b"
        """,
    )
    with pytest.raises(ManifestError, match="twice"):
        Manifest.load(path)


def test_entry_missing_a_required_field_is_rejected_by_name(tmp_path: Path) -> None:
    """A malformed entry is named, not silently skipped."""
    path = write(
        tmp_path,
        """
        [[artifact]]
        name = "no-source"
        """,
    )
    with pytest.raises(ManifestError, match="no-source"):
        Manifest.load(path)


def test_unknown_artifact_lookup_raises(tmp_path: Path) -> None:
    """A run record relies on the manifest refusing an unknown name."""
    path = write(
        tmp_path,
        """
        [[artifact]]
        name = "known"
        source = "https://example.invalid/k"
        """,
    )
    with pytest.raises(KeyError):
        Manifest.load(path)["missing"]
