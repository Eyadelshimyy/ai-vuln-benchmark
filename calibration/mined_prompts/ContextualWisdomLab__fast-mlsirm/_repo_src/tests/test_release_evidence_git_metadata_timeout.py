"""Fail-first reliability contracts for release evidence Git metadata reads."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_release_index():
    """Load the release evidence index builder for boundary tests."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_release_evidence_index.py"
    spec = importlib.util.spec_from_file_location("build_release_evidence_index", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_bounds_git_metadata_lookup(monkeypatch, tmp_path: Path) -> None:
    """A hung ``git rev-parse`` must fail closed under a package-owned deadline."""
    module = _load_release_index()
    observed_timeouts: list[object] = []

    def timeout_run(*args, **kwargs):
        observed_timeouts.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(
            cmd=args[0] if args else kwargs.get("args", ["git", "rev-parse", "HEAD"]),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(module.subprocess, "run", timeout_run)

    with pytest.raises(RuntimeError, match="source commit lookup timed out"):
        module._source_commit(tmp_path)

    assert observed_timeouts == [module.GIT_METADATA_TIMEOUT_SECONDS]


def test_source_commit_rejects_git_command_failure(monkeypatch, tmp_path: Path) -> None:
    """A non-zero Git lookup cannot degrade release provenance to ``unknown``."""
    module = _load_release_index()

    def failed_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=args[0] if args else kwargs.get("args", ["git", "rev-parse", "HEAD"]),
        )

    monkeypatch.setattr(module.subprocess, "run", failed_run)

    with pytest.raises(RuntimeError, match="source commit lookup failed"):
        module._source_commit(tmp_path)


def test_source_commit_rejects_missing_git_executable(monkeypatch, tmp_path: Path) -> None:
    """An unavailable Git executable must block release-evidence provenance."""
    module = _load_release_index()

    def missing_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", missing_run)

    with pytest.raises(RuntimeError, match="source commit lookup failed"):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "unknown",
        "a" * 39,
        "a" * 41,
        "a" * 63,
        "a" * 65,
        "g" * 40,
        "g" * 64,
        "A" * 40,
        "A" * 64,
    ],
)
def test_source_commit_rejects_noncanonical_identity(
    monkeypatch, tmp_path: Path, stdout: str
) -> None:
    """Only a full canonical hexadecimal Git SHA may identify release source."""
    module = _load_release_index()

    def malformed_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else kwargs.get("args", []),
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", malformed_run)

    with pytest.raises(RuntimeError, match="source commit lookup returned invalid identity"):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    "expected",
    [
        "0123456789abcdef0123456789abcdef01234567",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ],
)
def test_source_commit_accepts_canonical_full_sha(
    monkeypatch, tmp_path: Path, expected: str
) -> None:
    """Canonical full SHA-1 and SHA-256 identities remain valid provenance."""
    module = _load_release_index()

    def valid_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else kwargs.get("args", []),
            returncode=0,
            stdout=f"{expected}\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", valid_run)

    assert module._source_commit(tmp_path) == expected
