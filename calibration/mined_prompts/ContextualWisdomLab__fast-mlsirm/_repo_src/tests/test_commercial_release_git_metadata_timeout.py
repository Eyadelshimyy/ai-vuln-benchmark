"""Reliability contracts for commercial-release source commit discovery."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_commercial_release_builder():
    """Load the commercial-release builder as a standalone script module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_commercial_release.py"
    spec = importlib.util.spec_from_file_location("build_commercial_release_timeout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_timeout_is_bounded_and_fails_closed(monkeypatch, tmp_path):
    """A hung local Git lookup must fail promptly with a stable package error."""
    module = _load_commercial_release_builder()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "HEAD"],
            timeout=kwargs.get("timeout", 999),
            output="COMMERCIAL_RELEASE_TIMEOUT_STDOUT_SECRET",
            stderr="COMMERCIAL_RELEASE_TIMEOUT_STDERR_SECRET",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup timed out$"):
        module._source_commit(tmp_path)

    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


def test_source_commit_rejects_git_command_failure(monkeypatch, tmp_path):
    """A non-zero Git exit must not degrade commercial provenance to ``unknown``."""
    module = _load_commercial_release_builder()

    def failed_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(module.subprocess, "run", failed_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


def test_source_commit_rejects_missing_git_executable(monkeypatch, tmp_path):
    """An unavailable Git executable must block commercial release provenance."""
    module = _load_commercial_release_builder()

    def missing_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", missing_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "unknown",
        "abc123",
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
def test_source_commit_rejects_noncanonical_identity(monkeypatch, tmp_path, stdout):
    """Abbreviated or malformed Git identities must fail closed."""
    module = _load_commercial_release_builder()

    def malformed_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout)

    monkeypatch.setattr(module.subprocess, "run", malformed_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup returned invalid identity$"):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    "expected",
    [
        "0123456789abcdef0123456789abcdef01234567",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ],
)
def test_source_commit_accepts_canonical_full_identity(monkeypatch, tmp_path, expected: str):
    """Canonical full SHA-1 and SHA-256 identities remain valid provenance."""
    module = _load_commercial_release_builder()
    seen: dict[str, object] = {}

    def valid_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=f"{expected}\n", stderr=""
        )

    monkeypatch.setattr(module.subprocess, "run", valid_run)

    assert module._source_commit(tmp_path) == expected
    assert seen["timeout"] == module.GIT_METADATA_TIMEOUT_SECONDS
