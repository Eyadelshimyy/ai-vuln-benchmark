"""Reliability contracts for Figma-evidence source commit discovery."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


VALID_SHA1 = "0123456789abcdef0123456789abcdef01234567"
VALID_SHA256 = (
    "0123456789abcdef0123456789abcdef"
    "0123456789abcdef0123456789abcdef"
)


def _load_figma_evidence_sync():
    """Load the Figma evidence sync builder as a standalone script module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_figma_evidence_sync.py"
    spec = importlib.util.spec_from_file_location("build_figma_evidence_sync_timeout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_timeout_is_bounded_and_fails_closed(monkeypatch, tmp_path):
    """A hung local Git lookup must fail promptly with a stable package error."""
    module = _load_figma_evidence_sync()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "HEAD"],
            timeout=kwargs.get("timeout", 999),
            output="FIGMA_EVIDENCE_TIMEOUT_STDOUT_SECRET",
            stderr="FIGMA_EVIDENCE_TIMEOUT_STDERR_SECRET",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup timed out$"):
        module._source_commit(tmp_path)

    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


@pytest.mark.parametrize("source_commit", [VALID_SHA1, VALID_SHA256])
def test_source_commit_accepts_canonical_full_object_ids(
    monkeypatch, tmp_path, source_commit
):
    """Canonical full SHA-1 and SHA-256 object identities remain supported."""
    module = _load_figma_evidence_sync()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=f"{source_commit}\n"
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._source_commit(tmp_path) == source_commit
    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "abc123\n",
        f"{'A' * 40}\n",
        f"{'g' * 40}\n",
        f"{'a' * 39}\n",
        f"{'a' * 41}\n",
        f"{'a' * 65}\n",
    ],
)
def test_source_commit_rejects_noncanonical_object_ids(monkeypatch, tmp_path, stdout):
    """Malformed, abbreviated, uppercase, and oversized object IDs fail closed."""
    module = _load_figma_evidence_sync()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError, match=r"^source commit lookup returned invalid object id$"
    ):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(
            returncode=128, cmd=["git", "rev-parse", "HEAD"]
        ),
        FileNotFoundError("git is unavailable"),
        OSError("git metadata unavailable"),
    ],
)
def test_source_commit_non_timeout_failures_fail_closed(monkeypatch, tmp_path, failure):
    """Git command and operating-system failures cannot degrade to unknown provenance."""
    module = _load_figma_evidence_sync()

    def fake_run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)
