"""Reliability contracts for buyer-packet source commit discovery."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_packet_builder():
    """Load the buyer-packet builder as a standalone script module."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_buyer_packet.py"
    spec = importlib.util.spec_from_file_location("build_buyer_packet_timeout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_timeout_is_bounded_and_fails_closed(monkeypatch, tmp_path):
    """A hung local Git lookup must fail promptly with a stable package error."""
    module = _load_packet_builder()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "HEAD"],
            timeout=kwargs.get("timeout", 999),
            output="BUYER_PACKET_TIMEOUT_STDOUT_SECRET",
            stderr="BUYER_PACKET_TIMEOUT_STDERR_SECRET",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup timed out$"):
        module._source_commit(tmp_path)

    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


@pytest.mark.parametrize(
    "expected",
    [
        "0123456789abcdef0123456789abcdef01234567",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ],
)
def test_source_commit_forwards_bounded_deadline_on_success(
    monkeypatch, tmp_path, expected: str
):
    """Canonical full Git identities use the same package-owned deadline."""
    module = _load_packet_builder()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=f"{expected}\n"
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._source_commit(tmp_path) == expected
    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


def test_source_commit_rejects_non_timeout_git_failure(monkeypatch, tmp_path):
    """A non-zero Git exit must not degrade buyer provenance to ``unknown``."""
    module = _load_packet_builder()

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


def test_source_commit_rejects_missing_git_executable(monkeypatch, tmp_path):
    """An unavailable Git executable must block buyer-packet provenance."""
    module = _load_packet_builder()

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

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
    module = _load_packet_builder()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup returned invalid identity$"):
        module._source_commit(tmp_path)
