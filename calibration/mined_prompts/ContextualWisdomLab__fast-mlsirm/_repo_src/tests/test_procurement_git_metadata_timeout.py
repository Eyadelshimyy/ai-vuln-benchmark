"""Reliability contracts for procurement source commit discovery."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_procurement_builder():
    """Load the procurement due-diligence builder as a standalone module."""
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_procurement_due_diligence.py"
    )
    spec = importlib.util.spec_from_file_location("procurement_timeout", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_timeout_is_bounded_and_fails_closed(monkeypatch, tmp_path):
    """A hung local Git lookup must fail promptly with a stable package error."""
    module = _load_procurement_builder()
    seen: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "HEAD"],
            timeout=kwargs.get("timeout", 999),
            output="PROCUREMENT_TIMEOUT_STDOUT_SECRET",
            stderr="PROCUREMENT_TIMEOUT_STDERR_SECRET",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup timed out$"):
        module._source_commit(tmp_path)

    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


def test_source_commit_forwards_bounded_deadline_on_success(monkeypatch, tmp_path):
    """Successful Git metadata lookup uses the same package-owned deadline."""
    module = _load_procurement_builder()
    seen: dict[str, object] = {}
    candidate = "0123456789abcdef0123456789abcdef01234567"

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=f"{candidate}\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._source_commit(tmp_path) == candidate
    timeout = seen.get("timeout")
    assert isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
    assert 0 < timeout <= 30


def test_source_commit_rejects_nonzero_git_exit(monkeypatch, tmp_path):
    """A non-zero ``git rev-parse`` exit must raise instead of returning ``unknown``."""
    module = _load_procurement_builder()

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=args[0])

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


def test_source_commit_rejects_missing_git_executable(monkeypatch, tmp_path):
    """Executable and OS-level subprocess failures must also fail closed."""
    module = _load_procurement_builder()

    def fake_run(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


def _succeeding_run(stdout: str):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    return run


def test_source_commit_rejects_empty_stdout(monkeypatch, tmp_path):
    """Empty Git stdout carries no reconstructable identity and must fail closed."""
    module = _load_procurement_builder()
    monkeypatch.setattr(module.subprocess, "run", _succeeding_run("\n"))

    with pytest.raises(
        RuntimeError,
        match=r"^source commit is not a full lowercase SHA-1 or SHA-256 object id$",
    ):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    ("stdout", "label"),
    [
        ("a1b2c3d", "abbreviated sha"),
        ("A" * 40, "uppercase sha"),
        ("a" * 39, "undersized sha-1"),
        ("a" * 41, "oversized sha-1"),
        ("b" * 63, "undersized sha-256"),
        ("b" * 65, "oversized sha-256"),
        ("g" * 40, "non-hexadecimal identity"),
    ],
)
def test_source_commit_rejects_noncanonical_identity(monkeypatch, tmp_path, stdout, label):
    """Only full lowercase hexadecimal object identities may reach evidence output."""
    module = _load_procurement_builder()
    monkeypatch.setattr(module.subprocess, "run", _succeeding_run(f"{stdout}\n"))

    with pytest.raises(RuntimeError, match="not a full lowercase SHA-1 or SHA-256"):
        module._source_commit(tmp_path)


@pytest.mark.parametrize(
    "candidate",
    [
        "0123456789abcdef0123456789abcdef01234567",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ],
)
def test_source_commit_accepts_canonical_full_identity(monkeypatch, tmp_path, candidate):
    """Canonical lowercase SHA-1 and SHA-256 identities pass through unchanged."""
    module = _load_procurement_builder()
    monkeypatch.setattr(module.subprocess, "run", _succeeding_run(f"{candidate}\n"))

    assert module._source_commit(tmp_path) == candidate
