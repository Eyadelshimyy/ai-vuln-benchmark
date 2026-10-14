"""Fail-first reliability contracts for PR queue governance Git metadata reads."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_governance():
    """Load the PR queue governance script as a module for boundary tests."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_pr_queue_governance.py"
    spec = importlib.util.spec_from_file_location("build_pr_queue_governance", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_source_commit_bounds_git_metadata_lookup(monkeypatch, tmp_path: Path) -> None:
    """A hung ``git rev-parse`` must fail closed under a package-owned deadline."""
    module = _load_governance()
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


def test_source_commit_rejects_nonzero_git_exit(monkeypatch, tmp_path: Path) -> None:
    """A non-zero ``git rev-parse`` exit must raise instead of returning ``unknown``."""

    module = _load_governance()

    def failing_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=args[0] if args else ["git", "rev-parse", "HEAD"],
        )

    monkeypatch.setattr(module.subprocess, "run", failing_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


def test_source_commit_rejects_missing_git_executable(monkeypatch, tmp_path: Path) -> None:
    """Executable and OS-level subprocess failures must also fail closed."""

    module = _load_governance()

    def missing_run(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(module.subprocess, "run", missing_run)

    with pytest.raises(RuntimeError, match=r"^source commit lookup failed$"):
        module._source_commit(tmp_path)


def _succeeding_run(stdout: str):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0] if args else ["git", "rev-parse", "HEAD"],
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    return run


def test_source_commit_rejects_empty_stdout(monkeypatch, tmp_path: Path) -> None:
    """Empty Git stdout carries no reconstructable identity and must fail closed."""

    module = _load_governance()
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
def test_source_commit_rejects_noncanonical_identity(
    monkeypatch, tmp_path: Path, stdout: str, label: str
) -> None:
    """Only full lowercase hexadecimal object identities may reach evidence output."""

    module = _load_governance()
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
def test_source_commit_accepts_canonical_full_identity(
    monkeypatch, tmp_path: Path, candidate: str
) -> None:
    """Canonical lowercase SHA-1 and SHA-256 identities pass through unchanged."""

    module = _load_governance()
    monkeypatch.setattr(module.subprocess, "run", _succeeding_run(f"{candidate}\n"))

    assert module._source_commit(tmp_path) == candidate
