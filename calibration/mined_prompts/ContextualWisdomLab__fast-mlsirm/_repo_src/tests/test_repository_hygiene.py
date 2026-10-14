"""Repository-hygiene regression tests for merge-conflict artifacts."""

from __future__ import annotations

from pathlib import Path
import subprocess


def _tracked_paths(repository_root: Path) -> tuple[str, ...]:
    """Return repository-tracked paths without scanning generated build trees."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return tuple(
        path.decode("utf-8")
        for path in completed.stdout.split(b"\0")
        if path
    )


def test_merge_conflict_backup_artifacts_are_absent_and_ignored():
    """Tracked source excludes ``.orig``/``.rej`` files and ignores recurrence."""
    repository_root = Path(__file__).resolve().parents[1]
    artifacts = sorted(
        path
        for path in _tracked_paths(repository_root)
        if Path(path).suffix in {".orig", ".rej"}
    )
    assert artifacts == []

    ignore_lines = {
        line.strip()
        for line in (repository_root / ".gitignore")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert "*.orig" in ignore_lines
    assert "*.rej" in ignore_lines
