"""
Centralized path utilities.

All user-facing and persisted paths must be repository-relative.
Never expose absolute host filesystem paths in API responses, findings, or stored records.
"""

from __future__ import annotations

from pathlib import Path


def to_relative_path(absolute: Path, repo_root: Path) -> str:
    """Return a POSIX-style relative path string, validated within repo_root.

    Raises ValueError if absolute is not inside repo_root.
    """
    try:
        rel = absolute.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except ValueError:
        raise ValueError(f"Path {absolute} is not inside repository root {repo_root}") from None


def is_within_repo(path: Path, repo_root: Path) -> bool:
    """Return True if path resolves to a location inside repo_root."""
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def safe_join(repo_root: Path, relative: str) -> Path:
    """Join a repository-relative path string to the root, rejecting traversal.

    Raises ValueError if the resolved path escapes repo_root.
    """
    joined = (repo_root / relative).resolve()
    if not is_within_repo(joined, repo_root):
        raise ValueError(f"Path traversal detected: {relative!r}")
    return joined
