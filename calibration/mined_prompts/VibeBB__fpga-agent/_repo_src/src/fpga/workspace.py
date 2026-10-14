"""Workspace path validation."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    return Path(os.environ.get("OPENHANDS_PROJECT_DIR") or Path.cwd()).resolve()


def workspace_path(value: str | Path, root: Path | None = None) -> Path:
    base = (root or workspace_root()).resolve()
    candidate = Path(value)
    raw = candidate if candidate.is_absolute() else base / candidate
    try:
        relative = raw.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path is outside the workspace: {value}") from exc
    current = base
    for part in relative.parts:
        if part == "..":
            current = current.parent
            if current != base and base not in current.parents:
                raise ValueError(f"path is outside the workspace: {value}")
        elif part not in ("", "."):
            current = current / part
        if current.is_symlink():
            raise ValueError(f"workspace path contains a symlink: {value}")
    resolved = raw.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"path is outside the workspace: {value}")
    return resolved
