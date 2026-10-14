"""Lock source inspection to the project's approved repository root."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

from app.core.config import get_settings
from app.security_testing.errors import RestrictedActivityError


def resolve_repo_root(
    *,
    project_repository_path: str | None,
    client_repo_root: str | None,
    mode: str,
    project_id: str,
) -> str:
    """Return a canonical directory the session may read.

    Persisted project paths win. Client-supplied roots cannot escape the
    project or the configured lab allow-list.
    """
    if project_repository_path:
        root = _canonical_directory(project_repository_path)
        if client_repo_root:
            requested = _canonical_path(client_repo_root)
            if requested != root and not _is_relative_to(requested, root):
                raise RestrictedActivityError("repo_root_not_owned_by_project")
        return str(root)
    if mode == "lab":
        candidate = client_repo_root or "."
        root = _canonical_directory(candidate)
        if not _lab_root_allowed(root):
            raise RestrictedActivityError("lab_repo_root_not_allowed")
        return str(root)
    # Live sessions may exist without a cloned repository. Source inspection
    # stays denied until a project-bound root is configured.
    if client_repo_root:
        raise RestrictedActivityError("repo_root_requires_project")
    return ""


def safe_source_path(root: str | Path, relative: str) -> Path:
    """Resolve a repo-relative path, rejecting traversal and outside symlinks."""
    if not str(root).strip():
        raise RestrictedActivityError("repo_root_required")
    base = _canonical_directory(str(root))
    cleaned = unicodedata.normalize("NFC", relative or "")
    cleaned = cleaned.replace("\\", "/")
    cleaned = cleaned.replace("\x00", "")
    while cleaned.startswith("/"):
        cleaned = cleaned[1:]
    if not cleaned or cleaned in {".", ""}:
        raise RestrictedActivityError("path_required")
    parts: list[str] = []
    for part in cleaned.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise RestrictedActivityError("path_traversal")
        parts.append(part)
    candidate = (base.joinpath(*parts)).resolve()
    if not _is_relative_to(candidate, base):
        raise RestrictedActivityError("path_traversal")
    real = Path(os.path.realpath(candidate))
    if not _is_relative_to(real, base):
        raise RestrictedActivityError("path_traversal")
    return candidate


def _lab_root_allowed(root: Path) -> bool:
    settings = get_settings()
    allowed = list(getattr(settings, "security_agent_lab_roots", ()) or ())
    allowed.extend([".", str(Path.cwd()), "/tmp"])
    for item in allowed:
        try:
            base = _canonical_path(item)
        except RestrictedActivityError:
            continue
        if root == base or _is_relative_to(root, base):
            return True
    return False


def _canonical_directory(raw: str) -> Path:
    path = _canonical_path(raw)
    if not path.exists() or not path.is_dir():
        raise RestrictedActivityError("repo_root_not_a_directory")
    return path


def _canonical_path(raw: str) -> Path:
    text = unicodedata.normalize("NFC", (raw or "").strip())
    if not text:
        raise RestrictedActivityError("repo_root_required")
    return Path(text).expanduser().resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
