"""
Repair workspace — manages disposable copies of a repository for patch testing.

A patch is NEVER applied to the original repository.
Each candidate gets its own isolated workspace that is destroyed after use.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum repository size to copy into a workspace (bytes).
# Prevents runaway copies of giant repos.
_MAX_REPO_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB


class RepairWorkspace:
    """Disposable copy of a repository for safe patch application.

    Usage::
        async with RepairWorkspace.create(repo_path) as ws:
            ws.apply_patch(diff)
            result = await executor.execute(...)
    """

    def __init__(self, workspace_dir: str, repo_root: str) -> None:
        self._workspace_dir = Path(workspace_dir)
        self._repo_root = Path(repo_root)

    @classmethod
    def create(cls, repository_path: str) -> RepairWorkspace:
        """Copy the repository into a fresh temporary directory."""
        src = Path(repository_path).resolve()
        if not src.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {src}")

        # Rough size guard
        total = sum(
            f.stat().st_size for f in src.rglob("*") if f.is_file() and not _should_skip(f, src)
        )
        if total > _MAX_REPO_SIZE_BYTES:
            raise ValueError(
                f"Repository is too large to copy into a repair workspace ({total} bytes > {_MAX_REPO_SIZE_BYTES})"
            )

        tmpdir = tempfile.mkdtemp(prefix="bugforge_repair_")
        dest = Path(tmpdir) / "repo"
        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
        )
        logger.debug("Created repair workspace at %s (source: %s)", dest, src)
        return cls(workspace_dir=tmpdir, repo_root=str(dest))

    @property
    def repo_root(self) -> str:
        return str(self._repo_root)

    @property
    def workspace_dir(self) -> str:
        return str(self._workspace_dir)

    def apply_patch(self, patch_diff: str) -> tuple[bool, str]:
        """Apply a unified diff to the workspace copy.

        Returns (success, error_message).
        Does NOT modify the original repository.
        """
        if not patch_diff or not patch_diff.strip():
            return False, "Empty patch diff"

        # Try to apply via Python's built-in difflib (line-level only)
        try:
            return self._apply_unified_diff(patch_diff)
        except Exception as exc:
            logger.warning("Patch application failed: %s", exc)
            return False, str(exc)

    def _apply_unified_diff(self, diff: str) -> tuple[bool, str]:
        """Parse and apply a unified diff manually."""
        current_file: str | None = None
        original_lines: list[str] = []
        hunks: list[tuple[int, list[str]]] = []  # (original_start_0indexed, hunk_lines)
        hunk_lines: list[str] = []
        hunk_orig_start = 0

        for line in diff.splitlines(keepends=True):
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped.startswith("--- "):
                # Flush previous file
                if current_file is not None and hunks:
                    err = self._apply_hunks(current_file, original_lines, hunks)
                    if err:
                        return False, err
                current_file = None
                original_lines = []
                hunks = []
                hunk_lines = []
            elif stripped.startswith("+++ "):
                raw = stripped[4:].strip()
                if raw == "/dev/null":
                    current_file = None
                else:
                    # Strip leading b/ prefix
                    path = raw[2:] if raw.startswith("b/") else raw
                    current_file = path
                    target = self._repo_root / path
                    if target.exists():
                        original_lines = target.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines(keepends=True)
                    else:
                        original_lines = []
            elif stripped.startswith("@@ "):
                # Save previous hunk
                if hunk_lines:
                    hunks.append((hunk_orig_start, hunk_lines))
                    hunk_lines = []
                # Parse @@ -start,len +start,len @@
                m = re.search(r"@@ -(\d+)", stripped)
                hunk_orig_start = (int(m.group(1)) - 1) if m else 0
            elif current_file is not None:
                hunk_lines.append(stripped)

        # Flush last file
        if current_file is not None and (hunk_lines or hunks):
            if hunk_lines:
                hunks.append((hunk_orig_start, hunk_lines))
            err = self._apply_hunks(current_file, original_lines, hunks)
            if err:
                return False, err

        return True, ""

    def _apply_hunks(
        self,
        rel_path: str,
        original: list[str],
        hunks: list[tuple[int, list[str]]],
    ) -> str | None:
        """Apply hunks to a file. Return error string or None on success."""
        result = list(original)
        # Cumulative line-count delta after applying previous hunks
        offset = 0

        for orig_start, hunk in hunks:
            # Walk through all hunk lines tracking position sequentially.
            # context lines and removed lines each advance by one in the original;
            # added lines insert new content without advancing the original position.
            read_pos = orig_start + offset  # current cursor in result[]
            new_segment: list[str] = []  # replacement lines for this hunk
            orig_lines_consumed = 0  # how many result lines this hunk replaces

            for h in hunk:
                if h.startswith(" "):  # context line
                    ctx = h[1:].rstrip("\n")
                    actual = result[read_pos].rstrip("\n") if read_pos < len(result) else ""
                    if actual != ctx:
                        return (
                            f"Context mismatch in {rel_path} at line {orig_start + orig_lines_consumed + 1}: "
                            f"expected {ctx!r}, got {actual!r}; patch rejected"
                        )
                    new_segment.append(
                        result[read_pos]
                        if result[read_pos].endswith("\n")
                        else result[read_pos] + "\n"
                    )
                    read_pos += 1
                    orig_lines_consumed += 1
                elif h.startswith("-"):  # removed line
                    removed = h[1:].rstrip("\n")
                    actual = result[read_pos].rstrip("\n") if read_pos < len(result) else ""
                    if actual != removed:
                        return (
                            f"Hunk mismatch in {rel_path} at line {orig_start + orig_lines_consumed + 1}: "
                            f"expected {removed!r}, got {actual!r}; patch rejected"
                        )
                    # Don't add to new_segment — line is deleted
                    read_pos += 1
                    orig_lines_consumed += 1
                elif h.startswith("+"):  # added line
                    added = h[1:]
                    if not added.endswith("\n"):
                        added += "\n"
                    new_segment.append(added)

            hunk_start = orig_start + offset
            result[hunk_start : hunk_start + orig_lines_consumed] = new_segment
            offset += len(new_segment) - orig_lines_consumed

        target = (self._repo_root / rel_path).resolve()
        # Ensure resolved path stays inside workspace (prevent traversal)
        try:
            target.relative_to(self._repo_root.resolve())
        except ValueError:
            return f"Patch path '{rel_path}' escapes workspace root; patch rejected"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(result), encoding="utf-8")
        return None

    def destroy(self) -> None:
        """Remove the workspace directory."""
        try:
            shutil.rmtree(str(self._workspace_dir), ignore_errors=True)
            logger.debug("Destroyed repair workspace %s", self._workspace_dir)
        except Exception as exc:
            logger.warning("Failed to destroy repair workspace %s: %s", self._workspace_dir, exc)

    def __enter__(self) -> RepairWorkspace:
        return self

    def __exit__(self, *_: object) -> None:
        self.destroy()


def _should_skip(path: Path, base: Path) -> bool:
    """Return True for paths that should be excluded from copy size accounting."""
    try:
        rel = path.relative_to(base)
        parts = rel.parts
        return any(
            p in (".git", "__pycache__", "node_modules", ".venv", "venv", ".env") for p in parts
        )
    except ValueError:
        return False
