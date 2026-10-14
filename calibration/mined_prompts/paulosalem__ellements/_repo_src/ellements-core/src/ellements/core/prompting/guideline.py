"""Guideline library.

A *guideline* is a short normative text that is prefixed to the user
query (or system prompt) to constrain the LLM's behavior, e.g. style
rules, tone, structural requirements.

Guidelines can be authored as Markdown (preferred — prose) or JSON
(``{"name": "...", "text": "..."}``). Both extensions are scanned from
the configured folder. Each file's stem becomes its logical ID. Id
collisions across files raise immediately at construction or
:meth:`add_from_path`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..exceptions import GuidelineNotFoundError

_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".md", ".json"})


class GuidelineLibrary:
    """Loads guideline files and exposes ID-based access.

    Args:
        folder: Directory containing guideline ``.md`` or ``.json`` files.

    Raises:
        FileNotFoundError: If *folder* does not exist.
        ValueError: If two files in *folder* resolve to the same logical ID.
    """

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Guideline folder not found: {folder}")

        self._paths_by_id: dict[str, Path] = {}
        self.refresh()

    # ── Discovery ────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rescan the folder, replacing any previously loaded paths."""
        self._paths_by_id.clear()
        for path in sorted(self.folder.iterdir()):
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES:
                self._register(path)

    def add_from_path(self, path: Path | str) -> str:
        """Register an additional guideline file outside the library folder.

        Returns:
            The logical ID under which the guideline was registered.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file's extension is unsupported or its ID
                collides with an existing entry.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Guideline file not found: {file_path}")
        if file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported guideline file extension: {file_path.suffix}. "
                f"Expected one of {sorted(_SUPPORTED_SUFFIXES)}."
            )
        return self._register(file_path)

    def _register(self, path: Path) -> str:
        guideline_id = path.stem
        existing = self._paths_by_id.get(guideline_id)
        if existing is not None and existing != path:
            raise ValueError(
                f"Guideline ID collision: {guideline_id!r} is registered by both "
                f"{existing} and {path}. Rename one of the files."
            )
        self._paths_by_id[guideline_id] = path
        return guideline_id

    # ── Querying ─────────────────────────────────────────────────────

    def list_ids(self) -> list[str]:
        """Return all known guideline IDs in stable sorted order."""
        return sorted(self._paths_by_id)

    def __contains__(self, guideline_id: object) -> bool:
        return isinstance(guideline_id, str) and guideline_id in self._paths_by_id

    # ── Loading ──────────────────────────────────────────────────────

    def load(self, guideline_id: str) -> str:
        """Load a guideline by ID and return its text content.

        Raises:
            GuidelineNotFoundError: If *guideline_id* is not registered.
        """
        if guideline_id not in self._paths_by_id:
            available = ", ".join(self.list_ids()) or "(none)"
            raise GuidelineNotFoundError(
                f"Guideline {guideline_id!r} not found. Available: {available}."
            )
        return self._render_path(self._paths_by_id[guideline_id])

    @classmethod
    def format_guideline_text(cls, text: str) -> str:
        """Normalize raw guideline text supplied directly by the caller."""
        return str(text).strip()

    @classmethod
    def _render_path(cls, path: Path) -> str:
        suffix = path.suffix.lower()
        raw = path.read_text(encoding="utf-8")
        if suffix == ".md":
            return cls.format_guideline_text(raw)
        if suffix == ".json":
            payload: Any = json.loads(raw)
            if not isinstance(payload, dict) or "text" not in payload:
                raise ValueError(
                    f"Guideline JSON file must contain a 'text' field "
                    f"(file: {path})."
                )
            return cls.format_guideline_text(str(payload["text"]))
        raise ValueError(
            f"Unsupported guideline file extension: {suffix} (file: {path})"
        )


__all__ = ["GuidelineLibrary"]
