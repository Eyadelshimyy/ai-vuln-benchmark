"""Persona library.

A *persona* describes a character or role that the LLM should adopt.
Personas can be authored as JSON (rich structured form) or Markdown
(prose with optional YAML front-matter for metadata).

Both extensions are scanned from the configured folder. Each file's
stem becomes its logical ID. Id collisions across files raise
immediately at construction or :meth:`add_from_path` — by design.

JSON schema:
    ``{"persona": {"name": "...", "occupation": {...}, "preferences": {...}, ...}}``

Markdown schema (optional YAML front-matter):
    ``---``
    ``name: Alice``
    ``style: terse``
    ``---``
    ``Long-form persona description in prose…``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..exceptions import PersonaNotFoundError

_SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".json", ".md"})
_FRONT_MATTER_RE = re.compile(
    r"\A---\s*\n(?P<front>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


class PersonaLibrary:
    """Loads persona files and exposes ID-based access.

    Args:
        folder: Directory containing persona ``.json`` or ``.md`` files.

    Raises:
        FileNotFoundError: If *folder* does not exist.
        ValueError: If two files in *folder* resolve to the same logical ID.
    """

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Persona folder not found: {folder}")

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
        """Register an additional persona file outside the library folder.

        Returns:
            The logical ID under which the persona was registered.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file's extension is unsupported or its ID
                collides with an existing entry.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Persona file not found: {file_path}")
        if file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported persona file extension: {file_path.suffix}. "
                f"Expected one of {sorted(_SUPPORTED_SUFFIXES)}."
            )
        return self._register(file_path)

    def _register(self, path: Path) -> str:
        persona_id = path.stem
        existing = self._paths_by_id.get(persona_id)
        if existing is not None and existing != path:
            raise ValueError(
                f"Persona ID collision: {persona_id!r} is registered by both "
                f"{existing} and {path}. Rename one of the files."
            )
        self._paths_by_id[persona_id] = path
        return persona_id

    # ── Querying ─────────────────────────────────────────────────────

    def list_ids(self) -> list[str]:
        """Return all known persona IDs in stable sorted order."""
        return sorted(self._paths_by_id)

    def __contains__(self, persona_id: object) -> bool:
        return isinstance(persona_id, str) and persona_id in self._paths_by_id

    # ── Loading ──────────────────────────────────────────────────────

    def load(self, persona_id: str) -> str:
        """Load a persona by ID and return its formatted text representation.

        Raises:
            PersonaNotFoundError: If *persona_id* is not registered.
        """
        if persona_id not in self._paths_by_id:
            available = ", ".join(self.list_ids()) or "(none)"
            raise PersonaNotFoundError(
                f"Persona {persona_id!r} not found. Available: {available}."
            )
        return self._render_path(self._paths_by_id[persona_id])

    def get_default_id(self) -> str | None:
        """Return the first persona ID alphabetically, or ``None`` if empty."""
        ids = self.list_ids()
        return ids[0] if ids else None

    # ── Rendering ────────────────────────────────────────────────────

    @classmethod
    def format_persona_data(cls, persona_data: dict[str, Any]) -> str:
        """Format a persona JSON document as readable text.

        The output mirrors the typical "persona" block: ``name`` and
        ``occupation`` first, followed by goals, preferences,
        personality traits, beliefs, and finally communication style.
        Each section is rendered only when present, so partial
        documents stay clean.
        """
        persona = (
            persona_data.get("persona") if isinstance(persona_data, dict) else None
        )
        if not isinstance(persona, dict):
            return "No persona information available"

        sections: list[str] = []
        sections.extend(_render_identity(persona))
        sections.extend(_render_bullet_block(persona, "long_term_goals", "Long-term Goals"))

        preferences = persona.get("preferences")
        if isinstance(preferences, dict):
            for label, key in _PREFERENCE_SECTIONS:
                sections.extend(
                    _render_bullet_block(preferences, key, label, limit=10)
                )

        personality = persona.get("personality")
        if isinstance(personality, dict):
            sections.extend(
                _render_bullet_block(personality, "traits", "Personality Traits", limit=10)
            )

        sections.extend(_render_bullet_block(persona, "beliefs", "Beliefs"))

        if "style" in persona:
            sections.append(f"\nCommunication Style: {persona['style']}")

        return "\n".join(sections) if sections else "No persona information available"

    @classmethod
    def format_persona_markdown(
        cls, body: str, front_matter: dict[str, Any] | None
    ) -> str:
        """Combine optional front-matter metadata with the Markdown body."""
        front_matter = front_matter or {}
        header_lines: list[str] = []
        if "name" in front_matter:
            header_lines.append(f"Name: {front_matter['name']}")
        if "style" in front_matter:
            header_lines.append(f"Communication Style: {front_matter['style']}")
        for key, value in front_matter.items():
            if key in {"name", "style"}:
                continue
            header_lines.append(f"{key}: {value}")

        body_text = body.strip()
        if not header_lines:
            return body_text
        return "\n".join(header_lines) + "\n\n" + body_text

    @classmethod
    def _render_path(cls, path: Path) -> str:
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            return cls.format_persona_data(json.loads(text))
        if suffix == ".md":
            front_matter, body = _split_front_matter(text)
            return cls.format_persona_markdown(body, front_matter)
        raise ValueError(
            f"Unsupported persona file extension: {suffix} (file: {path})"
        )


def _split_front_matter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse an optional YAML front-matter block at the head of *text*."""
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return None, text
    front_raw = match.group("front")
    body = match.group("body")
    front = yaml.safe_load(front_raw) or {}
    if not isinstance(front, dict):
        raise ValueError(
            "Persona front-matter must parse to a YAML mapping; "
            f"got {type(front).__name__}."
        )
    return front, body


# ── Persona JSON rendering helpers ────────────────────────────────────
# A handful of pure helpers keep ``PersonaLibrary.format_persona_data``
# small and data-driven. Adding a new persona field becomes a one-line
# entry in either the call list or ``_PREFERENCE_SECTIONS``.

_PREFERENCE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Interests", "interests"),
    ("Likes", "likes"),
    ("Dislikes", "dislikes"),
)


def _render_identity(persona: dict[str, Any]) -> list[str]:
    """Render ``name`` plus the optional ``occupation`` sub-block."""
    lines: list[str] = []
    if "name" in persona:
        lines.append(f"Name: {persona['name']}")
    occupation = persona.get("occupation")
    if isinstance(occupation, dict):
        if "title" in occupation:
            lines.append(f"Occupation: {occupation['title']}")
        if "description" in occupation:
            lines.append(f"Description: {occupation['description']}")
    return lines


def _render_bullet_block(
    source: dict[str, Any],
    key: str,
    label: str,
    *,
    limit: int | None = None,
) -> list[str]:
    """Render ``source[key]`` as a labeled bullet list, if it's a list.

    Returns ``[]`` when the key is missing or holds a non-list value,
    so callers can extend unconditionally and skip empty sections.
    """
    values = source.get(key)
    if not isinstance(values, list) or not values:
        return []
    items = values[:limit] if limit is not None else values
    return [f"\n{label}:", *(f"  - {item}" for item in items)]


__all__ = ["PersonaLibrary"]
