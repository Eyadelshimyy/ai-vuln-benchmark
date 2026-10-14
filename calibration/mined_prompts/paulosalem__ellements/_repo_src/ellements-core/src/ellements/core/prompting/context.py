"""Shared persona and guideline selection state.

:class:`PromptContext` lets applications track the currently-active
persona and guideline and switch between three kinds of sources:

- by ID, against an attached :class:`PersonaLibrary` / :class:`GuidelineLibrary`
- by inline data (a dict for personas, a string for guidelines)
- by ad-hoc file path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .guideline import GuidelineLibrary
from .persona import PersonaLibrary


@dataclass
class PromptContext:
    """Active persona/guideline selection plus the libraries they came from."""

    persona_library: PersonaLibrary | None = None
    guideline_library: GuidelineLibrary | None = None
    _current_persona_id: str | None = field(default=None, init=False, repr=False)
    _current_guideline_id: str | None = field(default=None, init=False, repr=False)
    _custom_persona_text: str | None = field(default=None, init=False, repr=False)
    _custom_guideline_text: str | None = field(default=None, init=False, repr=False)

    # ── Persona ──────────────────────────────────────────────────────

    @property
    def current_persona_id(self) -> str | None:
        return self._current_persona_id

    def attach_persona_library(
        self,
        library: PersonaLibrary,
        *,
        default_persona: str | None = None,
        fallback_to_first: bool = True,
    ) -> None:
        """Attach a persona library and choose an initial active persona."""
        self.persona_library = library
        self._custom_persona_text = None
        if default_persona and default_persona in library:
            self._current_persona_id = default_persona
        elif fallback_to_first:
            self._current_persona_id = library.get_default_id()
        else:
            self._current_persona_id = None

    def set_persona_by_id(self, persona_id: str) -> None:
        """Activate a persona registered in the attached library."""
        if self.persona_library is None:
            raise ValueError("No persona library attached")
        if persona_id not in self.persona_library:
            available = ", ".join(self.persona_library.list_ids()) or "(none)"
            raise ValueError(
                f"Persona {persona_id!r} not found. Available: {available}."
            )
        self._current_persona_id = persona_id
        self._custom_persona_text = None

    def set_persona_from_data(self, persona_data: dict[str, Any]) -> None:
        """Activate a persona from in-memory data (no library required)."""
        self._custom_persona_text = PersonaLibrary.format_persona_data(persona_data)
        self._current_persona_id = None

    def set_persona_from_path(self, path: Path | str) -> None:
        """Activate a persona from an arbitrary file path.

        If a persona library is attached, the file is registered there
        first (raising on ID collision). Otherwise the file is loaded
        as a one-shot custom persona without affecting any library.
        """
        if self.persona_library is not None:
            persona_id = self.persona_library.add_from_path(path)
            self.set_persona_by_id(persona_id)
            return
        # No library — load standalone via a throwaway library on a tmp folder.
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Persona file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix not in {".json", ".md"}:
            raise ValueError(
                f"Unsupported persona file extension: {suffix} (file: {file_path})"
            )
        self._custom_persona_text = PersonaLibrary._render_path(file_path)  # noqa: SLF001
        self._current_persona_id = None

    def get_persona_text(self) -> str | None:
        """Return the active persona text, or ``None`` if no persona is active."""
        if self._custom_persona_text is not None:
            return self._custom_persona_text
        if self.persona_library is None or self._current_persona_id is None:
            return None
        return self.persona_library.load(self._current_persona_id)

    def clear_persona(self) -> None:
        """Deactivate any current persona."""
        self._current_persona_id = None
        self._custom_persona_text = None

    # ── Guideline ────────────────────────────────────────────────────

    @property
    def current_guideline_id(self) -> str | None:
        return self._current_guideline_id

    def attach_guideline_library(
        self,
        library: GuidelineLibrary,
        *,
        default_guideline: str | None = None,
    ) -> None:
        """Attach a guideline library and optionally select a default."""
        self.guideline_library = library
        self._custom_guideline_text = None
        if default_guideline and default_guideline in library:
            self._current_guideline_id = default_guideline
        else:
            self._current_guideline_id = None

    def set_guideline_by_id(self, guideline_id: str) -> None:
        """Activate a guideline registered in the attached library."""
        if self.guideline_library is None:
            raise ValueError("No guideline library attached")
        if guideline_id not in self.guideline_library:
            available = ", ".join(self.guideline_library.list_ids()) or "(none)"
            raise ValueError(
                f"Guideline {guideline_id!r} not found. Available: {available}."
            )
        self._current_guideline_id = guideline_id
        self._custom_guideline_text = None

    def set_guideline_from_text(self, text: str) -> None:
        """Activate a guideline from inline text (no library required)."""
        self._custom_guideline_text = GuidelineLibrary.format_guideline_text(text)
        self._current_guideline_id = None

    def set_guideline_from_path(self, path: Path | str) -> None:
        """Activate a guideline from an arbitrary file path."""
        if self.guideline_library is not None:
            guideline_id = self.guideline_library.add_from_path(path)
            self.set_guideline_by_id(guideline_id)
            return
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Guideline file not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix not in {".md", ".json"}:
            raise ValueError(
                f"Unsupported guideline file extension: {suffix} (file: {file_path})"
            )
        self._custom_guideline_text = GuidelineLibrary._render_path(file_path)  # noqa: SLF001
        self._current_guideline_id = None

    def get_guideline_text(self) -> str | None:
        """Return the active guideline text, or ``None`` if none active."""
        if self._custom_guideline_text is not None:
            return self._custom_guideline_text
        if self.guideline_library is None or self._current_guideline_id is None:
            return None
        return self.guideline_library.load(self._current_guideline_id)

    def clear_guideline(self) -> None:
        """Deactivate any current guideline."""
        self._current_guideline_id = None
        self._custom_guideline_text = None

    # ── Composition ──────────────────────────────────────────────────

    def apply_guideline(self, query: str) -> str:
        """Prefix the active guideline (if any) to *query*."""
        guideline_text = self.get_guideline_text()
        if not guideline_text:
            return query
        return f"{guideline_text}\n\n{query}"


__all__ = ["PromptContext"]
