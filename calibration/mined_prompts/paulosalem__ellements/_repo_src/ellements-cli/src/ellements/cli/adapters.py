"""Adapters between ellements core types and TUI provider Protocols.

The :mod:`ellements.cli.agent_tui` module defines ``PersonaProvider`` /
``GuidelineProvider`` Protocols so the TUI never imports an
``AgentController`` directly. This module supplies the canonical
adapters that bridge a :class:`~ellements.core.PersonaLibrary` /
:class:`~ellements.core.GuidelineLibrary` + controller pair to those
Protocols.

Apps that don't use ``AgentController`` can write their own adapters
and pass them to :class:`~ellements.cli.agent_tui.TuiConfig` directly.
"""

from __future__ import annotations

from typing import Any

from .agent_tui import GuidelineProvider, PersonaProvider


class _LibraryPersonaProvider:
    """Bridge a ``PersonaLibrary`` + controller to ``PersonaProvider``."""

    def __init__(self, library: Any, controller: Any) -> None:
        self._library = library
        self._controller = controller

    def list_ids(self) -> list[str]:
        return list(self._library.list_ids())

    def current_id(self) -> str | None:
        return getattr(self._controller, "current_persona_id", None)

    def select(self, persona_id: str) -> None:
        self._controller.set_persona_by_id(persona_id)


class _LibraryGuidelineProvider:
    """Bridge a ``GuidelineLibrary`` + controller to ``GuidelineProvider``."""

    def __init__(self, library: Any, controller: Any) -> None:
        self._library = library
        self._controller = controller

    def list_ids(self) -> list[str]:
        return list(self._library.list_ids())

    def current_id(self) -> str | None:
        return getattr(self._controller, "current_guideline_id", None)

    def select(self, guideline_id: str) -> None:
        self._controller.set_guideline_by_id(guideline_id)

    def clear(self) -> None:
        self._controller.clear_guideline()


def persona_provider_from_library(library: Any, controller: Any) -> PersonaProvider:
    """Adapt a :class:`PersonaLibrary` + controller pair to the TUI."""
    return _LibraryPersonaProvider(library, controller)


def guideline_provider_from_library(
    library: Any, controller: Any
) -> GuidelineProvider:
    """Adapt a :class:`GuidelineLibrary` + controller pair to the TUI."""
    return _LibraryGuidelineProvider(library, controller)


__all__ = [
    "guideline_provider_from_library",
    "persona_provider_from_library",
]
