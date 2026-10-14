"""Backend-agnostic agent controller for testable agent behavior.

Controllers encapsulate agent lifecycle, conversation state (via a
backend-provided session object), persona/guideline composition, and
statistics tracking — separately from any presentation layer (TUI,
HTTP, etc.). They operate purely through the :class:`AgentBackend`
Protocol, so the same controller works against any backend that
implements it.
"""

from __future__ import annotations

import inspect
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ellements.core import (
    GuidelineLibrary,
    PersonaLibrary,
    ToolRegistry,
)
from ellements.core.prompting import PromptContext

from .backend import AgentBackend
from .runner import AgentRunResult, AgentRunStats, run_agent_with_progress

ProgressCallback = Callable[[str], Awaitable[None] | None]
StatsCallback = Callable[[AgentRunStats], Awaitable[None] | None]


@dataclass
class ControllerConfig:
    """Static configuration for an :class:`AgentController`."""

    model: str
    max_turns: int = 200
    show_activity_log: bool = True
    persona_folder: Path | None = None
    guideline_folder: Path | None = None
    default_persona_id: str | None = None
    default_guideline_id: str | None = None
    session_storage_path: Path | None = None


class AgentController(ABC):
    """Base class encapsulating the run-loop and state for one agent.

    Subclasses implement :meth:`_build_agent` and
    :meth:`_get_fallback_instructions`. The controller can be shared
    across UIs (CLI, TUI, HTTP), tests, or batch jobs because it
    exposes no presentation concerns.
    """

    def __init__(
        self, backend: AgentBackend, config: ControllerConfig
    ) -> None:
        self.backend = backend
        self.config = config
        self.prompt_context = self._build_prompt_context(config)

        self._session = self._create_session()
        self._conversation_history: list[dict[str, Any]] = []
        self._session_stats = AgentRunStats()

    @staticmethod
    def _build_prompt_context(config: ControllerConfig) -> PromptContext:
        """Wire the configured persona / guideline folders into a fresh context.

        Extracted from ``__init__`` so the assembly is described once,
        in one place, instead of inline with the other initialization
        bookkeeping. Subclasses can override this hook if they want to
        substitute different libraries or pre-populate additional
        slots.
        """
        context = PromptContext()
        if config.persona_folder is not None:
            context.attach_persona_library(
                PersonaLibrary(config.persona_folder),
                default_persona=config.default_persona_id,
                fallback_to_first=True,
            )
        if config.guideline_folder is not None:
            context.attach_guideline_library(
                GuidelineLibrary(config.guideline_folder),
                default_guideline=config.default_guideline_id,
            )
        return context

    # ── Subclass hooks ─────────────────────────────────────────────

    @abstractmethod
    def _build_agent(self) -> Any:
        """Build and return the runtime-native agent instance."""

    @abstractmethod
    def _get_fallback_instructions(self) -> str:
        """Return fallback instructions used when :meth:`_build_agent` fails."""

    # ── Public surface ─────────────────────────────────────────────

    async def run(
        self,
        query: str,
        *,
        progress_callback: ProgressCallback | None = None,
        stats_callback: StatsCallback | None = None,
    ) -> AgentRunResult:
        """Run one query through the agent and capture stats."""
        try:
            agent = self._build_agent()
        except Exception as exc:  # noqa: BLE001
            if progress_callback is not None:
                _maybe_await = progress_callback(
                    f"⚠️  Build error: {exc}. Using fallback agent.\n"
                )
                if inspect.isawaitable(_maybe_await):
                    await _maybe_await
            agent = self._build_fallback_agent()

        prepared_query = self._prepare_query(query)
        result = await run_agent_with_progress(
            self.backend,
            agent,
            prepared_query,
            progress_callback=progress_callback,
            stats_callback=stats_callback,
            max_turns=self.config.max_turns,
            session=self._session,
            show_activity_log=self.config.show_activity_log,
        )

        self._conversation_history.append(
            {
                "query": query,
                "response": result.final_output,
                "stats": result.stats.to_dict(),
            }
        )
        self._session_stats.merge(result.stats)
        return result

    async def reset_conversation(self) -> None:
        """Clear the underlying session and reset accumulated stats."""
        clear = getattr(self._session, "clear_session", None)
        if clear is not None:
            clear_result = clear()
            if hasattr(clear_result, "__await__"):
                await clear_result
        self._conversation_history = []
        self._session_stats = AgentRunStats()

    # ── Persona / guideline pass-throughs ──────────────────────────

    def set_persona_by_id(self, persona_id: str) -> None:
        self.prompt_context.set_persona_by_id(persona_id)

    def set_persona_from_data(self, persona_data: dict[str, Any]) -> None:
        self.prompt_context.set_persona_from_data(persona_data)

    def set_persona_from_path(self, path: Path | str) -> None:
        self.prompt_context.set_persona_from_path(path)

    def set_guideline_by_id(self, guideline_id: str) -> None:
        self.prompt_context.set_guideline_by_id(guideline_id)

    def set_guideline_from_text(self, text: str) -> None:
        self.prompt_context.set_guideline_from_text(text)

    def set_guideline_from_path(self, path: Path | str) -> None:
        self.prompt_context.set_guideline_from_path(path)

    def clear_guideline(self) -> None:
        self.prompt_context.clear_guideline()

    @property
    def current_persona_id(self) -> str | None:
        return self.prompt_context.current_persona_id

    @property
    def current_guideline_id(self) -> str | None:
        return self.prompt_context.current_guideline_id

    @property
    def persona_library(self) -> PersonaLibrary | None:
        return self.prompt_context.persona_library

    @property
    def guideline_library(self) -> GuidelineLibrary | None:
        return self.prompt_context.guideline_library

    def get_current_persona_text(self) -> str | None:
        return self.prompt_context.get_persona_text()

    def get_current_guideline_text(self) -> str | None:
        return self.prompt_context.get_guideline_text()

    # ── Stats / history ────────────────────────────────────────────

    def get_session_stats(self) -> AgentRunStats:
        return self._session_stats

    def get_conversation_history(self) -> list[dict[str, Any]]:
        return list(self._conversation_history)

    def get_session(self) -> Any:
        return self._session

    # ── Internals ──────────────────────────────────────────────────

    def _prepare_query(self, query: str) -> str:
        return self.prompt_context.apply_guideline(query)

    def _build_fallback_agent(self) -> Any:
        return self.backend.create_agent(
            name=self.__class__.__name__,
            instructions=self._get_fallback_instructions(),
            tools=ToolRegistry(),
            model=self.config.model,
        )

    def _create_session(self) -> Any:
        """Create a per-controller session via the backend.

        The session shape is backend-specific; the controller never
        introspects it directly.
        """
        session_id = f"controller_{uuid.uuid4().hex}"
        return self.backend.create_session(
            session_id, storage_path=self.config.session_storage_path
        )


__all__ = ["AgentController", "ControllerConfig"]
