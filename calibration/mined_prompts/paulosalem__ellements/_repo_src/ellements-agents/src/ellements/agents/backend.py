"""The :class:`AgentBackend` Protocol — the single integration point.

Any agent framework that satisfies this Protocol can be plugged into
:class:`~ellements.agents.AgentBuilder`, :class:`AgentController`, and
:func:`run_agent_with_progress` without further changes.

:class:`AgentEvent` — the uniform vocabulary backends use to surface
streaming events — lives in :mod:`ellements.core.observability.events`
alongside the other observability primitives. It is re-exported from
this module for backend authors who only deal with the agent surface.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from ellements.core import ToolRegistry
from ellements.core.observability import AgentEvent


@runtime_checkable
class AgentBackend(Protocol):
    """Adapts an agent runtime to ellements' uniform agent surface."""

    name: str
    """Stable backend identifier, e.g. ``"openai_agents"``."""

    def create_agent(
        self,
        *,
        name: str,
        instructions: str,
        tools: ToolRegistry,
        model: str,
    ) -> Any:
        """Construct a runtime-native agent instance.

        *tools* is the canonical :class:`~ellements.core.ToolRegistry`;
        each backend adapts the contained :class:`~ellements.core.ToolSpec`
        instances to its native tool type.
        """

    def create_session(
        self, session_id: str | None = None, **kwargs: Any
    ) -> Any | None:
        """Construct a runtime-native session object, or ``None``.

        Backends that don't support sessions return ``None``. Backends
        that support sessions but were given ``None`` for *session_id*
        should also return ``None`` (caller wants no session).

        Extra backend-specific options (e.g. ``storage_path`` for
        ``OpenAIAgentsBackend``) are passed via ``**kwargs``; backends
        ignore options they don't recognize.
        """

    async def run(
        self,
        agent: Any,
        task: str,
        *,
        max_turns: int = 10,
        session: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run *agent* on *task* and return its native result object.

        Implementations should propagate :class:`asyncio.CancelledError`
        unchanged.
        """

    def stream_run(
        self,
        agent: Any,
        task: str,
        *,
        max_turns: int = 10,
        session: Any | None = None,
        **kwargs: Any,
    ) -> AgentStream:
        """Run *agent* on *task* and return a streaming handle.

        Use :class:`AgentStream` as both an async iterator (over
        :class:`AgentEvent`) and a final-result holder accessible via
        :attr:`AgentStream.result` after iteration completes.
        """


class AgentStream(Protocol):
    """Combined async-iterator + final-result handle for streaming runs."""

    def __aiter__(self) -> AsyncIterator[AgentEvent]: ...

    @property
    def result(self) -> Any:
        """The native runtime result, populated once streaming completes."""


__all__ = ["AgentBackend", "AgentEvent", "AgentStream"]
