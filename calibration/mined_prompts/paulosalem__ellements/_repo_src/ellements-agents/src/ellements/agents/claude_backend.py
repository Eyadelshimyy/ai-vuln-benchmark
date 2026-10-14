"""Anthropic Claude Agents backend implementing :class:`AgentBackend`.

This backend wraps the ``claude_agents`` (or ``anthropic-agents``) SDK
when installed. The shape mirrors :class:`OpenAIAgentsBackend` so
higher-level code never branches on backend identity.

If the Anthropic SDK is not installed, instantiating
:class:`ClaudeAgentsBackend` raises :class:`ImportError` — the same
contract as :class:`OpenAIAgentsBackend`.

Note:
    The Claude Agents SDK is still evolving; this adapter targets the
    interface published as of writing (``claude_agents.Agent`` +
    ``claude_agents.Runner`` with ``run``/``run_streamed``). When that
    interface changes, only this file needs to follow.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ellements.core import ToolRegistry, ToolSpec
from ellements.core.tools import bind_json_invoker

from .backend import AgentEvent


class ClaudeAgentsBackend:
    """Backend that adapts Anthropic's Claude Agents SDK to ellements."""

    name: str = "claude_agents"

    def __init__(self) -> None:
        Agent, Runner = _load_sdk()
        self._Agent = Agent
        self._Runner = Runner

    # ── Agent construction ─────────────────────────────────────────

    def create_agent(
        self,
        *,
        name: str,
        instructions: str,
        tools: ToolRegistry,
        model: str,
    ) -> Any:
        return self._Agent(
            name=name,
            instructions=instructions,
            tools=[_adapt_spec_for_claude(spec) for spec in tools.values()],
            model=model,
        )

    # ── Session construction ───────────────────────────────────────

    def create_session(
        self, session_id: str | None = None, **kwargs: Any
    ) -> Any | None:
        """Claude Agents has no native session shape; always returns ``None``."""
        return None

    # ── Run (non-streaming) ────────────────────────────────────────

    async def run(
        self,
        agent: Any,
        task: str,
        *,
        max_turns: int = 10,
        session: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._Runner.run(
            agent, task, max_turns=max_turns, session=session, **kwargs
        )

    # ── Run (streaming) ────────────────────────────────────────────

    def stream_run(
        self,
        agent: Any,
        task: str,
        *,
        max_turns: int = 10,
        session: Any | None = None,
        **kwargs: Any,
    ) -> _ClaudeAgentStream:
        streaming_result = self._Runner.run_streamed(
            agent, task, max_turns=max_turns, session=session, **kwargs
        )
        return _ClaudeAgentStream(streaming_result)


def _load_sdk() -> tuple[Any, Any]:
    try:
        from claude_agents import Agent, Runner
    except ImportError as exc:
        raise ImportError(
            "Claude Agents SDK not installed. "
            "Install the appropriate Anthropic agents package "
            "(currently 'claude_agents')."
        ) from exc
    return Agent, Runner


class _ClaudeAgentStream:
    """:class:`AgentStream` impl translating Claude SDK events to AgentEvent.

    Translation rules mirror :class:`_OpenAIAgentStream` so consumers
    receive the same uniform event vocabulary.
    """

    def __init__(self, streaming_result: Any) -> None:
        self._streaming_result = streaming_result

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[AgentEvent]:
        async for raw in self._streaming_result.stream_events():
            event = _translate_event(raw)
            if event is not None:
                yield event

    @property
    def result(self) -> Any:
        return self._streaming_result


def _translate_event(raw: Any) -> AgentEvent | None:
    event_type = getattr(raw, "type", None) or getattr(raw, "kind", None)
    if event_type in {None, "raw", "raw_response"}:
        return None
    if event_type in {"agent_active", "agent_updated"}:
        return AgentEvent(
            type="agent_active",
            payload={"name": getattr(raw, "name", "Agent")},
        )
    if event_type in {"tool_call", "tool_use"}:
        return AgentEvent(
            type="tool_call",
            payload={
                "name": getattr(raw, "tool_name", None)
                or getattr(raw, "name", "unknown"),
                "arguments": getattr(raw, "arguments", None)
                or getattr(raw, "input", "{}"),
            },
        )
    if event_type in {"tool_output", "tool_result"}:
        return AgentEvent(
            type="tool_output",
            payload={
                "output": getattr(raw, "output", None)
                or getattr(raw, "result", None)
            },
        )
    if event_type in {"message", "text", "completion"}:
        text = getattr(raw, "text", None) or getattr(raw, "content", "")
        if not text:
            return None
        return AgentEvent(
            type="message",
            payload={
                "text": str(text),
                "final": bool(getattr(raw, "final", False)),
            },
        )
    return None


class _ClaudeToolView:
    """Surface a :class:`ToolSpec` as the SimpleTool-like attributes the SDK reads.

    The Claude Agents SDK consumes objects exposing ``name``,
    ``description``, ``params_json_schema``, and ``on_invoke_tool``. We
    construct a thin adapter from each :class:`ToolSpec` rather than
    forcing the SDK to learn ellements' canonical type.
    """

    __slots__ = ("name", "description", "params_json_schema", "_invoke", "on_invoke_tool")

    def __init__(self, spec: ToolSpec) -> None:
        self.name = spec.name
        self.description = spec.description
        self.params_json_schema = spec.params_json_schema
        self._invoke = spec.invoke
        self.on_invoke_tool = bind_json_invoker(spec.invoke)

    async def invoke(self, **kwargs: Any) -> Any:
        return await self._invoke(**kwargs)


def _adapt_spec_for_claude(spec: ToolSpec) -> Any:
    """Adapt a canonical :class:`ToolSpec` to the Claude Agents SDK shape."""
    return _ClaudeToolView(spec)


__all__ = ["ClaudeAgentsBackend"]
