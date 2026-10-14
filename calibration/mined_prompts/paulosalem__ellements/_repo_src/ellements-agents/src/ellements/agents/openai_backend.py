"""OpenAI Agents SDK backend implementing :class:`AgentBackend`."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from ellements.core import ToolRegistry, ToolSpec
from ellements.core.tools import bind_json_invoker

from .backend import AgentEvent


class OpenAIAgentsBackend:
    """Backend that adapts the ``openai-agents`` SDK to ellements."""

    name: str = "openai_agents"

    def __init__(self) -> None:
        try:
            from agents import Agent, Runner

            self._Agent = Agent
            self._Runner = Runner
        except ImportError as exc:
            raise ImportError(
                "openai-agents package not installed. "
                "Install with: pip install 'ellements[agents]'"
            ) from exc

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
            tools=[_adapt_spec_for_openai(spec) for spec in tools.values()],
            model=model,
        )

    # ── Session construction ───────────────────────────────────────

    def create_session(
        self,
        session_id: str | None = None,
        *,
        storage_path: Path | str | None = None,
    ) -> Any | None:
        """Create an ``openai-agents`` ``SQLiteSession`` instance.

        Returns ``None`` if the SDK does not expose ``SQLiteSession``.
        """
        try:
            from agents import SQLiteSession
        except ImportError:
            return None
        sid = session_id or f"openai_agents_{uuid.uuid4().hex}"
        if storage_path is None:
            return SQLiteSession(sid)
        return SQLiteSession(sid, str(storage_path))

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
    ) -> _OpenAIAgentStream:
        streaming_result = self._Runner.run_streamed(
            agent, task, max_turns=max_turns, session=session, **kwargs
        )
        return _OpenAIAgentStream(streaming_result)


class _OpenAIAgentStream:
    """:class:`AgentStream` impl that translates SDK events to AgentEvent."""

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
    """Map an OpenAI Agents SDK event into a uniform :class:`AgentEvent`."""
    event_type = getattr(raw, "type", None)
    if event_type == "raw_response_event":
        return None
    if event_type == "agent_updated_stream_event":
        new_agent = getattr(raw, "new_agent", None)
        return AgentEvent(
            type="agent_active",
            payload={"name": getattr(new_agent, "name", "Agent")},
        )
    if event_type != "run_item_stream_event":
        return None
    item = getattr(raw, "item", None)
    if item is None:
        return None
    item_type = getattr(item, "type", None)
    if item_type == "tool_call_item":
        raw_item = getattr(item, "raw_item", None) or item
        return AgentEvent(
            type="tool_call",
            payload={
                "name": getattr(raw_item, "name", "unknown"),
                "arguments": getattr(raw_item, "arguments", "{}"),
            },
        )
    if item_type == "tool_call_output_item":
        return AgentEvent(
            type="tool_output",
            payload={"output": getattr(item, "output", None)},
        )
    if item_type == "message_output_item":
        raw_item = getattr(item, "raw_item", None)
        text = _extract_message_text(raw_item)
        if not text:
            return None
        return AgentEvent(
            type="message",
            payload={"text": text, "final": False},
        )
    return None


def _extract_message_text(raw_item: Any) -> str:
    if raw_item is None or not hasattr(raw_item, "content"):
        return ""
    content_items = raw_item.content or []
    return "\n".join(
        getattr(c, "text", "") for c in content_items if hasattr(c, "text")
    )


def _adapt_spec_for_openai(spec: ToolSpec) -> Any:
    """Adapt a canonical :class:`ToolSpec` to the OpenAI Agents SDK shape."""
    from agents.tool import FunctionTool

    return FunctionTool(
        name=spec.name,
        description=spec.description,
        params_json_schema=spec.params_json_schema,
        on_invoke_tool=bind_json_invoker(spec.invoke),
    )


__all__ = ["OpenAIAgentsBackend"]
