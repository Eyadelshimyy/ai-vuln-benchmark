"""Event records emitted across ellements layers.

This module owns the **vocabulary** observers consume. We keep it
deliberately small and immutable-by-default so that every observer
implementation knows the exact shape of every field, with no surprises
from subclassing.

Two families of events live here:

LLM call events (consumed by :class:`LLMObserver`)
    Every method on :class:`~ellements.core.LLMClient` (``complete``,
    ``complete_structured``, ``complete_with_tools``, ``stream``,
    ``generate_image``, ``loglikelihood``) fires the same three events:
    :class:`LLMRequestEvent` before the call leaves the client,
    :class:`LLMResponseEvent` on success, :class:`LLMErrorEvent` on
    failure. The triple shares a ``call_id`` so observers can correlate
    them.

Agent run events (consumed by streaming agent backends)
    :class:`AgentEvent` is the uniform shape yielded by
    :meth:`~ellements.agents.AgentBackend.stream_run`. Backends
    translate their native event streams (OpenAI Agents SDK,
    Claude Agents SDK, etc.) onto this small vocabulary so UIs can
    consume them without backend-specific code.

Co-locating both families here makes ellements' observability surface
discoverable from a single place. Layers that need only one family
import only what they need (events are independent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# LLM call events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LLMRequestEvent:
    """Event fired before an LLM call leaves the client.

    Observers receive this event before the network request hits the
    provider, which makes it the right place to record the exact
    payload (messages, parameters, available tools) that will be sent.

    Attributes:
        call_id: UUID that identifies this call across the
            ``request``/``response``/``error`` triple.
        method: The :class:`LLMClient` method name that initiated this
            call (e.g. ``"complete"``, ``"stream"``).
        model: The model identifier the call is targeting.
        messages: The serialized message list as sent to the provider.
        temperature: The temperature parameter, if specified.
        max_tokens: The max-tokens parameter, if specified.
        tools: Provider-formatted tool definitions, if any.
        extra_params: Additional provider-specific parameters.
        metadata: Free-form, observer-supplied annotation channel.
            Observers may carry context (e.g. workstream id) here from
            request through response/error.
    """

    call_id: str
    method: str
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None
    max_tokens: int | None
    tools: list[dict[str, Any]] = field(default_factory=list)
    extra_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponseEvent:
    """Event fired after a successful LLM call returns.

    Attributes:
        call_id: Matches the originating :class:`LLMRequestEvent`.
        method: Same as on the request event.
        model: Same as on the request event.
        response: The final assistant text. For streaming calls this
            is the concatenated content.
        duration_ms: Wall-clock latency in milliseconds.
        tool_calls: Tool calls the model issued during this turn,
            already merged with any tool-call results the client
            looped back internally.
        usage: Provider-reported token usage, when available.
        metadata: Free-form, observer-supplied annotation channel.
    """

    call_id: str
    method: str
    model: str
    response: str
    duration_ms: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMErrorEvent:
    """Event fired when an LLM call raises (after any retries).

    Attributes:
        call_id: Matches the originating :class:`LLMRequestEvent`.
        method: Same as on the request event.
        model: Same as on the request event.
        error: The actual exception instance that ended the call.
            Observers should not assume the exception is JSON-safe;
            persistent loggers usually record ``type(error).__name__``
            and ``str(error)`` instead of the live object.
        duration_ms: Wall-clock time spent before the failure.
        metadata: Free-form, observer-supplied annotation channel.
    """

    call_id: str
    method: str
    model: str
    error: BaseException
    duration_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent run events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentEvent:
    """One incremental event surfaced by streaming agent runs.

    Backends map their native event types onto a small,
    backend-agnostic vocabulary so UIs can consume agent streams
    without knowing which SDK powers the backend:

    - ``"agent_active"``: a new sub-agent took the floor.
      Payload: ``{"name": str}``.
    - ``"tool_call"``: the agent invoked a tool.
      Payload: ``{"name": str, "arguments": Mapping[str, Any] | str}``.
    - ``"tool_output"``: a tool produced an output.
      Payload: ``{"output": Any}``.
    - ``"message"``: the agent produced a (possibly partial) message.
      Payload: ``{"text": str, "final": bool}``.

    Backends MAY surface additional event types under
    backend-namespaced keys (e.g. ``"openai.raw_response"``).
    Consumers should ignore unknown types.

    Attributes:
        type: One of the type strings above (or a backend-namespaced
            extension).
        payload: Type-specific payload as documented above.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentEvent",
    "LLMErrorEvent",
    "LLMRequestEvent",
    "LLMResponseEvent",
]
