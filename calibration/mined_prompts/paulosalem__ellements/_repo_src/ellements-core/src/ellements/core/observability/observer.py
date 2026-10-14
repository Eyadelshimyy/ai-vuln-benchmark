"""Observer Protocols for ellements events.

An *observer* receives uniform events from one of ellements' event
streams without coupling to the producer. Two protocols live here:

:class:`LLMObserver` — push-based. Every
:class:`~ellements.core.LLMClient` method fires the same three events
(:class:`LLMRequestEvent`, :class:`LLMResponseEvent`,
:class:`LLMErrorEvent`) to every attached observer. Observers are
``async`` so they can do I/O (file writes, network telemetry) without
blocking the event loop.

Agent events are pulled, not pushed: callers iterate the async stream
returned by :meth:`~ellements.agents.AgentBackend.stream_run`. There is
no ``AgentObserver`` Protocol because streaming itself is the observer
mechanism — the same backend supports zero, one, or many concurrent
consumers via independent ``stream_run`` invocations.

To convert a streaming agent into a push-based observable, callers
typically wrap the stream and forward events to whatever sink they
want (e.g. a GUI bridge). Such adapters belong outside core.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .events import LLMErrorEvent, LLMRequestEvent, LLMResponseEvent


@runtime_checkable
class LLMObserver(Protocol):
    """Async push-based observer attached to an :class:`LLMClient`.

    The client guarantees the event triple semantics: every call fires
    exactly one ``on_request``, followed by exactly one of
    ``on_response`` (success) or ``on_error`` (failure). The events
    share a ``call_id`` so stateful observers can correlate them.

    All three methods are ``async`` to permit non-blocking I/O. The
    client awaits each method in sequence; observers that need to
    fan out to many sinks should buffer internally rather than block
    the LLM call.
    """

    async def on_request(self, event: LLMRequestEvent) -> None:
        """Receive the pre-flight event for an LLM call."""

    async def on_response(self, event: LLMResponseEvent) -> None:
        """Receive the success event for an LLM call."""

    async def on_error(self, event: LLMErrorEvent) -> None:
        """Receive the failure event for an LLM call."""


__all__ = ["LLMObserver"]
