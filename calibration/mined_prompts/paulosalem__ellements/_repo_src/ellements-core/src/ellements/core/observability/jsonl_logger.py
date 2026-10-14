"""Async-safe, date-rolling JSON-lines logger for LLM calls.

:class:`JsonlPromptLogger` is the canonical reference implementation of
:class:`~ellements.core.observability.LLMObserver`. It writes one
self-contained JSON object per LLM call to a daily ``.jsonl`` file:

- **One file per UTC day**: the active filename incorporates the
  current UTC date, so file rotation happens automatically without
  background timers.
- **Race-free**: every write is serialized through an
  :class:`asyncio.Lock`, so concurrent strategies sharing a logger
  cannot interleave bytes.
- **Resilient**: I/O failures are downgraded to a warning rather than
  propagated. Telemetry should never break the production path it is
  observing.
- **Self-contained records**: the request, response, tool calls,
  usage, and metadata for a single call all live in one line. There
  is no cross-line schema to reassemble.

For a human-readable view of an existing log, see
:func:`~ellements.core.observability.format_log_markdown`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .events import LLMErrorEvent, LLMRequestEvent, LLMResponseEvent

_logger = logging.getLogger(__name__)


class LoggedCall(BaseModel):
    """One self-contained log record for a single LLM call.

    Pydantic gives us free JSON round-tripping (including for
    multimodal content embedded inside ``messages``) and machine-
    readable schema introspection for downstream tooling.

    The model is intentionally permissive on input (``messages`` and
    ``tools`` are ``list[dict]``) because LLM payloads evolve faster
    than we can express their full structure in types.

    Attributes:
        call_id: UUID matching the originating
            :class:`LLMRequestEvent`.
        timestamp: ISO 8601 UTC timestamp of the call start.
        method: The :class:`LLMClient` method name (e.g.
            ``"complete"``).
        status: ``"ok"``, ``"error"``, or ``"pending"`` (the latter
            is an internal transient state never written to disk).
        model: The model identifier the call was sent to.
        temperature: The temperature parameter, if specified.
        max_tokens: The max-tokens parameter, if specified.
        messages: The serialized message list.
        tools: Provider-formatted tool definitions.
        response: The final assistant text on success.
        tool_calls: Tool calls the model issued.
        usage: Provider-reported token usage.
        duration_ms: Wall-clock latency.
        extra_params: Additional provider-specific parameters.
        metadata: Free-form annotation channel from observers.
        error_type: Exception class name on failure.
        error_message: Exception ``str()`` on failure.
    """

    call_id: str = Field(description="UUID identifying this call across events")
    timestamp: str = Field(description="ISO 8601 UTC timestamp of the call start")
    method: str = Field(description="LLMClient method name (e.g. 'complete')")
    status: str = Field(description="'ok' or 'error'")
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    response: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    duration_ms: int = 0
    extra_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class JsonlPromptLogger:
    """Async-safe, UTC-day-rolling JSON-lines observer for LLM calls.

    The logger stages each call as a :class:`LoggedCall` in an
    in-memory pending map keyed by ``call_id``. ``on_request`` opens
    the record; ``on_response``/``on_error`` close it and flush a
    single JSON line to disk. This means **the file only contains
    completed calls** — partial calls (e.g. interrupted) leave no
    trace.

    Args:
        log_dir: Directory where ``llm-log-YYYY-MM-DD.jsonl`` files
            are written. Created if it does not exist.
        filename_pattern: ``strftime``-formatted prefix; the resulting
            file is ``log_dir / f"{filename_pattern}.jsonl"``. The
            default ``"llm-log-%Y-%m-%d"`` produces
            ``llm-log-2024-05-16.jsonl``.

    Concurrency:
        All writes are serialized by an :class:`asyncio.Lock`, so
        multiple concurrent strategies sharing a single logger
        produce a valid JSON-lines file (no torn rows).

    Date rollover:
        Every write computes the current UTC date through
        :attr:`current_path`. When the date changes, the next write
        opens a new file. No background timers, no locking on the
        clock.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        filename_pattern: str = "llm-log-%Y-%m-%d",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.log_dir.chmod(0o700)
        self._filename_pattern = filename_pattern
        self._lock = asyncio.Lock()
        # In-flight requests indexed by call_id so on_response / on_error
        # can merge with the request payload (messages, tools, params)
        # without every method plumbing a full record around.
        self._pending: dict[str, LoggedCall] = {}

    # ── Observer protocol ────────────────────────────────────────────

    async def on_request(self, event: LLMRequestEvent) -> None:
        """Stage a pending record for *event*.

        The record is not flushed to disk until the matching response
        or error arrives. This guarantees logged lines correspond to
        completed calls.
        """
        record = LoggedCall(
            call_id=event.call_id,
            timestamp=datetime.now(UTC).isoformat(),
            method=event.method,
            status="pending",
            model=event.model,
            temperature=event.temperature,
            max_tokens=event.max_tokens,
            messages=list(event.messages),
            tools=list(event.tools),
            extra_params=dict(event.extra_params),
            metadata=dict(event.metadata),
        )
        self._pending[event.call_id] = record

    async def on_response(self, event: LLMResponseEvent) -> None:
        """Finalize the pending record with success data and flush."""
        record = self._pending.pop(event.call_id, None)
        if record is None:
            # The pending record may be missing if this logger was
            # attached mid-flight; fall back to a partial record that
            # still captures everything the response event knows.
            record = LoggedCall(
                call_id=event.call_id,
                timestamp=datetime.now(UTC).isoformat(),
                method=event.method,
                status="ok",
                model=event.model,
            )
        record.status = "ok"
        record.response = event.response
        record.tool_calls = list(event.tool_calls)
        record.usage = event.usage
        record.duration_ms = event.duration_ms
        record.metadata = {**record.metadata, **event.metadata}
        await self._write(record)

    async def on_error(self, event: LLMErrorEvent) -> None:
        """Finalize the pending record with error data and flush."""
        record = self._pending.pop(event.call_id, None)
        if record is None:
            record = LoggedCall(
                call_id=event.call_id,
                timestamp=datetime.now(UTC).isoformat(),
                method=event.method,
                status="error",
                model=event.model,
            )
        record.status = "error"
        record.duration_ms = event.duration_ms
        record.error_type = type(event.error).__name__
        record.error_message = str(event.error)
        record.metadata = {**record.metadata, **event.metadata}
        await self._write(record)

    # ── Paths ────────────────────────────────────────────────────────

    @property
    def current_path(self) -> Path:
        """The absolute path the next write will land in.

        Computed fresh on every access so UTC day rollover applies
        without any external state.
        """
        stamp = datetime.now(UTC).strftime(self._filename_pattern)
        return self.log_dir / f"{stamp}.jsonl"

    # ── Internals ────────────────────────────────────────────────────

    async def _write(self, record: LoggedCall) -> None:
        """Serialize *record* and append it to :attr:`current_path`.

        I/O happens inside :func:`asyncio.to_thread` so the event loop
        is not blocked even on slow filesystems. The lock guarantees
        ordering between concurrent writes.
        """
        line = record.model_dump_json() + "\n"
        async with self._lock:
            path = self.current_path
            try:
                await asyncio.to_thread(self._append_line, path, line)
            except OSError as exc:
                # Telemetry must never break the path it is observing.
                _logger.warning(
                    "JsonlPromptLogger write failed for %s: %s", path, exc
                )

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        if os.name != "nt":
            path.chmod(0o600)


__all__ = ["JsonlPromptLogger", "LoggedCall"]
