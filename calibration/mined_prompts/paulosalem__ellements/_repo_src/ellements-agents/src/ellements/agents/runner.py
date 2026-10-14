"""Run agents through any :class:`AgentBackend` with progress + stats.

The runner is backend-agnostic: it consumes the uniform
:class:`~ellements.agents.AgentEvent` vocabulary emitted by
:meth:`AgentBackend.stream_run`. Stats are kept in a clean dict-backed
:class:`AgentRunStats` model — there are no legacy ``searches`` /
``results_examined`` aliases. Callers that want them increment the
metric explicitly via :meth:`AgentRunStats.increment_metric`.

The single entry point is :func:`run_agent_with_progress`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .backend import AgentBackend, AgentEvent

DEFAULT_MAX_TURNS = 200

ProgressCallback = Callable[[str], Awaitable[None] | None]
StatsCallback = Callable[["AgentRunStats"], Awaitable[None] | None]


# ── Stats ──────────────────────────────────────────────────────────


@dataclass
class AgentRunStats:
    """Statistics gathered during one agent run."""

    tool_calls: int = 0
    tool_outputs: int = 0
    observed_items: int = 0
    tool_names: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, int] = field(default_factory=dict)

    def increment_metric(self, name: str, amount: int = 1) -> None:
        self.metrics[name] = self.metrics.get(name, 0) + amount

    def increment_tool_call(self, tool_name: str) -> None:
        self.tool_calls += 1
        clean = (tool_name or "unknown").strip() or "unknown"
        self.tool_names[clean] = self.tool_names.get(clean, 0) + 1

    def increment_tool_output(self, amount: int = 1) -> None:
        self.tool_outputs += amount

    def observe_items(self, count: int) -> None:
        if count > 0:
            self.observed_items += count

    def merge(self, other: AgentRunStats) -> None:
        self.tool_calls += other.tool_calls
        self.tool_outputs += other.tool_outputs
        self.observed_items += other.observed_items
        for name, count in other.tool_names.items():
            self.tool_names[name] = self.tool_names.get(name, 0) + count
        for name, count in other.metrics.items():
            self.metrics[name] = self.metrics.get(name, 0) + count

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "tool_outputs": self.tool_outputs,
            "observed_items": self.observed_items,
            "tool_names": dict(self.tool_names),
            "metrics": dict(self.metrics),
        }


@dataclass
class AgentRunResult:
    """Wrapper exposing the backend-native result plus collected stats.

    Attributes:
        result: The raw backend-native result object. Inspect this
            directly to access provider-specific data (e.g. token
            usage, raw items, tool-call traces).
        stats: Uniform :class:`AgentRunStats` collected during the
            run, regardless of which backend produced *result*.

    The ``final_output`` property is the canonical, backend-agnostic
    accessor for the assistant's final text response. Anything else
    must go through ``result.result.<native_attr>`` — the wrapper
    intentionally does **not** proxy unknown attributes to keep the
    public surface explicit and discoverable.
    """

    result: Any
    stats: AgentRunStats

    @property
    def final_output(self) -> str:
        """Return the assistant's final text reply (``""`` if missing)."""
        return getattr(self.result, "final_output", "")


# ── Public entry point ─────────────────────────────────────────────


async def run_agent_with_progress(
    backend: AgentBackend,
    agent: Any,
    query: str,
    *,
    progress_callback: ProgressCallback | None = None,
    stats_callback: StatsCallback | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    session: Any | None = None,
    show_activity_log: bool = True,
) -> AgentRunResult:
    """Run *agent* through *backend* and report progress incrementally.

    When ``show_activity_log`` is True the runner uses
    :meth:`AgentBackend.stream_run`; otherwise it uses
    :meth:`AgentBackend.run`.
    """
    stats = AgentRunStats()
    await _emit(
        progress_callback,
        "🚀 Starting agent…\n"
        f"  [cyan]Backend:[/cyan] {backend.name}\n"
        f"  [cyan]Max turns:[/cyan] {max_turns}\n"
        f"  [cyan]Activity log:[/cyan] "
        f"{'enabled' if show_activity_log else 'disabled'}\n"
        f"  [cyan]Model:[/cyan] {getattr(agent, 'model', 'default')}\n",
    )

    if show_activity_log and progress_callback is not None:
        stream = backend.stream_run(
            agent, query, max_turns=max_turns, session=session
        )
        await _emit(
            progress_callback,
            "\n[bold cyan]📋 Agent Activity Log:[/bold cyan]\n",
        )
        try:
            async for event in stream:
                await _handle_event(
                    event, stats, progress_callback, stats_callback
                )
        except asyncio.CancelledError:
            await _emit(
                progress_callback,
                "\n[bold red]🛑 Streaming cancelled[/bold red]",
            )
            raise
        result = stream.result
    else:
        result = await backend.run(
            agent, query, max_turns=max_turns, session=session
        )
        _update_stats_from_native(stats, result)

    await _emit(
        progress_callback,
        "\n\n[bold green]🎉 Final Answer[/bold green]\n",
    )
    return AgentRunResult(result=result, stats=stats)


# ── Helpers ────────────────────────────────────────────────────────


async def _handle_event(
    event: AgentEvent,
    stats: AgentRunStats,
    progress_callback: ProgressCallback | None,
    stats_callback: StatsCallback | None,
) -> None:
    """Dispatch a single :class:`AgentEvent` to its per-type handler."""
    handler = _EVENT_HANDLERS.get(event.type)
    if handler is None:
        return
    await handler(event, stats, progress_callback, stats_callback)


async def _on_agent_active(
    event: AgentEvent,
    _stats: AgentRunStats,
    progress_callback: ProgressCallback | None,
    _stats_callback: StatsCallback | None,
) -> None:
    await _emit(
        progress_callback,
        f"\n[bold]🔄 Agent active:[/bold] "
        f"[cyan]{event.payload.get('name', 'Agent')}[/cyan]",
    )


async def _on_tool_call(
    event: AgentEvent,
    stats: AgentRunStats,
    progress_callback: ProgressCallback | None,
    stats_callback: StatsCallback | None,
) -> None:
    name = str(event.payload.get("name", "unknown"))
    args = event.payload.get("arguments", "{}")
    stats.increment_tool_call(name)
    await _emit_stats(stats_callback, stats)
    await _emit(
        progress_callback,
        f"\n[bold yellow]🔧 Tool:[/bold yellow] [magenta]{name}[/magenta]",
    )
    formatted = _format_tool_args(args)
    if formatted:
        await _emit(progress_callback, f"   {formatted}")


async def _on_tool_output(
    event: AgentEvent,
    stats: AgentRunStats,
    progress_callback: ProgressCallback | None,
    stats_callback: StatsCallback | None,
) -> None:
    output = event.payload.get("output")
    stats.increment_tool_output()
    result_count = _count_results(output)
    if result_count:
        stats.observe_items(result_count)
    await _emit_stats(stats_callback, stats)
    await _emit(progress_callback, "\n[bold green]✅ Result[/bold green]")
    for line in _format_tool_output(output):
        await _emit(progress_callback, f"   {line}")


async def _on_message(
    event: AgentEvent,
    _stats: AgentRunStats,
    progress_callback: ProgressCallback | None,
    _stats_callback: StatsCallback | None,
) -> None:
    text = str(event.payload.get("text", ""))
    if not text:
        return
    if len(text) < 500:
        await _emit(
            progress_callback,
            f"\n[bold blue]💭 Thinking[/bold blue]\n   [dim]{text}[/dim]",
        )
    else:
        preview = text[:150].replace("\n", " ")
        await _emit(
            progress_callback,
            f"\n[bold blue]💭 Composing response:[/bold blue] "
            f"[dim]{preview}…[/dim]",
        )


_EventHandler = Callable[
    [AgentEvent, AgentRunStats, ProgressCallback | None, StatsCallback | None],
    Awaitable[None],
]

# Per-event-type dispatch table for ``_handle_event``. Adding a new
# event type means appending one entry here rather than threading
# another ``elif`` branch through a growing if-cascade.
_EVENT_HANDLERS: dict[str, _EventHandler] = {
    "agent_active": _on_agent_active,
    "tool_call": _on_tool_call,
    "tool_output": _on_tool_output,
    "message": _on_message,
}


async def _emit(cb: ProgressCallback | None, message: str) -> None:
    if cb is None:
        return
    result = cb(message)
    if asyncio.iscoroutine(result):
        await result


async def _emit_stats(
    cb: StatsCallback | None, stats: AgentRunStats
) -> None:
    if cb is None:
        return
    result = cb(stats)
    if asyncio.iscoroutine(result):
        await result


def _update_stats_from_native(stats: AgentRunStats, result: Any) -> None:
    """Populate stats from native-result items when streaming was disabled."""
    new_items = getattr(result, "new_items", None) or []
    for item in new_items:
        item_type = getattr(item, "type", None)
        if item_type == "tool_call_item":
            raw_item = getattr(item, "raw_item", None) or item
            stats.increment_tool_call(getattr(raw_item, "name", "unknown"))
        elif item_type == "tool_call_output_item":
            stats.increment_tool_output()
            output = getattr(item, "output", None)
            stats.observe_items(_count_results(output))


def _count_results(output: Any) -> int:
    """Best-effort count of result-like items in a tool output."""
    if output is None:
        return 0
    results = getattr(output, "results", None)
    if isinstance(results, list):
        return len(results)
    return 0


def _format_tool_args(args: Any) -> str:
    """Format tool args (JSON string or mapping) into a Rich-styled line."""
    if isinstance(args, str):
        if args in ("", "{}", "null"):
            return ""
        try:
            data = json.loads(args)
        except json.JSONDecodeError:
            return f"[dim]{args}[/dim]"
    elif isinstance(args, dict):
        data = args
    else:
        return ""
    parts: list[str] = []
    for key, value in data.items():
        formatted = (
            f'"{value}"' if isinstance(value, str) else json.dumps(value)
        )
        parts.append(f"[blue]{key}[/blue]=[green]{formatted}[/green]")
    return ", ".join(parts)


def _format_tool_output(output: Any) -> list[str]:
    if output is None:
        return []
    if isinstance(output, str):
        if len(output) > 200:
            return [
                f"[cyan]Text:[/cyan] {len(output.split())} words",
                f"[dim]Preview:[/dim] {output[:150]}…",
            ]
        return [f"[green]{output}[/green]"]
    text = str(output)
    if len(text) > 300:
        return [f"[dim]{text[:300]}…[/dim]"]
    return [f"[dim]{text}[/dim]"]


__all__ = [
    "AgentRunResult",
    "AgentRunStats",
    "DEFAULT_MAX_TURNS",
    "ProgressCallback",
    "StatsCallback",
    "run_agent_with_progress",
]
