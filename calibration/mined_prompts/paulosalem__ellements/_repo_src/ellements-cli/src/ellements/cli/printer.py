"""High-level CLI printer with app branding.

Wraps the low-level building blocks in :mod:`components` into a stateful,
branded workflow API that any CLI app can instantiate.

All chrome (headers, spinners, events, stats) goes to *stderr*. Only
actual output (composed prompt, JSON data) goes to *stdout* so pipelines
stay clean.

Verbose progress events are dispatched through a pluggable
:class:`EventRenderer` Protocol. Apps with their own event vocabulary
register a renderer at construction time; the printer remains agnostic
to specific event names.

Usage::

    from ellements.cli import CliPrinter

    printer = CliPrinter("Prompt Composer", icon="🧩")
    printer.header({"Spec": "file.md", "Model": "gpt-4o-mini"})
    printer.variables({"topic": "cats"})
    printer.status("Composing prompt…")
    # pass printer.event as an on_event callback
    result = await controller.compose(spec, vars, on_event=printer.event)
    printer.result_markdown(result.composed_prompt)
    printer.stats({"Tool calls": 3, "Issues": 1}, elapsed=2.1)
    printer.done()
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rich.console import Console

from . import components as _c


@runtime_checkable
class EventRenderer(Protocol):
    """Renders verbose progress events to the supplied :class:`Console`."""

    def render(
        self,
        event_type: str,
        data: Mapping[str, Any],
        console: Console,
    ) -> None: ...


class CliPrinter:
    """Branded CLI printer for terminal output.

    Args:
        app_name: Display name of the application.
        icon: Emoji or symbol prepended to the title.
        border_style: Default Rich border style for panels.
        verbose: When False, :meth:`event` is a no-op.
        event_renderer: Optional plug-in for rendering domain-specific
            verbose events. When None, :meth:`event` accepts events but
            does nothing with them in verbose mode.
    """

    def __init__(
        self,
        app_name: str,
        *,
        icon: str = "",
        border_style: str = "bright_blue",
        verbose: bool = False,
        event_renderer: EventRenderer | None = None,
    ) -> None:
        self.app_name = app_name
        self.icon = icon
        self.border_style = border_style
        self.verbose = verbose
        self.event_renderer = event_renderer
        self.console = Console(stderr=True)
        self.output_console = Console()

    # ── Header / preamble ──────────────────────────────────────────

    def header(self, subtitle_items: Mapping[str, str] | None = None) -> None:
        _c.app_header(
            self.app_name,
            dict(subtitle_items) if subtitle_items else None,
            icon=self.icon,
            border_style=self.border_style,
            console=self.console,
        )

    def variables(
        self, data: Mapping[str, Any], *, title: str = "📋 Variables"
    ) -> None:
        _c.key_value_table(dict(data), title=title, console=self.console)

    # ── Status / progress ──────────────────────────────────────────

    def status(self, message: str) -> None:
        self.console.print()
        self.console.print(
            f"[bold bright_blue]⚙  {message}[/]", highlight=False
        )

    def event(self, event_type: str, data: Mapping[str, Any]) -> None:
        """Forward an event to the configured renderer, if any.

        Designed to be passed directly as an ``on_event`` callback. Does
        nothing when *verbose* is False or no renderer is installed.
        """
        if not self.verbose or self.event_renderer is None:
            return
        self.event_renderer.render(event_type, data, self.console)

    # ── Results ────────────────────────────────────────────────────

    def result_markdown(
        self, content: str, *, title: str = "Result"
    ) -> None:
        _c.result_markdown(
            content,
            title=title,
            console=self.console,
            output_console=self.output_console,
        )

    def result_json(self, data: Any) -> None:
        _c.result_json(data, output_console=self.output_console)

    def issues(
        self,
        issues_list: Sequence[Mapping[str, str]],
        *,
        title: str = "Issues",
    ) -> None:
        _c.issues_block(
            [dict(i) for i in issues_list],
            title=title,
            console=self.console,
        )

    def stats(
        self,
        stats_dict: Mapping[str, str | int | float],
        *,
        elapsed: float | None = None,
    ) -> None:
        merged = dict(stats_dict)
        if elapsed is not None:
            merged["Time"] = f"{elapsed:.1f}s"
        _c.stats_footer(merged, console=self.console)

    def file_written(self, path: str | Path) -> None:
        _c.success(
            f"Output written to [bright_cyan]{path}[/]",
            console=self.console,
        )

    def done(self, message: str = "Done. 👋") -> None:
        self.console.print(f"[dim]{message}[/]")

    # ── Chat ───────────────────────────────────────────────────────

    def welcome(
        self,
        subtitle_items: Mapping[str, str] | None = None,
        *,
        commands: Mapping[str, str] | None = None,
    ) -> None:
        _c.welcome_banner(
            self.app_name,
            dict(subtitle_items) if subtitle_items else None,
            commands=dict(commands) if commands else None,
            icon=self.icon,
            border_style=self.border_style,
            console=self.console,
        )

    def user_message(self, content: str) -> None:
        _c.chat_message(
            "user",
            content,
            console=self.console,
            output_console=self.output_console,
        )

    def assistant_message(self, content: str) -> None:
        _c.chat_message(
            "assistant",
            content,
            console=self.console,
            output_console=self.output_console,
        )

    def tool_badge(self, tool_name: str) -> None:
        _c.tool_use_badge(tool_name, console=self.console)

    def input_prompt(self) -> str:
        return _c.chat_input_prompt()

    @contextmanager
    def thinking(self, message: str = "Thinking…") -> Iterator[None]:
        """Show a spinner while work is happening."""
        with self.console.status(
            f"[bold bright_blue]{message}[/]",
            spinner="dots",
            spinner_style="bright_blue",
        ):
            yield

    def goodbye(self, message: str = "Goodbye! 👋") -> None:
        self.console.print(f"\n[dim italic]{message}[/]\n")

    # ── One-liners ─────────────────────────────────────────────────

    def success(self, message: str) -> None:
        _c.success(message, console=self.console)

    def error(self, message: str) -> None:
        _c.error(message, console=self.console)

    def warning(self, message: str) -> None:
        _c.warning(message, console=self.console)


__all__ = ["CliPrinter", "EventRenderer"]
