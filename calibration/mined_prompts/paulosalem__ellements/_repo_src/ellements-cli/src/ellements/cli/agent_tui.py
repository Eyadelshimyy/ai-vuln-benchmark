"""Generic Textual TUI for agent interactions.

This is intentionally *application-agnostic*: nothing about the runner,
the persona/guideline source, the save destination, or any custom
visualization is hardcoded. Callers wire in their own implementations
via the four plug-in protocols:

- :class:`AgentRunner`: how a query becomes a response.
- :class:`SaveHandler`: how a response is persisted.
- :class:`PersonaProvider` / :class:`GuidelineProvider`: optional
  identity/style switching.

The runner receives a :class:`TuiControls` handle for emitting
incremental updates (progress messages, stat snapshots, custom-widget
payloads) and checking for cancellation. Modal UI state (idle,
awaiting filename, awaiting persona selection, awaiting guideline
selection) is tracked through a single :class:`Mode` enum — no scattered
boolean flags.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rich.markdown import Markdown
from rich.panel import Panel
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.suggester import Suggester
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

# ── Plug-in protocols ───────────────────────────────────────────────


@runtime_checkable
class SaveHandler(Protocol):
    """Persists a response to wherever the embedding app wants it."""

    def suggest_filename(self, query: str | None, response: str) -> str:
        """Return a default filename to pre-fill the save prompt."""

    def save(self, content: str, filename: str) -> Path:
        """Persist *content* and return the final on-disk path."""


@runtime_checkable
class PersonaProvider(Protocol):
    """Adapts a persona library to the TUI's selection model."""

    def list_ids(self) -> list[str]: ...
    def current_id(self) -> str | None: ...
    def select(self, persona_id: str) -> None: ...


@runtime_checkable
class GuidelineProvider(Protocol):
    """Adapts a guideline library to the TUI's selection model."""

    def list_ids(self) -> list[str]: ...
    def current_id(self) -> str | None: ...
    def select(self, guideline_id: str) -> None: ...
    def clear(self) -> None: ...


@runtime_checkable
class AgentRunner(Protocol):
    """Executes a single query and returns a structured result.

    Implementations may emit incremental progress/stats/widget updates
    through :class:`TuiControls`. They should periodically check
    ``controls.is_cancelled()`` and raise :class:`asyncio.CancelledError`
    when set.
    """

    async def run(self, query: str, controls: TuiControls) -> RunResult: ...


@dataclass(slots=True)
class RunResult:
    """The outcome of an :meth:`AgentRunner.run` invocation."""

    text: str
    stats: Mapping[str, Any] | None = None


# ── Mode enum (replaces scattered boolean modal flags) ──────────────


class Mode(Enum):
    """Modal input state for the TUI."""

    IDLE = "idle"
    AWAITING_FILENAME = "awaiting_filename"
    AWAITING_PERSONA = "awaiting_persona"
    AWAITING_GUIDELINE = "awaiting_guideline"


# ── Default SaveHandler ─────────────────────────────────────────────


class FileSaveHandler:
    """Default :class:`SaveHandler` writing Markdown files under *base_dir*.

    Args:
        base_dir: Directory to write into. Created if missing.
        prefix: String prepended to auto-generated filenames.
    """

    def __init__(
        self, base_dir: Path | str = ".", *, prefix: str = "response"
    ) -> None:
        self._base_dir = Path(base_dir)
        self._prefix = prefix

    def suggest_filename(self, query: str | None, response: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if query:
            slug = re.sub(r"[^\w\s]", "", query).strip().split()[:3]
            if slug:
                return f"{'_'.join(slug).lower()}_{timestamp}.md"
        return f"{self._prefix}_{timestamp}.md"

    def save(self, content: str, filename: str) -> Path:
        if not filename.endswith(".md"):
            filename = f"{filename}.md"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = (self._base_dir / filename).resolve()
        path.write_text(content, encoding="utf-8")
        return path



# ── Slash commands ──────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class SlashCommand:
    """One slash command shown in the command menu."""

    name: str
    description: str
    requires_persona: bool = False
    requires_guideline: bool = False
    requires_custom_widget: bool = False


BUILTIN_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/help", "Show help and available commands"),
    SlashCommand("/cancel", "Cancel the currently running agent"),
    SlashCommand("/clear", "Clear the output panel"),
    SlashCommand("/reset", "Reset the entire session"),
    SlashCommand("/save", "Save last response to a file"),
    SlashCommand("/persona", "Switch user persona", requires_persona=True),
    SlashCommand(
        "/guideline",
        "Select guideline to refine prompt",
        requires_guideline=True,
    ),
    SlashCommand(
        "/viz",
        "Toggle custom visualization panel",
        requires_custom_widget=True,
    ),
    SlashCommand("/stats", "Toggle statistics panel"),
    SlashCommand("/quit", "Exit the application"),
)


# ── TuiConfig ───────────────────────────────────────────────────────


@dataclass(slots=True)
class TuiConfig:
    """Wiring for :class:`AgentTUI`."""

    agent_name: str
    agent_description: str
    runner: AgentRunner

    save_handler: SaveHandler = field(default_factory=FileSaveHandler)
    persona_provider: PersonaProvider | None = None
    guideline_provider: GuidelineProvider | None = None
    custom_widget: Any | None = None
    custom_widget_title: str = "Visualization"
    theme: str = "monokai"
    initial_stats: Mapping[str, Any] = field(default_factory=dict)


# ── TuiControls (handed to the runner) ──────────────────────────────


class TuiControls:
    """Callbacks the runner uses to push updates and check cancellation."""

    def __init__(self, app: AgentTUI) -> None:
        self._app = app

    def progress(self, message: str) -> None:
        """Emit a progress line to the output panel (thread-safe)."""
        self._app._post_progress(message)

    def update_stats(self, stats: Mapping[str, Any]) -> None:
        """Replace the current stats snapshot (thread-safe)."""
        self._app._post_stats(stats)

    def update_custom_widget(self, payload: Any) -> None:
        """Route *payload* to the custom widget's ``handle_payload`` (if any).

        Implementations whose custom widget implements
        ``handle_payload(payload)`` will receive it; otherwise this is a
        no-op.
        """
        self._app._post_custom_widget(payload)

    def is_cancelled(self) -> bool:
        return self._app._cancel_requested


# ── Slash command suggester ─────────────────────────────────────────


class _SlashSuggester(Suggester):  # type: ignore[misc]
    def __init__(self, commands: list[str]) -> None:
        super().__init__(use_cache=False, case_sensitive=False)
        self._commands = commands

    async def get_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None
        for cmd in self._commands:
            if cmd.startswith(value.lower()):
                return cmd
        return None


# ── Command menu overlay ────────────────────────────────────────────


class _CommandMenuOverlay(Container):  # type: ignore[misc]
    def __init__(self, commands: list[SlashCommand], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.commands = commands

    def compose(self) -> ComposeResult:
        yield Label(
            "[bold cyan]Available Commands[/bold cyan]",
            id="command-menu-title",
        )
        yield OptionList(
            *[
                Option(
                    f"{c.name}  [dim]{c.description}[/dim]", id=c.name
                )
                for c in self.commands
            ],
            id="command-menu-list",
        )


# ── AgentTUI ────────────────────────────────────────────────────────


class AgentTUI(App[None]):  # type: ignore[misc]
    """Application-agnostic Textual TUI for any :class:`AgentRunner`."""

    CSS = """
    Screen { background: $surface; }
    #title-bar {
        background: $surface; color: $text; height: 1;
        content-align: left middle; padding: 0 2; text-style: bold;
    }
    #agent-info { display: none; }
    #main-container {
        height: 1fr; background: $surface;
        border: solid $primary; margin: 0 1;
    }
    #output-panel {
        border: none; height: 1fr; background: $surface; padding: 1 2;
    }
    #input-container {
        height: auto; background: $surface; padding: 1 2;
        border-top: solid $primary;
    }
    #status-bar {
        height: 1; background: $surface; color: $text-muted;
        padding: 0 2; text-style: dim;
    }
    Input { border: solid $primary; background: $surface; color: $text; }
    Input:focus { border: solid $accent; }
    Button { margin: 0 1; background: $surface; color: $text; border: solid $primary; }
    Button:hover { background: $primary; border: solid $accent; }
    Button.-primary { background: $primary; border: solid $primary; }
    Button.-primary:hover { background: $accent; border: solid $accent; }
    _CommandMenuOverlay {
        display: none; layer: overlay; offset: 2 2;
        width: 60; height: auto; max-height: 15;
        background: $panel; border: thick $accent; padding: 0;
    }
    _CommandMenuOverlay.visible { display: block; }
    #command-menu-title {
        width: 100%; height: auto; padding: 0 1;
        background: $boost; color: $text; text-style: bold;
    }
    #command-menu-list {
        width: 100%; height: auto; background: $panel;
        border: none; padding: 0 1;
    }
    #stats-panel {
        width: 30; height: 1fr; background: $surface;
        border-left: solid $primary; padding: 1 2; display: none;
    }
    #stats-panel.visible { display: block; }
    #stats-title {
        text-style: bold; color: $accent; margin-bottom: 1;
    }
    #custom-panel {
        width: 50; height: 1fr; background: $surface;
        border-left: solid $primary; padding: 1 2; display: none;
    }
    #custom-panel.visible { display: block; }
    #custom-title { text-style: bold; color: $accent; margin-bottom: 1; }
    #content-area { width: 1fr; height: 1fr; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+x", "cancel", "Cancel", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+r", "reset", "Reset", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    def __init__(self, config: TuiConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._mode: Mode = Mode.IDLE
        self._mode_payload: Any = None

        self._query_history: list[str] = []
        self._last_query: str | None = None
        self._last_response: str | None = None

        self._is_processing: bool = False
        self._cancel_requested: bool = False
        self._current_worker: Any = None

        self._command_menu_visible: bool = False
        self._stats_panel_visible: bool = True
        self._custom_panel_visible: bool = config.custom_widget is not None

        self._stats_snapshot: dict[str, Any] = dict(config.initial_stats)
        self._session_start: datetime = datetime.now()
        self._stats_timer: Any = None

        self.theme = config.theme

    # ── Compose / mount ─────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        commands = list(self._visible_commands())
        yield _CommandMenuOverlay(commands, id="command-menu")

        yield Static(f"🤖 {self._config.agent_name}", id="title-bar")
        yield Static(f"💡 {self._config.agent_description}", id="agent-info")

        with Horizontal(id="main-container"):
            with Container(id="content-area"):
                yield RichLog(
                    id="output-panel",
                    highlight=True,
                    markup=True,
                    wrap=True,
                    auto_scroll=True,
                )

            if self._config.custom_widget is not None:
                with VerticalScroll(
                    id="custom-panel",
                    classes="visible" if self._custom_panel_visible else "",
                ):
                    yield Static(
                        f"[bold cyan]{self._config.custom_widget_title}[/]",
                        id="custom-title",
                    )
                    yield self._config.custom_widget

            with VerticalScroll(
                id="stats-panel",
                classes="visible" if self._stats_panel_visible else "",
            ):
                yield Static(
                    "[bold cyan]📊 Session Statistics[/]", id="stats-title"
                )
                yield Static("", id="stats-content")

        with Horizontal(id="input-container"):
            yield Input(
                placeholder="Enter query or /command…",
                id="query-input",
                suggester=_SlashSuggester([c.name for c in commands]),
            )
            yield Button("Send", variant="primary", id="send-button")
            yield Button("Clear", id="clear-button")

        yield Static("Ready", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#query-input", Input).focus()
        self._show_welcome()
        if self._stats_panel_visible:
            self._refresh_stats_display()
            self._stats_timer = self.set_interval(
                1.0, self._refresh_stats_display
            )

    # ── Welcome banner ─────────────────────────────────────────────

    def _show_welcome(self) -> None:
        out = self.query_one("#output-panel", RichLog)
        lines = [
            "[dim]• Type your query and press Enter[/]",
            "[dim]• /cancel or Ctrl+X — cancel running agent[/]",
            "[dim]• /save — save last response to a file[/]",
            "[dim]• /clear — clear output[/]",
            "[dim]• /reset — reset session[/]",
            "[dim]• /quit — exit[/]",
            "[dim]• /help — show this help[/]",
        ]
        if self._config.persona_provider is not None:
            lines.append("[dim]• /persona — switch persona[/]")
        if self._config.guideline_provider is not None:
            lines.append("[dim]• /guideline — pick guideline[/]")
        if self._config.custom_widget is not None:
            lines.append("[dim]• /viz — toggle custom visualization[/]")

        text = (
            f"[bold]{self._config.agent_name}[/]\n\n"
            f"{self._config.agent_description}\n\n"
            "[dim]Commands:[/]\n" + "\n".join(lines)
        )
        out.write(Panel(text, border_style="dim", padding=(1, 2)))

    # ── Slash command catalogue ────────────────────────────────────

    def _visible_commands(self) -> list[SlashCommand]:
        return [
            c
            for c in BUILTIN_COMMANDS
            if not (c.requires_persona and self._config.persona_provider is None)
            and not (
                c.requires_guideline and self._config.guideline_provider is None
            )
            and not (
                c.requires_custom_widget and self._config.custom_widget is None
            )
        ]

    # ── Input event handlers ───────────────────────────────────────

    async def on_input_submitted(self, _event: Input.Submitted) -> None:
        await self._process_input()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-button":
            await self._process_input()
        elif event.button.id == "clear-button":
            self.action_clear()

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "query-input":
            return
        value = event.value
        if value.startswith("/"):
            if len(value) == 1:
                self._show_command_menu()
            else:
                self._filter_command_menu(value)
        else:
            self._hide_command_menu()

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option_list.id != "command-menu-list":
            return
        command = event.option.id or ""
        query_input = self.query_one("#query-input", Input)
        query_input.value = command
        self._hide_command_menu()
        query_input.focus()
        await self._handle_slash_command(command)

    async def on_key(self, event: Any) -> None:
        query_input = self.query_one("#query-input", Input)

        if event.key == "escape" and self._command_menu_visible:
            self._hide_command_menu()
            query_input.focus()
            event.prevent_default()
            event.stop()
            return

        if self._command_menu_visible and event.key in ("down", "up"):
            option_list = self.query_one("#command-menu-list", OptionList)
            if not option_list.has_focus:
                option_list.focus()
                event.prevent_default()
                event.stop()
            return

        if event.key == "tab" and query_input.has_focus:
            if self._command_menu_visible:
                option_list = self.query_one("#command-menu-list", OptionList)
                if option_list.option_count > 0:
                    idx = option_list.highlighted or 0
                    option = option_list.get_option_at_index(idx)
                    if option.id:
                        query_input.value = option.id
                        query_input.cursor_position = len(option.id)
                        event.prevent_default()
                        event.stop()
                return

            current = query_input.value
            if current.startswith("/") and query_input.suggester is not None:
                suggestion = await query_input.suggester.get_suggestion(current)
                if suggestion:
                    query_input.value = suggestion
                    query_input.cursor_position = len(suggestion)
                    event.prevent_default()
                    event.stop()

    # ── Process input dispatcher ───────────────────────────────────

    async def _process_input(self) -> None:
        query_input = self.query_one("#query-input", Input)
        value = query_input.value.strip()

        if self._mode is not Mode.IDLE:
            self._handle_modal_input(query_input, value)
            return

        query_input.placeholder = "Enter query or /command…"
        if not value:
            return
        query_input.value = ""

        if value.startswith("/"):
            await self._handle_slash_command(value)
        elif self._is_processing:
            self._write_line(
                "[yellow]⚠️  Agent is running. Use Ctrl+X to cancel.[/]"
            )
        else:
            self._start_query(value)

    def _handle_modal_input(self, query_input: Input, value: str) -> None:
        """Dispatch *value* to the active modal flow and return to IDLE.

        ``AgentTUI`` uses a tiny state machine — the input box does
        triple duty as filename / persona-id / guideline-id picker
        depending on ``self._mode``. Centralising the "clear box, run
        handler, reset mode" loop here keeps :meth:`_process_input`
        focused on the IDLE → query path.
        """
        query_input.value = ""

        if self._mode is Mode.AWAITING_FILENAME:
            filename = value or str(self._mode_payload or "")
            if filename:
                self._do_save(filename)
        elif self._mode is Mode.AWAITING_PERSONA:
            self._do_select_persona(value)
        elif self._mode is Mode.AWAITING_GUIDELINE:
            self._do_select_guideline(value)

        self._mode = Mode.IDLE
        self._mode_payload = None
        query_input.placeholder = "Enter query or /command…"

    # ── Slash command dispatch ─────────────────────────────────────

    async def _handle_slash_command(self, command: str) -> None:
        cmd = command.lower().strip()
        handler = {
            "/quit": self.action_quit,
            "/cancel": self.action_cancel,
            "/clear": self.action_clear,
            "/reset": self.action_reset,
            "/help": self._show_welcome,
            "/save": self._begin_save,
            "/persona": self._begin_persona,
            "/guideline": self._begin_guideline,
            "/stats": self._toggle_stats_panel,
            "/viz": self._toggle_custom_panel,
        }.get(cmd.split()[0] if cmd else cmd)

        if handler is None:
            self._write_line(f"[yellow]Unknown command: {command}[/]")
            self._write_line("[dim]Type /help for available commands[/]")
            return
        handler()

    # ── Save flow ──────────────────────────────────────────────────

    def _begin_save(self) -> None:
        if not self._last_response:
            self._write_line("[yellow]No response to save yet[/]")
            return

        suggested = self._config.save_handler.suggest_filename(
            self._last_query, self._last_response
        )
        self._write_panel(
            f"[cyan]Save response to:[/]\n\n[bold]{suggested}[/]\n\n"
            "[dim]Press Enter to accept, or type a different filename[/]",
            border_style="cyan",
        )
        query_input = self.query_one("#query-input", Input)
        query_input.placeholder = f"Filename (or Enter for: {suggested})"
        query_input.value = suggested
        self._mode = Mode.AWAITING_FILENAME
        self._mode_payload = suggested

    def _do_save(self, filename: str) -> None:
        if self._last_response is None:
            return
        try:
            path = self._config.save_handler.save(self._last_response, filename)
        except Exception as exc:
            self._write_line(f"[red]Error saving file: {exc}[/]")
            self._update_status("Save failed")
            return
        self._write_line(
            f"[green]✓ Saved successfully![/]\n[cyan]Location:[/] [bold]{path}[/]"
        )
        self._update_status(f"Saved to {path.name}")

    # ── Persona flow ───────────────────────────────────────────────

    def _begin_persona(self) -> None:
        provider = self._config.persona_provider
        if provider is None:
            self._write_line("[yellow]Persona switching not available[/]")
            return

        ids = provider.list_ids()
        if not ids:
            self._write_line("[yellow]No personas found[/]")
            return
        current = provider.current_id()
        lines = "\n".join(
            f"  {'→' if pid == current else ' '} {i + 1}. {pid}"
            for i, pid in enumerate(ids)
        )
        self._write_panel(
            f"[cyan]Available Personas:[/]\n\n{lines}\n\n"
            "[dim]Type the number or name (Enter to cancel)[/]",
            border_style="cyan",
        )
        query_input = self.query_one("#query-input", Input)
        query_input.placeholder = "Persona number or name…"
        query_input.value = ""
        self._mode = Mode.AWAITING_PERSONA
        self._mode_payload = ids

    def _do_select_persona(self, selection: str) -> None:
        provider = self._config.persona_provider
        ids: list[str] = list(self._mode_payload or [])
        if not selection or provider is None:
            self._write_line("[dim]Persona switch cancelled[/]")
            return
        chosen = _resolve_selection(selection, ids)
        if chosen is None:
            self._write_line(f"[yellow]No persona matches: {selection}[/]")
            return
        try:
            provider.select(chosen)
        except Exception as exc:
            self._write_line(f"[red]Failed to switch persona: {exc}[/]")
            return
        self._write_line(f"[green]✓ Switched to persona: {chosen}[/]")
        self._update_status(f"Persona: {chosen}")

    # ── Guideline flow ─────────────────────────────────────────────

    def _begin_guideline(self) -> None:
        provider = self._config.guideline_provider
        if provider is None:
            self._write_line("[yellow]Guideline selection not available[/]")
            return

        ids = provider.list_ids()
        if not ids:
            self._write_line("[yellow]No guidelines found[/]")
            return
        current = provider.current_id()
        zero = (
            "  → 0. [None — clear current guideline]"
            if current is None
            else "    0. [None — clear current guideline]"
        )
        rest = "\n".join(
            f"  {'→' if gid == current else ' '} {i + 1}. {gid}"
            for i, gid in enumerate(ids)
        )
        self._write_panel(
            f"[cyan]Available Guidelines:[/]\n\n{zero}\n{rest}\n\n"
            "[dim]Type number/name; 0/none/clear removes (Enter to cancel)[/]",
            border_style="cyan",
        )
        query_input = self.query_one("#query-input", Input)
        query_input.placeholder = "Guideline number, name, or 0 to clear…"
        query_input.value = ""
        self._mode = Mode.AWAITING_GUIDELINE
        self._mode_payload = ids

    def _do_select_guideline(self, selection: str) -> None:
        provider = self._config.guideline_provider
        ids: list[str] = list(self._mode_payload or [])
        if not selection or provider is None:
            self._write_line("[dim]Guideline selection cancelled[/]")
            return
        if selection.lower() in ("0", "none", "clear"):
            try:
                provider.clear()
            except Exception as exc:
                self._write_line(f"[red]Failed to clear guideline: {exc}[/]")
                return
            self._write_line("[green]✓ Guideline cleared[/]")
            self._update_status("Guideline: None")
            return
        chosen = _resolve_selection(selection, ids)
        if chosen is None:
            self._write_line(f"[yellow]No guideline matches: {selection}[/]")
            return
        try:
            provider.select(chosen)
        except Exception as exc:
            self._write_line(f"[red]Failed to switch guideline: {exc}[/]")
            return
        self._write_line(f"[green]✓ Switched to guideline: {chosen}[/]")
        self._update_status(f"Guideline: {chosen}")

    # ── Stats & custom panels ──────────────────────────────────────

    def _toggle_stats_panel(self) -> None:
        panel = self.query_one("#stats-panel")
        self._stats_panel_visible = not self._stats_panel_visible
        if self._stats_panel_visible:
            panel.add_class("visible")
            self._refresh_stats_display()
            if self._stats_timer is None:
                self._stats_timer = self.set_interval(
                    1.0, self._refresh_stats_display
                )
        else:
            panel.remove_class("visible")
            if self._stats_timer is not None:
                self._stats_timer.stop()
                self._stats_timer = None

    def _toggle_custom_panel(self) -> None:
        if self._config.custom_widget is None:
            self._write_line(
                "[yellow]No custom visualization available[/]"
            )
            return
        panel = self.query_one("#custom-panel")
        self._custom_panel_visible = not self._custom_panel_visible
        if self._custom_panel_visible:
            panel.add_class("visible")
        else:
            panel.remove_class("visible")

    def _refresh_stats_display(self) -> None:
        content = self.query_one("#stats-content", Static)
        elapsed = datetime.now() - self._session_start
        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            dur = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            dur = f"{minutes}m {seconds}s"
        else:
            dur = f"{seconds}s"

        lines = [
            "\n[dim]Session Duration[/]",
            f"[bold cyan]{dur}[/]",
            "",
        ]
        for key, value in self._stats_snapshot.items():
            lines.append(f"[dim]{key}[/]")
            lines.append(f"[bold green]{value}[/]")
            lines.append("")
        content.update("\n".join(lines))

    # ── Worker: run agent ──────────────────────────────────────────

    def _start_query(self, query: str) -> None:
        self._current_worker = self._run_agent_worker(query)

    @work(exclusive=True, thread=True)  # type: ignore[untyped-decorator]
    async def _run_agent_worker(self, query: str) -> None:
        self._is_processing = True
        self._cancel_requested = False
        self._last_query = query
        self._query_history.append(query)
        self._update_status("[bright_cyan]Processing…[/]")
        self._post_query_panel(query)
        self._start_thinking()

        controls = TuiControls(self)
        try:
            if self._cancel_requested:
                raise asyncio.CancelledError("Cancelled by user")
            result = await self._invoke_runner(query, controls)
            self._last_response = result.text
            if result.stats is not None:
                self._stats_snapshot = dict(result.stats)
            self._post_response_panel(result.text)
            self._update_status("[bright_green]Ready[/]")
        except asyncio.CancelledError:
            self.call_from_thread(self._show_cancelled_panel)
            self.call_from_thread(
                lambda: self._update_status("[red]Cancelled[/]")
            )
        except Exception as exc:
            err = str(exc)
            self.call_from_thread(self._show_error_panel, err)
            self.call_from_thread(
                lambda: self._update_status("[bright_red]Error[/]")
            )
        finally:
            self.call_from_thread(self._on_query_finished)

    async def _invoke_runner(
        self, query: str, controls: TuiControls
    ) -> RunResult:
        """Invoke the runner protocol method or a plain callable."""
        runner: Any = self._config.runner
        if hasattr(runner, "run"):
            result = await runner.run(query, controls)
        else:
            # Allow plain awaitable callables for ergonomics.
            sig = inspect.signature(runner)
            kwargs: dict[str, Any] = {}
            if "controls" in sig.parameters:
                kwargs["controls"] = controls
            result = await runner(query, **kwargs)

        if isinstance(result, RunResult):
            return result
        if isinstance(result, str):
            return RunResult(text=result)
        text = getattr(result, "final_output", None) or str(result)
        stats = getattr(result, "stats", None)
        if stats is not None and not isinstance(stats, Mapping):
            stats = {
                attr: getattr(stats, attr)
                for attr in dir(stats)
                if not attr.startswith("_")
                and not callable(getattr(stats, attr))
            }
        return RunResult(text=text, stats=stats)

    # ── Posting from worker thread ─────────────────────────────────

    def _post_query_panel(self, query: str) -> None:
        def render() -> None:
            self._write_panel(
                f"[bold bright_white]{query}[/]",
                title=(
                    f"[bright_cyan]You[/] · "
                    f"{datetime.now().strftime('%H:%M')}"
                ),
                border_style="bright_cyan",
            )

        self.call_from_thread(render)

    def _post_response_panel(self, response: str) -> None:
        def render() -> None:
            out = self.query_one("#output-panel", RichLog)
            out.write(
                Panel(
                    Markdown(response),
                    title=(
                        f"[bright_green]{self._config.agent_name}[/] · "
                        f"{datetime.now().strftime('%H:%M')}"
                    ),
                    border_style="green",
                    padding=(1, 2),
                )
            )
            out.write("")

        self.call_from_thread(render)

    def _show_cancelled_panel(self) -> None:
        self._write_panel(
            "[bold red]Agent execution cancelled by user[/]\n\n"
            "[dim]The current operation has been interrupted.[/]",
            border_style="red",
        )

    def _show_error_panel(self, error: str) -> None:
        self._write_panel(
            f"[bold bright_red]Error: {error}[/]",
            border_style="bright_red",
        )

    def _on_query_finished(self) -> None:
        self._stop_thinking()
        self._is_processing = False
        self._current_worker = None
        self._cancel_requested = False
        with contextlib.suppress(Exception):
            self.query_one("#query-input", Input).focus()

    # ── Worker-side callbacks for TuiControls ──────────────────────

    def _post_progress(self, message: str) -> None:
        def render() -> None:
            out = self.query_one("#output-panel", RichLog)
            for line in message.split("\n"):
                if line.strip():
                    out.write(f"[bright_yellow]{line}[/]")
            out.refresh()

        with contextlib.suppress(Exception):
            self.call_from_thread(render)

    def _post_stats(self, stats: Mapping[str, Any]) -> None:
        def render() -> None:
            self._stats_snapshot = dict(stats)
            self._refresh_stats_display()

        with contextlib.suppress(Exception):
            self.call_from_thread(render)

    def _post_custom_widget(self, payload: Any) -> None:
        widget = self._config.custom_widget
        if widget is None or not hasattr(widget, "handle_payload"):
            return

        def render() -> None:
            widget.handle_payload(payload)

        with contextlib.suppress(Exception):
            self.call_from_thread(render)

    # ── Thinking spinner ───────────────────────────────────────────

    _SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def _start_thinking(self) -> None:
        def render() -> None:
            out = self.query_one("#output-panel", RichLog)
            out.write("[bright_yellow]Processing query…[/]")
            self._spinner_frame = 0
            self._spinner_timer = self.set_interval(0.1, self._tick_spinner)

        self.call_from_thread(render)

    def _stop_thinking(self) -> None:
        timer = getattr(self, "_spinner_timer", None)
        if timer is not None:
            timer.stop()
            self._spinner_timer = None

    def _tick_spinner(self) -> None:
        if not self._is_processing:
            return
        frame = self._SPINNER_FRAMES[
            self._spinner_frame % len(self._SPINNER_FRAMES)
        ]
        try:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update(f"[dim]{frame} Processing…[/]")
        except Exception:
            return
        self._spinner_frame += 1

    # ── Tiny helpers ───────────────────────────────────────────────

    def _show_command_menu(self) -> None:
        try:
            overlay = self.query_one("#command-menu", _CommandMenuOverlay)
            option_list = self.query_one("#command-menu-list", OptionList)
            option_list.clear_options()
            for cmd in overlay.commands:
                option_list.add_option(
                    Option(
                        f"{cmd.name}  [dim]{cmd.description}[/]",
                        id=cmd.name,
                    )
                )
            overlay.add_class("visible")
            self._command_menu_visible = True
        except Exception:
            pass

    def _hide_command_menu(self) -> None:
        try:
            self.query_one("#command-menu", _CommandMenuOverlay).remove_class(
                "visible"
            )
            self._command_menu_visible = False
        except Exception:
            pass

    def _filter_command_menu(self, value: str) -> None:
        try:
            overlay = self.query_one("#command-menu", _CommandMenuOverlay)
            option_list = self.query_one("#command-menu-list", OptionList)
            matches = [
                c for c in overlay.commands if c.name.startswith(value.lower())
            ]
            if not matches:
                self._hide_command_menu()
                return
            option_list.clear_options()
            for cmd in matches:
                option_list.add_option(
                    Option(
                        f"{cmd.name}  [dim]{cmd.description}[/]",
                        id=cmd.name,
                    )
                )
            overlay.add_class("visible")
            self._command_menu_visible = True
        except Exception:
            pass

    def _write_line(self, text: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#output-panel", RichLog).write(text)

    def _write_panel(
        self,
        text: str,
        *,
        title: str | None = None,
        border_style: str = "dim",
    ) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#output-panel", RichLog).write(
                Panel(
                    text,
                    title=title,
                    border_style=border_style,
                    padding=(1, 2),
                )
            )

    def _update_status(self, message: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#status-bar", Static).update(message)

    # ── Actions (bindings) ─────────────────────────────────────────

    def action_clear(self) -> None:
        try:
            out = self.query_one("#output-panel", RichLog)
            out.clear()
            self._show_welcome()
            self._update_status("Cleared")
        except Exception:
            pass

    def action_reset(self) -> None:
        self._query_history.clear()
        self._last_response = None
        self._last_query = None
        self.action_clear()
        self._update_status("Reset")

    def action_save(self) -> None:
        self._begin_save()

    def action_cancel(self) -> None:
        if not self._is_processing:
            self._write_line("[dim]No agent currently running[/]")
            return
        self._write_line("[bold red]🛑 Cancellation requested…[/]")
        self._write_line(
            "[dim]Note: the agent will stop after the current operation.[/]"
        )
        self._cancel_requested = True
        if self._current_worker is not None:
            with contextlib.suppress(Exception):
                self._current_worker.cancel()
        self._update_status("[red]Cancelling…[/]")

    def action_quit(self) -> None:
        if self._is_processing and self._current_worker is not None:
            with contextlib.suppress(Exception):
                self._current_worker.cancel()
        self.exit()


# ── Selection helpers ───────────────────────────────────────────────


def _resolve_selection(selection: str, ids: list[str]) -> str | None:
    """Resolve a user selection by index (1-based), exact name, or substring."""
    try:
        idx = int(selection) - 1
    except ValueError:
        pass
    else:
        return ids[idx] if 0 <= idx < len(ids) else None

    lower = selection.lower()
    exact = [i for i in ids if i.lower() == lower]
    if exact:
        return exact[0]
    partial = [i for i in ids if lower in i.lower()]
    if len(partial) == 1:
        return partial[0]
    return None


# ── Public entry point ──────────────────────────────────────────────


def run_agent_tui(config: TuiConfig) -> None:
    """Run an :class:`AgentTUI` to completion."""
    AgentTUI(config).run()


__all__ = [
    "AgentRunner",
    "AgentTUI",
    "BUILTIN_COMMANDS",
    "FileSaveHandler",
    "GuidelineProvider",
    "Mode",
    "PersonaProvider",
    "RunResult",
    "SaveHandler",
    "SlashCommand",
    "TuiConfig",
    "TuiControls",
    "run_agent_tui",
]


# Allow type-checkers to see Screen is referenced (used in module annotation
# scope by Textual when running) without inflating runtime usage.
_ = Screen  # noqa: F841
