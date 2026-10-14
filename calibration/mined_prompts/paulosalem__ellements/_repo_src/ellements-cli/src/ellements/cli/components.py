"""Low-level Rich building blocks for CLI applications.

Pure functions with no shared state. Each accepts an optional *console*
parameter so callers can direct output to stderr (default) or stdout.

Usage::

    from ellements.cli import app_header, key_value_table

    app_header("My App", {"Model": "gpt-4o"}, icon="🚀")
    key_value_table({"topic": "cats"}, title="Variables")
"""

from __future__ import annotations

import json as _json
from collections.abc import Sequence
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Default console writes to stderr so stdout stays clean for data output
_default_console = Console(stderr=True)

# ── Issue / severity styling ──────────────────────────────────────────

ISSUE_STYLES: dict[str, tuple[str, str]] = {
    "warning":    ("⚠️ ", "yellow"),
    "error":      ("❌ ", "red"),
    "suggestion": ("💡 ", "cyan"),
    "info":       ("ℹ️ ", "dim"),
}


# ── Building-block functions ──────────────────────────────────────────

def app_header(
    title: str,
    subtitle_items: dict[str, str] | None = None,
    *,
    icon: str = "",
    border_style: str = "bright_blue",
    console: Console | None = None,
) -> None:
    """Render a styled application header panel.

    Args:
        title: Application name.
        subtitle_items: Key-value metadata pairs shown below the title
            (e.g. ``{"Model": "gpt-4o", "Spec": "file.md"}``).
        icon: Emoji or symbol prepended to the title.
        border_style: Rich style for the panel border.
        console: Console instance (defaults to stderr).
    """
    c = console or _default_console
    display_title = f"{icon} {title}".strip() if icon else title
    title_text = Text(display_title, style="bold bright_white")

    if subtitle_items:
        parts = [
            f"[dim]{k}:[/dim] [bright_cyan]{v}[/]"
            for k, v in subtitle_items.items()
        ]
        subtitle = "  ·  ".join(parts)
    else:
        subtitle = ""

    c.print(Panel(
        subtitle,
        title=title_text,
        title_align="left",
        border_style=border_style,
        box=box.DOUBLE,
        padding=(0, 2),
    ))


def key_value_table(
    data: dict[str, Any],
    *,
    title: str = "",
    name_style: str = "bright_cyan",
    value_style: str = "white",
    console: Console | None = None,
) -> None:
    """Render a two-column key→value table.

    Booleans are auto-formatted as colored ``true``/``false``.

    Args:
        data: Mapping of names to values.
        title: Optional table title.
        name_style: Rich style for the name column.
        value_style: Rich style for the value column.
        console: Console instance (defaults to stderr).
    """
    if not data:
        return
    c = console or _default_console

    table = Table(
        title=title or None,
        title_style="bold",
        box=box.ROUNDED,
        border_style="dim",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Name", style=name_style, no_wrap=True)
    table.add_column("Value", style=value_style)

    for key, val in data.items():
        if isinstance(val, bool):
            val_str = "[green]true[/]" if val else "[red]false[/]"
        else:
            val_str = str(val)
        table.add_row(f"[bold]{key}[/]", val_str)

    c.print(table)
    c.print()


def section_rule(
    title: str,
    *,
    style: str = "bright_green",
    console: Console | None = None,
) -> None:
    """Render a horizontal rule with a bold section title."""
    c = console or _default_console
    c.print(Rule(f"[bold]{title}", style=style))
    c.print()


def stats_footer(
    stats: dict[str, str | int | float],
    *,
    border_style: str = "dim",
    console: Console | None = None,
) -> None:
    """Render a compact stats panel (key·value pairs on one line)."""
    c = console or _default_console
    parts = [
        f"[dim]{k}:[/] [cyan]{v}[/]"
        for k, v in stats.items()
    ]
    c.print(Panel(
        "  ·  ".join(parts),
        border_style=border_style,
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def issue_line(
    issue_type: str,
    message: str,
    *,
    console: Console | None = None,
) -> None:
    """Print a single issue line with icon and color."""
    c = console or _default_console
    icon, style = ISSUE_STYLES.get(issue_type, ("ℹ️ ", "dim"))
    c.print(f"  {icon}[{style}]{issue_type.upper()}[/]: {message}")


def issues_block(
    issues: Sequence[dict[str, str]],
    *,
    title: str = "Issues",
    rule_style: str = "yellow",
    console: Console | None = None,
) -> None:
    """Print a section rule followed by a list of issues."""
    if not issues:
        return
    c = console or _default_console
    section_rule(title, style=rule_style, console=c)
    for issue in issues:
        issue_line(issue.get("type", "info"), issue.get("message", ""), console=c)
    c.print()


def success(message: str, *, console: Console | None = None) -> None:
    """Print a green success line."""
    (console or _default_console).print(f"[green]✓[/] {message}")


def error(message: str, *, console: Console | None = None) -> None:
    """Print a red error line."""
    (console or _default_console).print(f"[red]✗[/] {message}")


def warning(message: str, *, console: Console | None = None) -> None:
    """Print a yellow warning line."""
    (console or _default_console).print(f"[yellow]⚠[/] {message}")


def step_log(
    message: str,
    *,
    position: str = "mid",
    console: Console | None = None,
) -> None:
    """Print a tree-style step log line.

    Args:
        message: The message to display (may contain Rich markup).
        position: ``"first"`` (├─), ``"mid"`` (├─), ``"last"`` (└─),
            or ``"cont"`` (│ ) for continuation lines.
        console: Console instance.
    """
    c = console or _default_console
    prefixes = {
        "first": "[dim]├─[/]",
        "mid":   "[dim]├─[/]",
        "last":  "[dim]└─[/]",
        "cont":  "[dim]│ [/]",
    }
    prefix = prefixes.get(position, prefixes["mid"])
    c.print(f"   {prefix} {message}")


def result_markdown(
    content: str,
    *,
    title: str = "Result",
    rule_style: str = "bright_green",
    console: Console | None = None,
    output_console: Console | None = None,
) -> None:
    """Print Markdown content preceded by a section rule.

    The rule goes to *console* (stderr), the content to *output_console*
    (stdout) so piping works correctly.
    """
    c = console or _default_console
    out = output_console or Console()
    section_rule(title, style=rule_style, console=c)
    out.print(Markdown(content))
    c.print()


def result_json(
    data: Any,
    *,
    output_console: Console | None = None,
) -> None:
    """Print JSON data to stdout.

    Emits the JSON verbatim — wrapping is disabled and Rich markup/emoji
    interpretation is turned off — so downstream consumers (other CLIs,
    GUIs, scripts piping the output) can always parse the result with a
    strict JSON parser. Wrapping would otherwise inject literal newlines
    inside string values when stdout's width is narrower than the longest
    string, breaking JSON conformance per RFC 8259 §7.
    """
    out = output_console or Console()
    out.print(
        _json.dumps(data, indent=2, ensure_ascii=False),
        highlight=False,
        markup=False,
        emoji=False,
        soft_wrap=True,
    )


# ── Chat components ───────────────────────────────────────────────────

# Role styles: (label, label_style, border_style, content_style)
ROLE_STYLES: dict[str, tuple[str, str, str, str]] = {
    "user":      ("You", "bold bright_white", "bright_blue", "white"),
    "assistant": ("Assistant", "bold bright_green", "green", ""),
    "system":    ("System", "bold yellow", "yellow", "dim"),
}


def chat_message(
    role: str,
    content: str,
    *,
    label: str | None = None,
    console: Console | None = None,
    output_console: Console | None = None,
) -> None:
    """Render a styled chat message.

    - **user** messages: compact label + text.
    - **assistant** messages: bordered panel with Markdown rendering
      sent to *output_console* (stdout).
    - **system** messages: dimmed panel.

    Args:
        role: ``"user"``, ``"assistant"``, or ``"system"``.
        content: The message text (Markdown is rendered for assistant).
        label: Override the default role label.
        console: Console for chrome (stderr).
        output_console: Console for assistant content (stdout).
    """
    c = console or _default_console
    out = output_console or Console()
    style_info = ROLE_STYLES.get(role, ROLE_STYLES["assistant"])
    role_label = label or style_info[0]
    label_style = style_info[1]
    border_style = style_info[2]

    if role == "user":
        c.print(f"\n[{label_style}]{role_label}[/] [dim]›[/] {content}")
    elif role == "assistant":
        c.print()
        out.print(Panel(
            Markdown(content),
            title=f"[{label_style}]{role_label}[/]",
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        ))
    else:
        c.print(Panel(
            content,
            title=f"[{label_style}]{role_label}[/]",
            title_align="left",
            border_style=border_style,
            box=box.SIMPLE,
            padding=(0, 2),
        ))


def chat_input_prompt(
    label: str = "You",
    *,
    style: str = "bold bright_white",
) -> str:
    """Return a styled prompt string for ``console.input()``.

    Usage::

        user_input = console.input(chat_input_prompt())
    """
    return f"\n[{style}]{label}[/] [dim]›[/] "


def tool_use_badge(
    tool_name: str,
    *,
    console: Console | None = None,
) -> None:
    """Print an inline tool-use notification."""
    c = console or _default_console
    c.print(f"   [dim]🔧 Using[/] [bright_cyan]{tool_name}[/][dim]…[/]")


def welcome_banner(
    title: str,
    subtitle_items: dict[str, str] | None = None,
    *,
    commands: dict[str, str] | None = None,
    icon: str = "",
    border_style: str = "bright_blue",
    console: Console | None = None,
) -> None:
    """Render a welcome banner for interactive CLI apps.

    Shows the app title, metadata, and optional command hints.

    Args:
        title: Application name.
        subtitle_items: Key-value pairs (e.g. model, mode).
        commands: Command hints (e.g. ``{"quit": "Exit", "/help": "Help"}``).
        icon: Emoji prepended to title.
        border_style: Panel border style.
        console: Console instance (stderr).
    """
    c = console or _default_console
    display_title = f"{icon} {title}".strip() if icon else title
    title_text = Text(display_title, style="bold bright_white")

    lines: list[str] = []
    if subtitle_items:
        parts = [
            f"[dim]{k}:[/] [bright_cyan]{v}[/]"
            for k, v in subtitle_items.items()
        ]
        lines.append("  ".join(parts))

    if commands:
        if lines:
            lines.append("")
        cmd_parts = [
            f"  [dim bold]{k}[/]  [dim]{v}[/]"
            for k, v in commands.items()
        ]
        lines.append("[dim]Commands:[/]")
        lines.extend(cmd_parts)

    c.print(Panel(
        "\n".join(lines) if lines else "",
        title=title_text,
        title_align="left",
        border_style=border_style,
        box=box.DOUBLE,
        padding=(1, 2),
    ))
    c.print()
