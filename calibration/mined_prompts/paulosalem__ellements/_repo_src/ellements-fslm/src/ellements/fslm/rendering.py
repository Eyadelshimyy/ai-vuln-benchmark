"""Rich renderers for human-facing FSLM output."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from rich import box
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.json import JSON
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .definition import MachineDefinition
from .models import FSLMEvent, MachineSnapshot, MachineSpec, StepResult

ColorMode = Literal["auto", "always", "never"]

THEME = Theme(
    {
        "brand": "bold #6C63FF",
        "brand.dim": "#4A42B0",
        "accent": "bold #56D6C1",
        "accent.dim": "#3BA899",
        "amber": "bold #FFB347",
        "amber.dim": "#CC8F39",
        "danger": "bold #FF6B6B",
        "muted": "#A8A4CE",
        "success": "bold #56D6C1",
        "state": "bold #56D6C1",
        "transition": "bold #FFB347",
        "blocked": "bold #FF6B6B",
    }
)


def make_console(color: ColorMode = "auto") -> Console:
    """Create the standard FSLM Rich console."""

    return Console(
        theme=THEME,
        force_terminal=True if color == "always" else None,
        no_color=color == "never",
    )


def render_validation(
    spec: MachineSpec,
    *,
    console: Console | None = None,
    color: ColorMode = "auto",
) -> None:
    """Render a successful machine validation."""

    out = console or make_console(color)
    _app_header(
        "FSLM",
        {"Machine": spec.name, "States": str(len(spec.states)), "Transitions": str(len(spec.transitions))},
        icon="◆",
        border_style="brand",
        console=out,
    )
    out.print(
        Panel(
            "[success]✓ Machine definition is valid[/]\n"
            "[muted]The graph, declared states, transitions, and runtime policy loaded successfully.[/]",
            title="[success]Ready[/]",
            title_align="left",
            border_style="accent.dim",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def render_snapshot(
    snapshot: MachineSnapshot,
    *,
    spec: MachineSpec | None = None,
    title: str = "FSLM Snapshot",
    console: Console | None = None,
    color: ColorMode = "auto",
) -> None:
    """Render a snapshot in a compact, human-readable layout."""

    out = console or make_console(color)
    _app_header(
        title,
        {
            "Machine": snapshot.machine_name,
            "Instance": snapshot.machine_id,
            "Step": str(snapshot.step_index),
        },
        icon="◆",
        border_style="brand",
        console=out,
    )
    out.print(
        Panel(
            _state_card(
                snapshot.current_state,
                spec,
                label="Current state",
                border_style="accent.dim",
            ),
            border_style="brand.dim",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    if snapshot.variables:
        out.print(_mapping_panel("Snapshot variables", snapshot.variables, border_style="brand.dim"))


def render_step(
    result: StepResult,
    *,
    definition: MachineDefinition | None = None,
    event: FSLMEvent | None = None,
    state_dir: Path | None = None,
    console: Console | None = None,
    color: ColorMode = "auto",
) -> None:
    """Render one machine step as a polished state-transition view."""

    out = console or make_console(color)
    spec = definition.spec if definition is not None else None
    metadata = {
        "Machine": result.new_snapshot.machine_name,
        "Instance": result.new_snapshot.machine_id,
        "Step": str(result.new_snapshot.step_index),
        "Event": event.type if event is not None else result.event_id[:8],
    }
    _app_header(
        "FSLM State Movement",
        metadata,
        icon="◆",
        border_style=_status_border(result.status),
        console=out,
    )
    out.print(_movement_panel(result, spec))
    out.print(_decision_panel(result))
    _render_outputs(out, result)
    _render_actions(out, result)
    if result.violations:
        out.print(_list_panel("Violations", result.violations, border_style="danger"))
    out.print(_trace_panel(result, event))
    footer: dict[str, str | int | float] = {
        "Status": result.status,
        "Transition": result.selected_transition or "none",
        "State": result.target_state,
    }
    if state_dir is not None and not state_dir.name.startswith("tmp."):
        footer["Saved"] = _short_path(state_dir)
    _stats_footer(footer, border_style="brand.dim", console=out)


def _movement_panel(result: StepResult, spec: MachineSpec | None) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(ratio=5)
    grid.add_column(ratio=3)
    grid.add_column(ratio=5)
    grid.add_row(
        _state_card(result.source_state, spec, label="From", border_style="brand.dim"),
        _transition_card(result),
        _state_card(result.target_state, spec, label="To", border_style=_status_border(result.status)),
    )
    return Panel(
        grid,
        title="[bold white]Machine movement[/]",
        title_align="left",
        border_style=_status_border(result.status),
        box=box.DOUBLE,
        padding=(1, 2),
    )


def _state_card(
    state_name: str,
    spec: MachineSpec | None,
    *,
    label: str,
    border_style: str,
) -> Panel:
    title = Text(label, style="muted")
    state = spec.states.get(state_name) if spec is not None else None
    lines: list[RenderableType] = [
        Text(_humanize(state_name), style="state"),
        Text(state_name, style="muted"),
    ]
    objective = state.objective if state is not None else ""
    if objective:
        lines.append(Text(objective, style="white"))
    if state is not None and state.terminal:
        lines.append(Text("terminal state", style="amber"))
    return Panel(
        Group(*lines),
        title=title,
        title_align="left",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _transition_card(result: StepResult) -> Panel:
    selected = result.selected_transition or "No transition selected"
    body = Table.grid(expand=True)
    body.add_column(justify="center")
    body.add_row(Text("→", style="bold white"))
    body.add_row(Text(_humanize(selected), style="transition"))
    body.add_row(Text(selected, style="muted"))
    body.add_row(_status_badge(result.status))
    return Panel(
        Align.center(body),
        title="[muted]Transition[/]",
        title_align="left",
        border_style=_status_border(result.status),
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _decision_panel(result: StepResult) -> Panel:
    if not result.guard_results and not result.invariant_results:
        return Panel(
            "[muted]No guards or invariants were evaluated for this step.[/]",
            title="[bold white]Decision[/]",
            title_align="left",
            border_style="brand.dim",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style="brand.dim",
        show_header=True,
        header_style="#A8A4CE",
        expand=True,
    )
    table.add_column("Check", style="white", ratio=3)
    table.add_column("Decision", ratio=2)
    table.add_column("Confidence", justify="right", ratio=1)
    table.add_column("Notes", style="muted", ratio=4)
    for decision in [*result.guard_results, *result.invariant_results]:
        table.add_row(
            decision.id,
            "[success]✓ allowed[/]" if decision.allowed else "[blocked]✗ blocked[/]",
            f"{decision.confidence:.0%}",
            _decision_notes(decision.evidence, decision.uncertainties, decision.alternatives),
        )
    return Panel(
        table,
        title="[bold white]Decision path[/]",
        title_align="left",
        border_style=_status_border(result.status),
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _render_outputs(console: Console, result: StepResult) -> None:
    if not result.outputs:
        console.print(
            Panel(
                "[muted]This transition emitted no outputs.[/]",
                title="[bold white]Outputs[/]",
                title_align="left",
                border_style="brand.dim",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        return
    panels = []
    for output in result.outputs:
        parts: list[RenderableType] = []
        if output.description:
            parts.append(Text(output.description, style="muted"))
        parts.append(_mapping_table(output.payload))
        if output.destination:
            parts.append(Text(f"Destination: {output.destination}", style="accent"))
        panels.append(
            Panel(
                Group(*parts),
                title=f"[accent]{output.type}[/]",
                title_align="left",
                border_style="accent.dim",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
    console.print(
        Panel(
            Group(*panels),
            title="[bold white]Emitted outputs[/]",
            title_align="left",
            border_style="accent.dim",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _render_actions(console: Console, result: StepResult) -> None:
    if not result.actions:
        return
    table = Table(
        box=box.SIMPLE,
        border_style="brand.dim",
        header_style="#A8A4CE",
        expand=True,
    )
    table.add_column("Action")
    table.add_column("Tool")
    table.add_column("Status")
    table.add_column("Message", style="muted")
    for action in result.actions:
        table.add_row(
            action.action_name,
            action.tool,
            _action_status(action.status),
            action.message,
        )
    console.print(
        Panel(
            table,
            title="[bold white]Actions[/]",
            title_align="left",
            border_style="brand.dim",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _trace_panel(result: StepResult, event: FSLMEvent | None) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(style="muted", no_wrap=True, ratio=1)
    table.add_column(style="white", ratio=4)
    if event is not None:
        table.add_row("Event", f"{event.type}  [muted]{event.id[:8]}[/]")
    candidates = result.trace.get("candidate_transitions", [])
    legal = result.trace.get("legal_transitions", [])
    if isinstance(candidates, list):
        table.add_row("Candidates", _join_names(candidates))
    if isinstance(legal, list):
        table.add_row("Legal", _join_names(legal))
    table.add_row("Snapshot", result.new_snapshot.id[:8])
    return Panel(
        table,
        title="[bold white]Trace[/]",
        title_align="left",
        border_style="brand.dim",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _mapping_panel(title: str, mapping: Mapping[str, Any], *, border_style: str) -> Panel:
    return Panel(
        _mapping_table(mapping),
        title=f"[bold white]{title}[/]",
        title_align="left",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _mapping_table(mapping: Mapping[str, Any]) -> Table:
    table = Table.grid(expand=True)
    table.add_column(style="muted", no_wrap=True, ratio=1)
    table.add_column(style="white", ratio=4)
    for key, value in mapping.items():
        table.add_row(_humanize(str(key)), _value_renderable(value))
    return table


def _list_panel(title: str, values: list[str], *, border_style: str) -> Panel:
    body = "\n".join(f"• {value}" for value in values)
    return Panel(
        body,
        title=f"[bold white]{title}[/]",
        title_align="left",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _value_renderable(value: Any) -> RenderableType:
    if isinstance(value, str):
        return Text(value)
    if isinstance(value, bool):
        return Text("true" if value else "false", style="success" if value else "danger")
    if isinstance(value, int | float) or value is None:
        return Text(str(value), style="accent")
    summary = _compact_summary(value)
    if summary is not None:
        return Text(summary)
    return Padding(
        JSON(json.dumps(value, ensure_ascii=False)),
        (0, 0, 0, 0),
    )


def _compact_summary(value: Any) -> str | None:
    if isinstance(value, Mapping):
        if "news" in value and isinstance(value["news"], Mapping):
            news = value["news"]
            title = str(news.get("title", "news item"))
            tone = news.get("tone")
            severity = news.get("severity")
            parts = [title]
            if tone is not None:
                parts.append(f"tone={tone}")
            if severity is not None:
                parts.append(f"severity={severity}")
            return " · ".join(parts)
        simple: list[str] = []
        for key, item in value.items():
            if len(simple) >= 4:
                simple.append("…")
                break
            if isinstance(item, str | int | float | bool) or item is None:
                simple.append(f"{key}={item}")
        if simple and len(simple) >= len(value):
            return " · ".join(simple)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " · ".join(value)
    return None


def _decision_notes(
    evidence: list[str],
    uncertainties: list[str],
    alternatives: list[str],
) -> str:
    notes = []
    if evidence:
        notes.append("evidence: " + "; ".join(evidence))
    if uncertainties:
        notes.append("uncertainty: " + "; ".join(uncertainties))
    if alternatives:
        notes.append("alternatives: " + "; ".join(alternatives))
    return " · ".join(notes) if notes else "—"


def _status_badge(status: str) -> Text:
    text = Text()
    if status == "transitioned":
        text.append("✓ transitioned", style="success")
    elif status == "blocked":
        text.append("✗ blocked", style="blocked")
    else:
        text.append("• no transition", style="amber")
    return text


def _action_status(status: str) -> str:
    if status in {"executed", "planned"}:
        return f"[success]{status}[/]"
    if status == "blocked":
        return f"[amber]{status}[/]"
    return f"[danger]{status}[/]"


def _status_border(status: str) -> str:
    if status == "transitioned":
        return "accent"
    if status == "blocked":
        return "danger"
    return "amber"


def _join_names(values: list[Any]) -> str:
    if not values:
        return "—"
    return "  ·  ".join(str(value) for value in values)


def _humanize(value: str) -> str:
    if not value:
        return "—"
    return value.replace("_", " ").replace("-", " ").strip().capitalize()


def _short_path(path: Path) -> str:
    if path.name.startswith("tmp."):
        return path.name
    text = str(path)
    home = str(Path.home())
    if text.startswith(f"{home}/"):
        text = f"~/{text[len(home) + 1:]}"
    if len(text) <= 56:
        return text
    return f"…{text[-55:]}"


def _app_header(
    title: str,
    subtitle_items: Mapping[str, str],
    *,
    icon: str,
    border_style: str,
    console: Console,
) -> None:
    components = _cli_components()
    components.app_header(
        title,
        dict(subtitle_items),
        icon=icon,
        border_style=border_style,
        console=console,
    )


def _stats_footer(
    stats: Mapping[str, str | int | float],
    *,
    border_style: str,
    console: Console,
) -> None:
    components = _cli_components()
    components.stats_footer(
        dict(stats),
        border_style=border_style,
        console=console,
    )


def _cli_components() -> ModuleType:
    return importlib.import_module("ellements.cli.components")


__all__ = [
    "ColorMode",
    "make_console",
    "render_snapshot",
    "render_step",
    "render_validation",
]
