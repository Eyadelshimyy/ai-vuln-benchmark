"""Observer protocol and built-in observers for FSLM runtime events."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class FSLMEventRecord(BaseModel):
    """One machine-level observer event."""

    model_config = ConfigDict(extra="forbid")

    type: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    machine_id: str
    step_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class FSLMObserver(Protocol):
    """Receives machine-level FSLM events."""

    async def on_event(self, event: FSLMEventRecord) -> None:
        """Handle one FSLM event."""


class RecordingFSLMObserver:
    """In-memory observer useful for tests."""

    def __init__(self) -> None:
        self.events: list[FSLMEventRecord] = []

    async def on_event(self, event: FSLMEventRecord) -> None:
        self.events.append(event)


class JsonlFSLMObserver:
    """Append FSLM observer events as JSON lines."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def on_event(self, event: FSLMEventRecord) -> None:
        line = json.dumps(event.model_dump(mode="json"), separators=(",", ":")) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)


class RichConsoleFSLMObserver:
    """Pretty console observer for watching machines run."""

    def __init__(self) -> None:
        from rich.console import Console

        self._console = Console()

    async def on_event(self, event: FSLMEventRecord) -> None:
        if event.type == "EventReceived":
            self._console.print(
                f"[bold cyan]event[/] {event.payload.get('event_type')} "
                f"[dim]in {event.payload.get('state')}[/]"
            )
        elif event.type == "TransitionSelected":
            self._console.print(
                f"[bold green]transition[/] {event.payload.get('transition')} "
                f"[dim]{event.payload.get('source')} -> "
                f"{event.payload.get('target')}[/]"
            )
        elif event.type == "GuardEvaluated":
            status = "ok" if event.payload.get("allowed") else "blocked"
            self._console.print(
                f"  [magenta]guard[/] {event.payload.get('id')}: {status} "
                f"[dim]confidence={event.payload.get('confidence')}[/]"
            )
        elif event.type == "InvariantChecked":
            status = "ok" if event.payload.get("allowed") else "violation"
            self._console.print(
                f"  [yellow]invariant[/] {event.payload.get('id')}: {status}"
            )
        elif event.type == "StepCompleted":
            self._console.print(
                f"[bold blue]step[/] {event.payload.get('status')} "
                f"[dim]state={event.payload.get('state')}[/]"
            )


__all__ = [
    "FSLMEventRecord",
    "FSLMObserver",
    "JsonlFSLMObserver",
    "RecordingFSLMObserver",
    "RichConsoleFSLMObserver",
]

