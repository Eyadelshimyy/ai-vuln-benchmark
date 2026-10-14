"""In-memory and local-file persistence for FSLM runtime records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import FSLMEvent, MachineSnapshot, StepResult


class InMemoryFSLMStore:
    """Lightweight in-memory store for snapshots/events/results/traces."""

    def __init__(self) -> None:
        self.snapshot: MachineSnapshot | None = None
        self.events: list[FSLMEvent] = []
        self.results: list[StepResult] = []
        self.traces: list[dict[str, Any]] = []

    async def save_snapshot(self, snapshot: MachineSnapshot) -> None:
        self.snapshot = snapshot

    async def load_snapshot(self) -> MachineSnapshot | None:
        return self.snapshot

    async def append_event(self, event: FSLMEvent) -> None:
        self.events.append(event)

    async def append_result(self, result: StepResult) -> None:
        self.results.append(result)

    async def append_trace(self, trace: dict[str, Any]) -> None:
        self.traces.append(trace)


class LocalFSLMStore:
    """Local JSON/JSONL store using the `.fslm/` layout."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def save_snapshot(self, snapshot: MachineSnapshot) -> None:
        self._write_json(self.root / "snapshot.json", snapshot.model_dump(mode="json"))

    async def load_snapshot(self) -> MachineSnapshot | None:
        path = self.root / "snapshot.json"
        if not path.is_file():
            return None
        return MachineSnapshot.model_validate(json.loads(path.read_text("utf-8")))

    async def append_event(self, event: FSLMEvent) -> None:
        self._append_jsonl(self.root / "events.jsonl", event.model_dump(mode="json"))

    async def append_result(self, result: StepResult) -> None:
        self._append_jsonl(self.root / "results.jsonl", result.model_dump(mode="json"))

    async def append_trace(self, trace: dict[str, Any]) -> None:
        self._append_jsonl(self.root / "traces.jsonl", trace)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, sort_keys=True) + "\n")


__all__ = ["InMemoryFSLMStore", "LocalFSLMStore"]
