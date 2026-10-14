"""Attack-path / data-flow graph for investigation priority. Not an exploit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowNode:
    kind: str
    name: str
    detail: str = ""


@dataclass
class AttackPathGraph:
    nodes: list[FlowNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)

    def add_path(self, names: tuple[str, ...], kinds: tuple[str, ...] | None = None) -> None:
        start = len(self.nodes)
        for index, name in enumerate(names):
            kind = kinds[index] if kinds and index < len(kinds) else "step"
            self.nodes.append(FlowNode(kind=kind, name=name))
            if index:
                self.edges.append((start + index - 1, start + index))

    def snapshot(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"kind": node.kind, "name": node.name, "detail": node.detail} for node in self.nodes
            ],
            "edges": [{"from": src, "to": dst} for src, dst in self.edges],
            "not_an_exploit": True,
        }
