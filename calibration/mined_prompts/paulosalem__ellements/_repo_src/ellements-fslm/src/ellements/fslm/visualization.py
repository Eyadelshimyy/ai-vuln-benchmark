"""Mermaid rendering for `MachineSpec`."""

from __future__ import annotations

from .models import MachineSpec


def to_mermaid(spec: MachineSpec) -> str:
    """Render a machine spec as a Mermaid state diagram."""
    lines = ["stateDiagram-v2", f"    [*] --> {spec.initial}"]
    for state in spec.states.values():
        if state.terminal:
            lines.append(f"    {state.name} --> [*]")
    for transition in spec.transitions.values():
        label = transition.name
        if transition.weight is not None:
            label = f"{label} ({transition.weight:g})"
        if transition.kind == "recovery":
            label = f"recovery: {label}"
        lines.append(f"    {transition.source} --> {transition.target}: {label}")
    return "\n".join(lines) + "\n"


__all__ = ["to_mermaid"]

