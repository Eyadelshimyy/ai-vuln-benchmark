"""Natural-language spec helpers.

The helpers intentionally use the neutral name ``nl`` rather than ``llm``:
LLMs are one implementation strategy, but human judges, rules plus retrieval,
or other semantic evaluators can satisfy the same contracts.
"""

from __future__ import annotations

from typing import Any

from .models import ActionSpec, GuardSpec, InvariantSpec, OutputSpec


def guard(
    id: str,
    text: str,
    *,
    min_confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> GuardSpec:
    """Declare a natural-language transition guard."""
    return GuardSpec(
        id=id,
        kind="nl",
        text=text,
        min_confidence=min_confidence,
        metadata=metadata or {},
    )


def invariant(
    id: str,
    text: str,
    *,
    severity: str = "error",
    min_confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> InvariantSpec:
    """Declare a natural-language state invariant."""
    return InvariantSpec(
        id=id,
        kind="nl",
        text=text,
        severity=severity,  # type: ignore[arg-type]
        min_confidence=min_confidence,
        metadata=metadata or {},
    )


def action(
    tool: str,
    text: str,
    *,
    name: str | None = None,
    requires_safety_check: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ActionSpec:
    """Declare a natural-language-planned tool action."""
    return ActionSpec(
        name=name or tool,
        kind="nl",
        tool=tool,
        text=text,
        requires_approval=requires_safety_check,
        metadata=metadata or {},
    )


def output(
    type: str,
    text: str,
    *,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> OutputSpec:
    """Declare a natural-language-produced output."""
    return OutputSpec(
        type=type,
        kind="nl",
        text=text,
        description=description,
        metadata=metadata or {},
    )


__all__ = ["action", "guard", "invariant", "output"]
