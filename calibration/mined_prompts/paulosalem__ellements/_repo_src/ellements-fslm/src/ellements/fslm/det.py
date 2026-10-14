"""Deterministic helper namespace for FSLM definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import ActionSpec, EffectSpec, GuardSpec, InvariantSpec, OutputSpec


@dataclass(frozen=True)
class BoundGuard:
    spec: GuardSpec
    fn: Callable[..., Any]


@dataclass(frozen=True)
class BoundInvariant:
    spec: InvariantSpec
    fn: Callable[..., Any]


@dataclass(frozen=True)
class BoundOutput:
    spec: OutputSpec
    fn: Callable[..., Any]


@dataclass(frozen=True)
class BoundAction:
    spec: ActionSpec
    fn: Callable[..., Any] | None = None
    argument_factory: Callable[..., Any] | None = None


@dataclass(frozen=True)
class BoundEffect:
    spec: EffectSpec
    fn: Callable[..., Any]


@dataclass(frozen=True)
class Increment:
    path: str
    by: int | float = 1


def guard(
    id: str,
    fn: Callable[..., Any] | None = None,
    *,
    ref: str | None = None,
    args: Mapping[str, Any] | None = None,
    min_confidence: float | None = None,
) -> GuardSpec | BoundGuard:
    """Declare a deterministic guard."""
    spec = GuardSpec(
        id=id,
        kind="deterministic",
        ref=ref or id,
        args=dict(args or {}),
        min_confidence=min_confidence,
    )
    return BoundGuard(spec, fn) if fn is not None else spec


def invariant(
    id: str,
    fn: Callable[..., Any] | None = None,
    *,
    ref: str | None = None,
    args: Mapping[str, Any] | None = None,
    severity: str = "error",
) -> InvariantSpec | BoundInvariant:
    """Declare a deterministic invariant."""
    spec = InvariantSpec(
        id=id,
        kind="deterministic",
        ref=ref or id,
        args=dict(args or {}),
        severity=severity,  # type: ignore[arg-type]
    )
    return BoundInvariant(spec, fn) if fn is not None else spec


def output(
    type: str,
    fn: Callable[..., Any] | None = None,
    *,
    ref: str | None = None,
    args: Mapping[str, Any] | None = None,
    description: str = "",
) -> OutputSpec | BoundOutput:
    """Declare a deterministic output."""
    spec = OutputSpec(
        type=type,
        kind="deterministic",
        ref=ref or type,
        args=dict(args or {}),
        description=description,
    )
    return BoundOutput(spec, fn) if fn is not None else spec


def action(
    tool: str,
    *,
    name: str | None = None,
    arguments: Mapping[str, Any] | Callable[..., Any] | None = None,
    requires_safety_check: bool = False,
    args: Mapping[str, Any] | None = None,
) -> BoundAction | ActionSpec:
    """Declare a deterministic tool action."""
    spec = ActionSpec(
        name=name or tool,
        kind="tool",
        tool=tool,
        arguments=dict(arguments or {}) if isinstance(arguments, Mapping) else {},
        args=dict(args or {}),
        requires_approval=requires_safety_check,
    )
    if callable(arguments):
        return BoundAction(spec, argument_factory=arguments)
    return spec


def call(
    name: str,
    fn: Callable[..., Any],
    *,
    args: Mapping[str, Any] | None = None,
) -> BoundAction:
    """Declare a deterministic side-effectful Python call."""
    spec = ActionSpec(
        name=name,
        kind="deterministic",
        ref=name,
        args=dict(args or {}),
    )
    return BoundAction(spec, fn=fn)


def effect(
    id: str,
    fn: Callable[..., Any],
    *,
    args: Mapping[str, Any] | None = None,
) -> BoundEffect:
    """Declare a pure deterministic effect function."""
    return BoundEffect(EffectSpec(id=id, ref=id, args=dict(args or {})), fn)


def increment(path: str, by: int | float = 1) -> Increment:
    """Declare an increment effect shorthand for `.effects(...)`."""
    return Increment(path=path, by=by)


__all__ = [
    "BoundAction",
    "BoundEffect",
    "BoundGuard",
    "BoundInvariant",
    "BoundOutput",
    "Increment",
    "action",
    "call",
    "effect",
    "guard",
    "increment",
    "invariant",
    "output",
]
