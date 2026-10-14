"""Small built-in binding functions for YAML-authored machines."""

from __future__ import annotations

from typing import Any

from .context import FSLMContext

_MISSING = object()


def payload_exists(ctx: FSLMContext, args: dict[str, Any]) -> bool:
    """Return whether a JSON path exists in the current event payload."""
    return _get_path(ctx, args["path"], missing=_MISSING) is not _MISSING


def payload_equals(ctx: FSLMContext, args: dict[str, Any]) -> bool:
    """Return whether a JSON path equals a value."""
    return bool(_get_path(ctx, args["path"]) == args.get("value"))


def payload_not_equals(ctx: FSLMContext, args: dict[str, Any]) -> bool:
    """Return whether a JSON path does not equal a value."""
    return bool(_get_path(ctx, args["path"]) != args.get("value"))


def var_exists(ctx: FSLMContext, args: dict[str, Any]) -> bool:
    """Return whether a snapshot variable exists."""
    return _get_path(ctx, args["path"], missing=_MISSING) is not _MISSING


def var_equals(ctx: FSLMContext, args: dict[str, Any]) -> bool:
    """Return whether a snapshot variable path equals a value."""
    return bool(_get_path(ctx, args["path"]) == args.get("value"))


def var_not_equals(ctx: FSLMContext, args: dict[str, Any]) -> bool:
    """Return whether a snapshot variable path does not equal a value."""
    return bool(_get_path(ctx, args["path"]) != args.get("value"))


def assign(_ctx: FSLMContext, args: dict[str, Any]) -> dict[str, Any]:
    """Return a variable assignment patch."""
    return {str(args["path"]).removeprefix("$.snapshot.variables."): args.get("value")}


def increment(ctx: FSLMContext, args: dict[str, Any]) -> dict[str, Any]:
    """Return a variable increment patch."""
    key = str(args["path"]).removeprefix("$.snapshot.variables.")
    current = _get_path(ctx, args["path"], missing=0)
    return {key: current + args.get("by", 1)}


def append(ctx: FSLMContext, args: dict[str, Any]) -> dict[str, Any]:
    """Return a variable list-append patch."""
    key = str(args["path"]).removeprefix("$.snapshot.variables.")
    current = _get_path(ctx, args["path"], missing=[])
    if not isinstance(current, list):
        current = [current]
    return {key: [*current, args.get("value")]}


def copy_payload_to_var(ctx: FSLMContext, args: dict[str, Any]) -> dict[str, Any]:
    """Copy an event payload value to a snapshot variable."""
    value = _get_path(ctx, args["payload_path"])
    key = str(args["var_path"]).removeprefix("$.snapshot.variables.")
    return {key: value}


def _get_path(ctx: FSLMContext, path: str, *, missing: Any = None) -> Any:
    root: Any
    if path.startswith("$.event.payload"):
        root = ctx.event.payload
        parts = path.removeprefix("$.event.payload").strip(".").split(".")
    elif path.startswith("$.snapshot.variables"):
        root = ctx.snapshot.variables
        parts = path.removeprefix("$.snapshot.variables").strip(".").split(".")
    else:
        root = {"event": {"payload": ctx.event.payload}, "snapshot": {"variables": ctx.snapshot.variables}}
        parts = path.removeprefix("$.").split(".")
    if parts == [""]:
        return root
    current = root
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return missing
    return current


__all__ = [
    "append",
    "assign",
    "copy_payload_to_var",
    "increment",
    "payload_equals",
    "payload_exists",
    "payload_not_equals",
    "var_equals",
    "var_exists",
    "var_not_equals",
]
