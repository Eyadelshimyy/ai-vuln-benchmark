"""Generic tool-creation utility shared by all agent backends."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ellements.core import SimpleTool


def create_tool_from_method(
    method: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> SimpleTool:
    """Wrap a method as a canonical :class:`~ellements.core.SimpleTool`.

    The result is backend-agnostic; backends adapt it to their native
    tool type at agent-construction time.
    """
    tool_name = name or method.__name__
    tool_description = description or (
        inspect.getdoc(method) or f"Execute {tool_name}"
    )
    return SimpleTool(method, name=tool_name, description=tool_description)


__all__ = ["create_tool_from_method"]
