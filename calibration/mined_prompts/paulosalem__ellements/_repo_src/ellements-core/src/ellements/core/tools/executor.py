"""Tool-call runtime helpers used by :class:`~ellements.core.LLMClient`.

The executor type and the result-stringification helper live here
together because they are always used in concert by the tool-calling
loop.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import BaseModel

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]
"""Async ``(tool_name, arguments) -> result_string`` runtime callable."""


def stringify_tool_result(result: Any) -> str:
    """Convert arbitrary tool output into a stable string payload."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, Mapping):
        return json.dumps(dict(result), default=_json_default)
    if isinstance(result, (list, tuple, set)):
        return json.dumps(list(result), default=_json_default)
    try:
        return json.dumps(result, default=_json_default)
    except TypeError:
        return str(result)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


__all__ = ["ToolExecutor", "stringify_tool_result"]
