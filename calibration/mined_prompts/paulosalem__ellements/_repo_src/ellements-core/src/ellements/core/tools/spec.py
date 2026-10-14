"""The canonical in-memory tool representation: :class:`ToolSpec`.

Every tool — built from a callable, declared via a builder, or adapted
from an external system — is converted to a ``ToolSpec`` before it
reaches an LLM provider or an agent runtime. This is the seam every
dialect and backend adapter speaks to.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Provider-neutral canonical tool record.

    Args:
        name: Stable identifier (the function name the LLM emits).
        description: Human-readable description shown to the model.
        params_json_schema: JSON Schema for the keyword parameters.
        invoke: Async callable executing the tool given keyword args.
    """

    name: str
    description: str
    params_json_schema: dict[str, Any]
    invoke: Callable[..., Awaitable[Any]]


__all__ = ["ToolSpec"]
