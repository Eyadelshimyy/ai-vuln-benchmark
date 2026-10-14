"""The :class:`Tool` Protocol — what every tool looks like to ellements.

A tool is any object that exposes a stable name, a description, a
provider-neutral JSON Schema for its parameters, and an awaitable
``invoke`` method that runs it.

Anything matching this shape is a tool. Inheritance is not required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Provider-neutral tool surface consumed by every backend.

    Implementations must expose:

    - ``name`` — stable identifier (matches the function name the LLM emits).
    - ``description`` — human-readable description shown to the model.
    - ``params_json_schema`` — JSON Schema (Draft-2020-12 compatible) for
      the parameters object. Provider-specific dialects are produced
      from this by :class:`~ellements.core.tools.ToolDialect`.
    - ``invoke`` — async invocation accepting keyword arguments.
    """

    name: str
    description: str
    params_json_schema: dict[str, Any]

    async def invoke(self, **kwargs: Any) -> Any: ...


__all__ = ["Tool"]
