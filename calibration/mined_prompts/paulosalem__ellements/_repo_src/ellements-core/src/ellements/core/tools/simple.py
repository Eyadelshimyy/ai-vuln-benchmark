"""The :class:`SimpleTool` convenience builder.

``SimpleTool`` wraps an arbitrary Python callable so it satisfies the
:class:`~ellements.core.tools.Tool` Protocol and can be materialized as
a :class:`~ellements.core.tools.ToolSpec`.

Construction is cheap; the JSON Schema is computed lazily on first
access and cached. The instance is also callable, so simple in-process
test harnesses can drive it without going through the schema path.

The module also exposes :func:`bind_json_invoker`, a helper used by
agent backends that bridge between a provider's JSON-payload calling
convention and a Python callable's positional/keyword binding.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .schemas import json_schema_for_callable, resolve_callable_signature
from .spec import ToolSpec


class SimpleTool:
    """Adapt a callable as a :class:`~ellements.core.tools.Tool`.

    Args:
        func: The callable to expose as a tool.
        name: Override the tool name. Defaults to ``func.__name__``.
        description: Override the description. Defaults to the
            callable's docstring.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.func = func
        self.name: str = name or getattr(func, "__name__", None) or "tool"
        self.description: str = description or inspect.getdoc(func) or self.name
        self.__name__ = self.name
        self.__doc__ = self.description
        self.__signature__ = resolve_callable_signature(func)
        self._params_json_schema: dict[str, Any] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    async def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Run the underlying callable, awaiting it if needed."""
        result = self.func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def on_invoke_tool(self, _context: Any, input_json: str) -> Any:
        """Invoke the tool from a JSON payload (OpenAI Agents SDK shape)."""
        return await _invoke_from_json(self.invoke, input_json)

    @property
    def params_json_schema(self) -> dict[str, Any]:
        """JSON Schema of the wrapped callable's parameters (cached)."""
        if self._params_json_schema is None:
            self._params_json_schema = json_schema_for_callable(self.func, tool_name=self.name)
        return self._params_json_schema

    def to_spec(self) -> ToolSpec:
        """Materialize this tool as a :class:`ToolSpec`."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            params_json_schema=self.params_json_schema,
            invoke=self.invoke,
        )


def bind_json_invoker(
    invoke: Callable[..., Awaitable[Any]],
) -> Callable[[Any, str], Awaitable[Any]]:
    """Wrap an awaitable to accept an OpenAI-Agents-style JSON payload.

    Many agent SDKs hand the tool a single JSON string and expect the
    adapter to parse it into ``*args``/``**kwargs``. This helper does
    that translation once so adapters don't reinvent it.
    """

    async def on_invoke(_context: Any, input_json: str) -> Any:
        return await _invoke_from_json(invoke, input_json)

    return on_invoke


async def _invoke_from_json(
    invoke: Callable[..., Awaitable[Any]],
    input_json: str,
) -> Any:
    text = str(input_json or "").strip()
    if not text:
        return await invoke()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return await invoke(text)
    if isinstance(payload, Mapping):
        return await invoke(**dict(payload))
    if isinstance(payload, list):
        return await invoke(*payload)
    if payload is None:
        return await invoke()
    return await invoke(payload)


__all__ = ["SimpleTool", "bind_json_invoker"]
