"""The :class:`ToolRegistry` — a typed, composable tool collection.

Every tool factory should return a ``ToolRegistry``. Apps compose
registries with :meth:`ToolRegistry.merge` (or with the
``Mapping`` interface — ``{**r1, **r2}`` and ``dict.update(r)`` both
work because :class:`ToolRegistry` is a read-only ``Mapping[str,
ToolSpec]``). LLM clients call :meth:`ToolRegistry.to_dialect` to
produce the provider-specific wire format, and
:meth:`ToolRegistry.executor` to obtain the async runtime.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from typing import Any

from .protocol import Tool
from .simple import SimpleTool
from .spec import ToolSpec

ToolInput = Tool | ToolSpec | Callable[..., Any]
"""Anything accepted by :meth:`ToolRegistry.register`."""

ToolExecutorFn = Callable[[str, dict[str, Any]], Awaitable[str]]
"""Async ``(tool_name, args) -> result_string`` runtime executor."""


class ToolRegistry(Mapping[str, ToolSpec]):
    """An ordered, named collection of tools.

    The registry preserves insertion order so the order of tools shown
    to a model is deterministic across runs. It implements the
    read-only :class:`~collections.abc.Mapping` interface, which means
    every standard dict idiom works without surprises:

    >>> combined = {**search_tools(), **crawler_tools()}
    >>> for name in registry:
    ...     spec = registry[name]
    """

    def __init__(self, tools: Iterable[ToolInput] | Mapping[str, ToolInput] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        if tools is None:
            return
        if isinstance(tools, Mapping):
            for name, tool in tools.items():
                self.register(tool, name=name)
        else:
            for tool in tools:
                self.register(tool)

    # ── registration ─────────────────────────────────────────────────

    def register(
        self,
        tool: ToolInput,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolSpec:
        """Add *tool* to the registry and return the materialized spec.

        Args:
            tool: A :class:`Tool`, :class:`ToolSpec`, or plain callable.
            name: Override the resulting tool name.
            description: Override the description.

        Returns:
            The :class:`ToolSpec` that was registered.

        Raises:
            ValueError: If a tool with the same name already exists.
        """
        spec = self._to_spec(tool, name=name, description=description)
        if spec.name in self._specs:
            raise ValueError(f"Tool {spec.name!r} is already registered.")
        self._specs[spec.name] = spec
        return spec

    def merge(self, other: Mapping[str, ToolInput]) -> ToolRegistry:
        """Return a new registry containing this and *other*.

        *other* may be another :class:`ToolRegistry` or any
        ``{name: tool}`` mapping.

        Raises:
            ValueError: If the two registries share a tool name.
        """
        other_specs: dict[str, ToolSpec]
        if isinstance(other, ToolRegistry):
            other_specs = dict(other._specs)
        else:
            other_specs = {
                name: self._to_spec(tool, name=name, description=None)
                for name, tool in other.items()
            }
        overlap = set(self._specs) & set(other_specs)
        if overlap:
            raise ValueError(
                f"Cannot merge registries: duplicate tool names {sorted(overlap)}."
            )
        merged = ToolRegistry()
        merged._specs.update(self._specs)
        merged._specs.update(other_specs)
        return merged

    # ── Mapping[str, ToolSpec] protocol ──────────────────────────────

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._specs

    def __getitem__(self, name: str) -> ToolSpec:
        return self._specs[name]

    # ── consumption ──────────────────────────────────────────────────

    def to_dialect(self, dialect: Any) -> list[dict[str, Any]]:
        """Render every tool as a provider-specific definition.

        Args:
            dialect: Any object with an ``emit(spec) -> dict`` method.

        Returns:
            The list of provider-specific tool definitions in
            registration order.
        """
        return [dialect.emit(spec) for spec in self._specs.values()]

    def executor(self) -> ToolExecutorFn:
        """Return an async ``(name, args) -> result_string`` executor."""
        from .executor import stringify_tool_result

        specs = dict(self._specs)

        async def execute(tool_name: str, arguments: dict[str, Any]) -> str:
            spec = specs.get(tool_name)
            if spec is None:
                available = ", ".join(sorted(specs)) or "(none)"
                raise KeyError(
                    f"Unknown tool {tool_name!r}. Available: {available}."
                )
            result = await spec.invoke(**arguments)
            return stringify_tool_result(result)

        return execute

    # ── construction helpers ─────────────────────────────────────────

    @staticmethod
    def _to_spec(
        tool: ToolInput,
        *,
        name: str | None,
        description: str | None,
    ) -> ToolSpec:
        if isinstance(tool, ToolSpec):
            if name is None and description is None:
                return tool
            return ToolSpec(
                name=name or tool.name,
                description=description or tool.description,
                params_json_schema=tool.params_json_schema,
                invoke=tool.invoke,
            )
        if isinstance(tool, SimpleTool):
            spec = tool.to_spec()
            if name is None and description is None:
                return spec
            return ToolSpec(
                name=name or spec.name,
                description=description or spec.description,
                params_json_schema=spec.params_json_schema,
                invoke=spec.invoke,
            )
        if isinstance(tool, Tool):
            return ToolSpec(
                name=name or tool.name,
                description=description or tool.description,
                params_json_schema=tool.params_json_schema,
                invoke=tool.invoke,
            )
        if callable(tool):
            return SimpleTool(tool, name=name, description=description).to_spec()
        raise TypeError(
            f"Cannot register {type(tool).__name__}; expected Tool, ToolSpec, or callable."
        )

    @classmethod
    def from_mapping(cls, tools: Mapping[str, ToolInput]) -> ToolRegistry:
        """Build a registry from a ``{name: tool}`` mapping."""
        return cls(tools)

    @classmethod
    def from_callables(cls, tools: Iterable[Callable[..., Any]]) -> ToolRegistry:
        """Build a registry from an iterable of plain callables."""
        return cls(tools)


__all__ = ["ToolExecutorFn", "ToolInput", "ToolRegistry"]
