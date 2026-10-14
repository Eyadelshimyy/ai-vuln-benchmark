"""Runtime context passed to FSLM callables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ellements.core import ToolRegistry

from .definition import MachineDefinition
from .models import (
    FSLMEvent,
    MachineSnapshot,
    MachineSpec,
    StateSpec,
    TransitionSpec,
)


@dataclass(slots=True)
class FSLMContext:
    """Context object passed to deterministic and NL hooks."""

    spec: MachineSpec
    definition: MachineDefinition
    state: StateSpec
    transition: TransitionSpec | None
    snapshot: MachineSnapshot
    event: FSLMEvent
    vars: Mapping[str, Any]
    step_id: str
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def tools(self) -> ToolRegistry | None:
        """Tool registry available to this machine, if any."""
        return self.definition.bindings.tools

    async def call_tool(self, name: str, **arguments: Any) -> Any:
        """Invoke a registered ellements tool by name."""
        if self.tools is None:
            raise KeyError(f"no tool registry configured for {name!r}")
        if name not in self.tools:
            raise KeyError(f"tool {name!r} is not registered")
        return await self.tools[name].invoke(**arguments)


__all__ = ["FSLMContext"]
