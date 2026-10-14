"""Executable FSLM definitions and runtime bindings."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

from ellements.core import ToolRegistry

from .models import MachineSpec

BindingCallable = Callable[..., Any]


@dataclass
class RuntimeBindings:
    """Non-serializable runtime objects referenced by a `MachineSpec`."""

    guards: dict[str, BindingCallable] = field(default_factory=dict)
    invariants: dict[str, BindingCallable] = field(default_factory=dict)
    effects: dict[str, BindingCallable] = field(default_factory=dict)
    actions: dict[str, BindingCallable] = field(default_factory=dict)
    outputs: dict[str, BindingCallable] = field(default_factory=dict)
    evaluators: dict[str, Any] = field(default_factory=dict)
    modules: dict[str, ModuleType] = field(default_factory=dict)
    tools: ToolRegistry | None = None

    def merge(self, other: RuntimeBindings) -> RuntimeBindings:
        """Return a binding set with `other` taking precedence."""
        return RuntimeBindings(
            guards={**self.guards, **other.guards},
            invariants={**self.invariants, **other.invariants},
            effects={**self.effects, **other.effects},
            actions={**self.actions, **other.actions},
            outputs={**self.outputs, **other.outputs},
            evaluators={**self.evaluators, **other.evaluators},
            modules={**self.modules, **other.modules},
            tools=other.tools or self.tools,
        )

    def resolve(self, ref: str, category: str) -> BindingCallable:
        """Resolve a binding reference for one callable category."""
        registry = self._registry(category)
        if ref in registry:
            return registry[ref]
        if ref.startswith("builtin."):
            return _builtin(ref)
        if "." in ref:
            alias, _, name = ref.partition(".")
            module = self.modules.get(alias)
            if module is not None and hasattr(module, name):
                value = getattr(module, name)
                if callable(value):
                    return cast(BindingCallable, value)
                raise TypeError(f"binding {ref!r} is not callable")
        imported = _import_ref(ref)
        if imported is not None:
            return imported
        raise KeyError(f"unresolved {category} binding reference {ref!r}")

    def _registry(self, category: str) -> dict[str, BindingCallable]:
        registries = {
            "guard": self.guards,
            "invariant": self.invariants,
            "effect": self.effects,
            "action": self.actions,
            "output": self.outputs,
        }
        return registries.get(category, {})


@dataclass
class MachineDefinition:
    """Executable machine definition: serializable spec plus bindings."""

    spec: MachineSpec
    bindings: RuntimeBindings = field(default_factory=RuntimeBindings)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.spec, name)

    def initial_snapshot(self, **kwargs: Any) -> Any:
        """Delegate initial snapshot creation to the spec."""
        return self.spec.initial_snapshot(**kwargs)

    def to_json_file(self, path: str) -> None:
        """Write the serializable spec as JSON."""
        self.spec.to_json_file(path)


def coerce_definition(
    value: MachineDefinition | MachineSpec | Any,
    *,
    bindings: RuntimeBindings | None = None,
) -> MachineDefinition:
    """Normalize specs/builders/callables into a `MachineDefinition`."""
    if isinstance(value, MachineDefinition):
        if bindings is None:
            return value
        return MachineDefinition(value.spec, value.bindings.merge(bindings))
    if isinstance(value, MachineSpec):
        return MachineDefinition(value, bindings or RuntimeBindings())
    build = getattr(value, "build", None)
    if callable(build):
        return coerce_definition(build(), bindings=bindings)
    if callable(value):
        return coerce_definition(value(), bindings=bindings)
    raise TypeError(
        "expected MachineDefinition, MachineSpec, builder with build(), "
        "or callable returning one"
    )


def bindings_from_mapping(mapping: Mapping[str, Mapping[str, BindingCallable]]) -> RuntimeBindings:
    """Build bindings from a nested mapping keyed by category."""
    return RuntimeBindings(
        guards=dict(mapping.get("guards", {})),
        invariants=dict(mapping.get("invariants", {})),
        effects=dict(mapping.get("effects", {})),
        actions=dict(mapping.get("actions", {})),
        outputs=dict(mapping.get("outputs", {})),
    )


def _builtin(ref: str) -> BindingCallable:
    from . import builtins

    name = ref.removeprefix("builtin.")
    value = getattr(builtins, name, None)
    if callable(value):
        return cast(BindingCallable, value)
    raise KeyError(f"unknown built-in binding {ref!r}")


def _import_ref(ref: str) -> BindingCallable | None:
    module_name: str
    attr_name: str
    if ":" in ref:
        module_name, attr_name = ref.rsplit(":", 1)
    elif "." in ref:
        module_name, attr_name = ref.rsplit(".", 1)
    else:
        return None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    value = getattr(module, attr_name)
    if not callable(value):
        raise TypeError(f"binding {ref!r} is not callable")
    return cast(BindingCallable, value)


__all__ = [
    "BindingCallable",
    "MachineDefinition",
    "RuntimeBindings",
    "bindings_from_mapping",
    "coerce_definition",
]
