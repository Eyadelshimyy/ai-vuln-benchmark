"""Fluent Python DSL for building `MachineSpec` objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .definition import MachineDefinition, RuntimeBindings
from .det import (
    BoundAction,
    BoundEffect,
    BoundGuard,
    BoundInvariant,
    BoundOutput,
    Increment,
)
from .models import (
    ActionSpec,
    AffordanceSpec,
    EventPatternSpec,
    ExecutionPolicy,
    GuardSpec,
    InvariantSpec,
    MachineSpec,
    OutputSpec,
    StateSpec,
    ToolRef,
    TransitionKind,
    TransitionSpec,
)


class MachineBuilder:
    """Low-boilerplate builder for `MachineSpec`."""

    def __init__(
        self,
        name: str,
        *,
        initial: str,
        description: str = "",
        version: str = "0.1",
    ) -> None:
        self.name = name
        self.initial = initial
        self.description = description.strip()
        self.version = version
        self._states: dict[str, StateSpec] = {}
        self._transitions: dict[str, TransitionSpec] = {}
        self._tools: dict[str, ToolRef] = {}
        self._outputs: dict[str, OutputSpec] = {}
        self._policy = ExecutionPolicy()
        self._bindings = RuntimeBindings()
        self._transition_counter = 0

    def policy(self, **kwargs: Any) -> MachineBuilder:
        """Update machine-wide execution policy."""
        data = self._policy.model_dump()
        data.update(kwargs)
        self._policy = ExecutionPolicy.model_validate(data)
        return self

    def use_tools(self, *tools: Any) -> MachineBuilder:
        """Register ellements tool specs or light tool references."""
        for tool in tools:
            name = getattr(tool, "name", None)
            if name is None:
                raise TypeError("tools must expose a `name` attribute")
            description = str(getattr(tool, "description", ""))
            schema = getattr(tool, "params_json_schema", None) or {}
            self._tools[str(name)] = ToolRef(
                name=str(name),
                description=description,
                params_json_schema=dict(schema),
            )
        return self

    def state(
        self,
        name: str,
        *,
        objective: str = "",
        description: str = "",
        tools: list[str] | None = None,
        emits: list[str | OutputSpec] | None = None,
        terminal: bool = False,
    ) -> StateBuilder:
        """Open a state-local affordance block."""
        state = StateSpec(
            name=name,
            objective=objective.strip(),
            description=description.strip(),
            tools=tools or [],
            emits=[_output_spec(item) for item in emits or []],
            terminal=terminal,
        )
        self._states[name] = state
        return StateBuilder(self, state)

    def build(self) -> MachineDefinition:
        """Materialize the executable machine definition."""
        spec = MachineSpec(
            name=self.name,
            description=self.description,
            version=self.version,
            initial=self.initial,
            states=self._states,
            transitions=self._transitions,
            tools=self._tools,
            outputs=self._outputs,
            policy=self._policy,
        )
        return MachineDefinition(spec=spec, bindings=self._bindings)

    def build_spec(self) -> MachineSpec:
        """Materialize only the serializable machine specification."""
        return self.build().spec

    def _register_transition(self, transition: TransitionSpec) -> None:
        if transition.name in self._transitions:
            raise ValueError(f"transition {transition.name!r} is already declared")
        self._transitions[transition.name] = transition
        state = self._states[transition.source]
        bucket = (
            state.recovery_transitions
            if transition.kind == "recovery"
            else state.transitions
        )
        bucket.append(transition.name)

    def _next_transition_name(
        self,
        source: str,
        event_type: str,
        target: str | None = None,
    ) -> str:
        if target is not None:
            base = f"{source}_{event_type}_to_{target}"
            if base not in self._transitions:
                return base
        self._transition_counter += 1
        return f"{source}_{event_type}_{self._transition_counter}"

    def _register_guard_binding(self, value: str | GuardSpec | BoundGuard | Any) -> GuardSpec:
        if isinstance(value, BoundGuard):
            self._bindings.guards[value.spec.ref or value.spec.id] = value.fn
            return value.spec
        if isinstance(value, GuardSpec):
            return value
        if callable(value):
            name = getattr(value, "__name__", "guard")
            self._bindings.guards[name] = value
            return GuardSpec(id=name, kind="deterministic", ref=name)
        return GuardSpec(id=str(value), kind="deterministic")

    def _register_invariant_binding(
        self,
        value: str | InvariantSpec | BoundInvariant | Any,
    ) -> InvariantSpec:
        if isinstance(value, BoundInvariant):
            self._bindings.invariants[value.spec.ref or value.spec.id] = value.fn
            return value.spec
        if isinstance(value, InvariantSpec):
            return value
        if callable(value):
            name = getattr(value, "__name__", "invariant")
            self._bindings.invariants[name] = value
            return InvariantSpec(id=name, kind="deterministic", ref=name)
        return InvariantSpec(id=str(value), kind="deterministic")

    def _register_action_binding(
        self,
        value: str | ActionSpec | BoundAction,
        *,
        name: str | None,
        instruction: str,
        arguments: Mapping[str, Any] | None,
        requires_approval: bool,
    ) -> ActionSpec:
        if isinstance(value, BoundAction):
            if value.fn is not None:
                self._bindings.actions[value.spec.ref or value.spec.name or "action"] = value.fn
            if value.argument_factory is not None:
                ref = f"{value.spec.name or value.spec.tool}_arguments"
                self._bindings.actions[ref] = value.argument_factory
                value.spec.args["arguments_ref"] = ref
            return value.spec
        if isinstance(value, ActionSpec):
            return value
        return ActionSpec(
            name=name or value,
            kind="tool",
            tool=str(value),
            instruction=instruction.strip(),
            arguments=dict(arguments or {}),
            requires_approval=requires_approval,
        )

    def _register_output_binding(
        self,
        value: str | OutputSpec | BoundOutput,
        *,
        description: str,
        payload_schema: Mapping[str, Any] | None,
    ) -> OutputSpec:
        if isinstance(value, BoundOutput):
            self._bindings.outputs[value.spec.ref or value.spec.type] = value.fn
            return value.spec
        if isinstance(value, OutputSpec):
            return value
        return OutputSpec(
            type=str(value),
            description=description.strip(),
            payload_schema=dict(payload_schema or {}),
        )


class StateBuilder:
    """State-local builder; this is the machine's visible locus of control."""

    def __init__(self, machine: MachineBuilder, state: StateSpec) -> None:
        self._machine = machine
        self._state = state

    def __enter__(self) -> StateBuilder:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def affordance(self, name: str, description: str = "") -> StateBuilder:
        """Declare a state-local affordance."""
        self._state.affordances.append(
            AffordanceSpec(name=name, description=description.strip())
        )
        return self

    def invariant(
        self,
        invariant: str | InvariantSpec | BoundInvariant | Any,
    ) -> StateBuilder:
        """Declare a state invariant."""
        spec = self._machine._register_invariant_binding(invariant)
        self._state.invariants.append(spec)
        return self

    def on(self, event_type: str, description: str = "") -> TransitionBuilder:
        """Begin a normal transition triggered by an event."""
        return TransitionBuilder(
            self._machine,
            self._state,
            event_type=event_type,
            event_description=description.strip(),
            kind="normal",
        )

    def recover(self, event_type: str, description: str = "") -> TransitionBuilder:
        """Begin an explicit recovery transition."""
        return TransitionBuilder(
            self._machine,
            self._state,
            event_type=event_type,
            event_description=description.strip(),
            kind="recovery",
        )


class TransitionBuilder:
    """Fluent builder for one or more transitions sharing an event trigger."""

    def __init__(
        self,
        machine: MachineBuilder,
        state: StateSpec,
        *,
        event_type: str,
        event_description: str,
        kind: TransitionKind,
    ) -> None:
        self._machine = machine
        self._state = state
        self._event_type = event_type
        self._event_description = event_description
        self._kind = kind
        self._registered: list[str] = []

    def to(self, target: str, name: str | None = None) -> TransitionBuilder:
        """Declare a transition to another state."""
        transition_name = name or self._machine._next_transition_name(
            self._state.name,
            self._event_type,
            target,
        )
        self._register(target=target, name=transition_name)
        return self

    def stay(self, name: str | None = None) -> TransitionBuilder:
        """Declare an explicit self-transition."""
        return self.to(self._state.name, name=name)

    def choose(
        self,
        *choices: tuple[float, str] | tuple[float, str, str],
    ) -> TransitionBuilder:
        """Declare weighted transitions for a probabilistic choice."""
        group = f"{self._state.name}:{self._event_type}:{len(self._machine._transitions)}"
        for choice in choices:
            weight = float(choice[0])
            target = str(choice[1])
            name = (
                str(choice[2])
                if len(choice) == 3
                else self._machine._next_transition_name(self._state.name, target, target)
            )
            self._register(target=target, name=name, weight=weight, random_group=group)
        return self

    def when(self, *guards: str | GuardSpec | BoundGuard | Any) -> TransitionBuilder:
        """Attach guards to all transitions declared by this builder."""
        for transition in self._current_transitions():
            transition.guards.extend(
                self._machine._register_guard_binding(guard) for guard in guards
            )
        return self

    def do(
        self,
        action: str | ActionSpec | BoundAction,
        *,
        name: str | None = None,
        instruction: str = "",
        arguments: Mapping[str, Any] | None = None,
        requires_approval: bool = False,
    ) -> TransitionBuilder:
        """Attach an action to all transitions declared by this builder."""
        action_spec = self._machine._register_action_binding(
            action,
            name=name,
            instruction=instruction,
            arguments=arguments,
            requires_approval=requires_approval,
        )
        for transition in self._current_transitions():
            transition.actions.append(action_spec)
        return self

    def call(
        self,
        fn: Any,
        *,
        name: str | None = None,
        args: Mapping[str, Any] | None = None,
    ) -> TransitionBuilder:
        """Attach a side-effectful Python call action."""
        if not callable(fn):
            raise TypeError("call() expects a callable")
        action_name = name or getattr(fn, "__name__", "python_call")
        return self.do(BoundAction(
            ActionSpec(
                name=action_name,
                kind="deterministic",
                ref=action_name,
                args=dict(args or {}),
            ),
            fn=fn,
        ))

    def emit(
        self,
        output: str | OutputSpec | BoundOutput,
        description: str = "",
        payload_schema: Mapping[str, Any] | None = None,
    ) -> TransitionBuilder:
        """Attach a typed output to all transitions declared by this builder."""
        spec = self._machine._register_output_binding(
            output,
            description=description,
            payload_schema=payload_schema,
        )
        for transition in self._current_transitions():
            transition.emits.append(spec)
        return self

    def effects(self, **variables: Any) -> TransitionBuilder:
        """Attach variable updates to all transitions declared by this builder."""
        normalized = {
            key: {"op": "increment", "path": value.path, "by": value.by}
            if isinstance(value, Increment)
            else value
            for key, value in variables.items()
        }
        for transition in self._current_transitions():
            transition.effects.update(normalized)
        return self

    def effect(self, effect: BoundEffect) -> TransitionBuilder:
        """Attach a pure deterministic effect callable."""
        self._machine._bindings.effects[effect.spec.ref] = effect.fn
        for transition in self._current_transitions():
            transition.effect_calls.append(effect.spec)
        return self

    def weight(self, value: float) -> TransitionBuilder:
        """Assign a probabilistic weight to all transitions declared here."""
        for transition in self._current_transitions():
            transition.weight = value
        return self

    def _register(
        self,
        *,
        target: str,
        name: str,
        weight: float | None = None,
        random_group: str | None = None,
    ) -> None:
        transition = TransitionSpec(
            name=name,
            source=self._state.name,
            target=target,
            event=EventPatternSpec(
                type=self._event_type,
                description=self._event_description,
            ),
            kind=self._kind,
            weight=weight,
            random_group=random_group,
        )
        self._machine._register_transition(transition)
        self._registered.append(name)

    def _current_transitions(self) -> list[TransitionSpec]:
        if not self._registered:
            raise ValueError("declare `.to(...)`, `.stay(...)`, or `.choose(...)` first")
        return [self._machine._transitions[name] for name in self._registered]


def machine(
    name: str,
    *,
    initial: str,
    description: str = "",
    version: str = "0.1",
) -> MachineBuilder:
    """Create a fluent FSLM builder."""
    return MachineBuilder(
        name,
        initial=initial,
        description=description,
        version=version,
    )


def _output_spec(value: str | OutputSpec) -> OutputSpec:
    return value if isinstance(value, OutputSpec) else OutputSpec(type=value)


__all__ = ["MachineBuilder", "StateBuilder", "TransitionBuilder", "machine"]
