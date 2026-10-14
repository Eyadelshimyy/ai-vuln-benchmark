"""Canonical Pydantic models for `ellements.fslm`."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import SpecValidationError

GuardKind = Literal["deterministic", "nl"]
InvariantKind = Literal["deterministic", "nl"]
ActionKind = Literal["deterministic", "nl", "tool"]
OutputKind = Literal["deterministic", "nl", "static"]
TransitionKind = Literal["normal", "recovery"]
ExecutionMode = Literal["execute", "dry_run"]
TraceMode = Literal["disabled", "jsonl"]
TerminalSafetyMode = Literal["nl", "disabled"]
StepStatus = Literal["transitioned", "no_transition", "blocked"]
ActionStatus = Literal["planned", "executed", "blocked", "failed"]


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BudgetSpec(BaseModel):
    """Execution budgets for one machine or state."""

    model_config = ConfigDict(extra="forbid")

    max_steps: int | None = Field(default=None, ge=1)
    max_nl_calls: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    max_terminal_actions: int | None = Field(default=None, ge=1)
    max_wall_seconds: float | None = Field(default=None, gt=0)


class ConfidencePolicy(BaseModel):
    """How structured natural-language confidence gates decisions."""

    model_config = ConfigDict(extra="forbid")

    min_guard_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_invariant_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    low_confidence_behavior: Literal[
        "block", "no_transition", "request_approval"
    ] = "block"


class ExecutionPolicy(BaseModel):
    """Machine-wide runtime policy."""

    model_config = ConfigDict(extra="forbid")

    execution: ExecutionMode = "execute"
    persistence: str | None = None
    traces: TraceMode = "disabled"
    verbose: bool = False
    terminal_safety: TerminalSafetyMode = "nl"
    confidence: ConfidencePolicy = Field(default_factory=ConfidencePolicy)
    budgets: BudgetSpec = Field(default_factory=BudgetSpec)


class ToolRef(BaseModel):
    """Serializable reference to an ellements tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    params_json_schema: dict[str, Any] = Field(default_factory=dict)


class AffordanceSpec(BaseModel):
    """A state-local capability description."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""


class EventPatternSpec(BaseModel):
    """Declared event pattern that can trigger a transition."""

    model_config = ConfigDict(extra="forbid")

    type: str
    description: str = ""
    source: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"type": value}
        return value


class GuardSpec(BaseModel):
    """A transition guard."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: GuardKind = "deterministic"
    text: str = ""
    ref: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"id": value}
        return value


class InvariantSpec(BaseModel):
    """A state invariant."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: InvariantKind = "deterministic"
    text: str = ""
    ref: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    severity: Literal["error", "warning"] = "error"
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"id": value}
        return value


class ActionSpec(BaseModel):
    """A declared tool/action invocation affordance."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    kind: ActionKind = "tool"
    tool: str | None = None
    ref: str | None = None
    instruction: str = ""
    text: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    args: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"tool": value}
        return value

    @model_validator(mode="after")
    def _default_name(self) -> Self:
        if self.name is None:
            self.name = self.tool or self.ref or "action"
        if self.kind == "tool" and self.tool is None:
            raise ValueError("tool actions require `tool`")
        if self.kind == "deterministic" and self.ref is None:
            raise ValueError("deterministic actions require `ref`")
        return self


class OutputSpec(BaseModel):
    """A typed output the FSLM may emit."""

    model_config = ConfigDict(extra="forbid")

    type: str
    kind: OutputKind = "static"
    description: str = ""
    text: str = ""
    ref: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    payload_schema: dict[str, Any] = Field(default_factory=dict)
    destination: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"type": value}
        return value


class EffectSpec(BaseModel):
    """A pure variable/snapshot patch."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    ref: str
    args: dict[str, Any] = Field(default_factory=dict)


class TransitionSpec(BaseModel):
    """One legal transition in a machine."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    target: str
    trigger: EventPatternSpec = Field(alias="event")
    description: str = ""
    guards: list[GuardSpec] = Field(default_factory=list)
    actions: list[ActionSpec] = Field(default_factory=list)
    emits: list[OutputSpec] = Field(default_factory=list)
    effects: dict[str, Any] = Field(default_factory=dict)
    effect_calls: list[EffectSpec] = Field(default_factory=list)
    kind: TransitionKind = "normal"
    weight: float | None = Field(default=None, gt=0)
    random_group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("guards", mode="before")
    @classmethod
    def _coerce_guards(cls, value: Any) -> Any:
        return _coerce_list(value)

    @field_validator("actions", mode="before")
    @classmethod
    def _coerce_actions(cls, value: Any) -> Any:
        return _coerce_list(value)

    @field_validator("emits", mode="before")
    @classmethod
    def _coerce_outputs(cls, value: Any) -> Any:
        return _coerce_list(value)

    @field_validator("effect_calls", mode="before")
    @classmethod
    def _coerce_effect_calls(cls, value: Any) -> Any:
        return _coerce_list(value)

class StateSpec(BaseModel):
    """One state and its local locus of control."""

    model_config = ConfigDict(extra="forbid")

    name: str
    objective: str = ""
    description: str = ""
    affordances: list[AffordanceSpec] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    emits: list[OutputSpec] = Field(default_factory=list)
    allowed_event_types: list[str] = Field(default_factory=list)
    invariants: list[InvariantSpec] = Field(default_factory=list)
    transitions: list[str] = Field(default_factory=list)
    recovery_transitions: list[str] = Field(default_factory=list)
    budgets: BudgetSpec | None = None
    terminal: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"objective": value}
        return value

    @field_validator("affordances", mode="before")
    @classmethod
    def _coerce_affordances(cls, value: Any) -> Any:
        items = _coerce_list(value)
        result: list[Any] = []
        for item in items:
            if isinstance(item, str):
                result.append({"name": item})
            else:
                result.append(item)
        return result

    @field_validator("emits", mode="before")
    @classmethod
    def _coerce_emits(cls, value: Any) -> Any:
        return _coerce_list(value)

    @field_validator("invariants", mode="before")
    @classmethod
    def _coerce_invariants(cls, value: Any) -> Any:
        return _coerce_list(value)


class MachineSpec(BaseModel):
    """Canonical serializable FSLM specification."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    description: str = ""
    version: str = "0.1"
    initial: str
    states: dict[str, StateSpec]
    transitions: dict[str, TransitionSpec] = Field(default_factory=dict)
    tools: dict[str, ToolRef] = Field(default_factory=dict)
    outputs: dict[str, OutputSpec] = Field(default_factory=dict)
    policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data["states"] = _normalize_states(data.get("states", {}))
        data["transitions"] = _normalize_transitions(
            data.get("transitions", {}),
            data["states"],
        )
        return data

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        if self.initial not in self.states:
            raise SpecValidationError(
                f"initial state {self.initial!r} is not declared"
            )
        for state_name, state in self.states.items():
            if state.name != state_name:
                raise SpecValidationError(
                    f"state key {state_name!r} does not match name {state.name!r}"
                )
        for transition in self.transitions.values():
            if transition.source not in self.states:
                raise SpecValidationError(
                    f"transition {transition.name!r} source "
                    f"{transition.source!r} is not declared"
                )
            if transition.target not in self.states:
                raise SpecValidationError(
                    f"transition {transition.name!r} target "
                    f"{transition.target!r} is not declared"
                )
            state = self.states[transition.source]
            bucket = (
                state.recovery_transitions
                if transition.kind == "recovery"
                else state.transitions
            )
            if transition.name not in bucket:
                bucket.append(transition.name)
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> MachineSpec:
        """Load a machine specification from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_json_file(self, path: str | Path) -> None:
        """Write this specification as JSON for interchange or debugging."""
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json", by_alias=True), indent=2),
            encoding="utf-8",
        )

    def initial_snapshot(
        self,
        *,
        machine_id: str | None = None,
        variables: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ) -> MachineSnapshot:
        """Create a fresh snapshot at the initial state."""
        return MachineSnapshot(
            machine_id=machine_id or self.name,
            machine_name=self.name,
            current_state=self.initial,
            variables=variables or {},
            random_seed=random_seed,
        )


class FSLMEvent(BaseModel):
    """One typed event fed into a machine step."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    type: str
    source: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=_utcnow)
    parent_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MachineSnapshot(BaseModel):
    """Durable runtime state for one machine instance."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    machine_id: str
    machine_name: str
    current_state: str
    variables: dict[str, Any] = Field(default_factory=dict)
    step_index: int = Field(default=0, ge=0)
    random_seed: int | None = None
    random_draws: int = Field(default=0, ge=0)
    pending_actions: list[dict[str, Any]] = Field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=_utcnow)


class DecisionResult(BaseModel):
    """Structured decision returned by deterministic or NL evaluators."""

    model_config = ConfigDict(extra="forbid")

    id: str
    allowed: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputRecord(BaseModel):
    """Concrete output emitted by a step."""

    model_config = ConfigDict(extra="forbid")

    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    destination: str | None = None


class ActionResult(BaseModel):
    """Concrete result of planning or executing one action."""

    model_config = ConfigDict(extra="forbid")

    action_name: str
    tool: str
    status: ActionStatus
    output: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class StepResult(BaseModel):
    """Result of one machine step."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(default_factory=_new_id)
    event_id: str
    source_state: str
    target_state: str
    status: StepStatus
    selected_transition: str | None = None
    guard_results: list[DecisionResult] = Field(default_factory=list)
    invariant_results: list[DecisionResult] = Field(default_factory=list)
    outputs: list[OutputRecord] = Field(default_factory=list)
    actions: list[ActionResult] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    random: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    new_snapshot: MachineSnapshot


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_states(raw_states: Any) -> dict[str, Any]:
    if not isinstance(raw_states, dict):
        raise SpecValidationError("states must be a mapping")
    normalized: dict[str, Any] = {}
    for state_name, state_data in raw_states.items():
        if isinstance(state_data, BaseModel):
            state_data = state_data.model_dump(mode="python", exclude_none=True)
        if state_data is None:
            state_data = {}
        if isinstance(state_data, str):
            state_data = {"objective": state_data}
        if not isinstance(state_data, dict):
            raise SpecValidationError(f"state {state_name!r} must be a mapping")
        data = dict(state_data)
        data.setdefault("name", state_name)
        normalized[state_name] = data
    return normalized


def _normalize_transitions(
    raw_transitions: Any,
    states: dict[str, Any],
) -> dict[str, Any]:
    transitions: dict[str, Any] = {}
    if isinstance(raw_transitions, dict):
        for name, transition in raw_transitions.items():
            data = _normalize_transition_data(transition)
            data.setdefault("name", name)
            transitions[data["name"]] = data
    elif isinstance(raw_transitions, list):
        for index, transition in enumerate(raw_transitions):
            data = _normalize_transition_data(transition)
            data.setdefault("name", f"transition_{index + 1}")
            transitions[data["name"]] = data
    elif raw_transitions not in (None, {}):
        raise SpecValidationError("transitions must be a mapping or list")

    for state_name, state_data in states.items():
        local_specs = list(state_data.get("transitions", []) or [])
        recovery_specs = list(state_data.get("recovery_transitions", []) or [])
        state_transition_names: list[str] = []
        state_recovery_names: list[str] = []
        for index, item in enumerate(local_specs):
            if isinstance(item, str):
                state_transition_names.append(item)
                continue
            data = _normalize_transition_data(item)
            data.setdefault("source", state_name)
            data.setdefault("kind", "normal")
            data.setdefault("name", f"{state_name}_{data['event']['type']}_{index + 1}")
            transitions[data["name"]] = data
            state_transition_names.append(data["name"])
        for index, item in enumerate(recovery_specs):
            if isinstance(item, str):
                state_recovery_names.append(item)
                continue
            data = _normalize_transition_data(item)
            data.setdefault("source", state_name)
            data["kind"] = "recovery"
            data.setdefault(
                "name", f"{state_name}_recover_{data['event']['type']}_{index + 1}"
            )
            transitions[data["name"]] = data
            state_recovery_names.append(data["name"])
        state_data["transitions"] = state_transition_names
        state_data["recovery_transitions"] = state_recovery_names
    return transitions


def _normalize_transition_data(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python", by_alias=True, exclude_none=True)
    if not isinstance(value, dict):
        raise SpecValidationError("transition entries must be mappings")
    data = dict(value)
    if "target" not in data and "to" in data:
        data["target"] = data.pop("to")
    if "event" not in data:
        trigger = data.pop("trigger", None)
        if trigger is not None:
            data["event"] = trigger
    if "event" not in data:
        raise SpecValidationError("transition is missing event")
    if isinstance(data["event"], str):
        data["event"] = {"type": data["event"]}
    if "target" not in data:
        raise SpecValidationError("transition is missing target")
    return data


__all__ = [
    "ActionResult",
    "ActionSpec",
    "ActionStatus",
    "ActionKind",
    "AffordanceSpec",
    "BudgetSpec",
    "ConfidencePolicy",
    "DecisionResult",
    "EventPatternSpec",
    "ExecutionMode",
    "ExecutionPolicy",
    "EffectSpec",
    "FSLMEvent",
    "GuardKind",
    "GuardSpec",
    "InvariantKind",
    "InvariantSpec",
    "MachineSnapshot",
    "MachineSpec",
    "OutputRecord",
    "OutputSpec",
    "OutputKind",
    "OutputSpec",
    "StateSpec",
    "StepResult",
    "StepStatus",
    "TerminalSafetyMode",
    "ToolRef",
    "TraceMode",
    "TransitionKind",
    "TransitionSpec",
]
