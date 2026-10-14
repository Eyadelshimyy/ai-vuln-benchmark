"""Deterministic FSLM kernel."""

from __future__ import annotations

import inspect
import random
from collections.abc import Iterable
from typing import Any, Protocol
from uuid import uuid4

from .context import FSLMContext
from .definition import MachineDefinition, coerce_definition
from .models import (
    ActionResult,
    ActionSpec,
    DecisionResult,
    FSLMEvent,
    GuardSpec,
    InvariantSpec,
    MachineSnapshot,
    MachineSpec,
    OutputRecord,
    StepResult,
    StepStatus,
    TransitionSpec,
)
from .observers import FSLMEventRecord, FSLMObserver


class GuardEvaluator(Protocol):
    """Evaluates transition guards."""

    async def evaluate_guard(
        self,
        guard: GuardSpec,
        *,
        spec: MachineSpec | MachineDefinition,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> DecisionResult:
        """Return a structured guard decision."""


class InvariantChecker(Protocol):
    """Checks state invariants."""

    async def check_invariant(
        self,
        invariant: InvariantSpec,
        *,
        spec: MachineSpec | MachineDefinition,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> DecisionResult:
        """Return a structured invariant decision."""


class ActionExecutor(Protocol):
    """Executes action specs selected by a transition."""

    async def execute_action(
        self,
        action: ActionSpec,
        *,
        spec: MachineSpec | MachineDefinition,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> ActionResult:
        """Execute one action and return its result."""


class FSLMKernel:
    """Small deterministic FSLM kernel.

    The kernel owns graph mechanics. Natural-language judgment, tool execution,
    persistence, and coordination plug in through explicit protocols.
    """

    def __init__(
        self,
        spec: MachineSpec | MachineDefinition,
        *,
        guard_evaluator: GuardEvaluator | None = None,
        invariant_checker: InvariantChecker | None = None,
        action_executor: ActionExecutor | None = None,
        observers: Iterable[FSLMObserver] | None = None,
    ) -> None:
        self.definition = coerce_definition(spec)
        self.spec = self.definition.spec
        self.guard_evaluator = guard_evaluator
        self.invariant_checker = invariant_checker
        self.action_executor = action_executor
        self.observers = list(observers or [])

    async def step(self, snapshot: MachineSnapshot, event: FSLMEvent) -> StepResult:
        """Apply one event to one snapshot."""
        await self._emit(
            "EventReceived",
            snapshot.machine_id,
            None,
            {"event_type": event.type, "state": snapshot.current_state},
        )
        candidates = self._candidate_transitions(snapshot, event)
        legal: list[tuple[TransitionSpec, list[DecisionResult]]] = []
        all_guard_results: list[DecisionResult] = []
        for transition in candidates:
            guard_results = await self._evaluate_guards(transition, snapshot, event)
            all_guard_results.extend(guard_results)
            if all(result.allowed for result in guard_results):
                legal.append((transition, guard_results))

        if not legal:
            result = StepResult(
                event_id=event.id,
                source_state=snapshot.current_state,
                target_state=snapshot.current_state,
                status="no_transition",
                guard_results=all_guard_results,
                new_snapshot=self._advance_snapshot(snapshot, snapshot.current_state),
                trace={"candidate_transitions": [item.name for item in candidates]},
            )
            await self._emit_step_completed(result)
            return result

        selected, random_record = self._select_transition(snapshot, legal)
        selected_transition, selected_guard_results = selected
        await self._emit(
            "TransitionSelected",
            snapshot.machine_id,
            None,
            {
                "transition": selected_transition.name,
                "source": selected_transition.source,
                "target": selected_transition.target,
            },
        )
        new_snapshot = await self._apply_transition(
            snapshot,
            selected_transition,
            event,
            result_step_id=None,
        )
        invariant_results = await self._check_invariants(new_snapshot, event)
        violations = [
            result.id
            for result in invariant_results
            if not result.allowed
            and self.spec.states[new_snapshot.current_state].invariants
        ]
        actions = await self._handle_actions(selected_transition, new_snapshot, event)
        outputs = await self._produce_outputs(selected_transition, new_snapshot, event)
        status: StepStatus = "blocked" if violations else "transitioned"
        result = StepResult(
            event_id=event.id,
            source_state=snapshot.current_state,
            target_state=new_snapshot.current_state,
            status=status,
            selected_transition=selected_transition.name,
            guard_results=selected_guard_results,
            invariant_results=invariant_results,
            outputs=outputs,
            actions=actions,
            violations=violations,
            random=random_record,
            trace={
                "candidate_transitions": [item.name for item in candidates],
                "legal_transitions": [item[0].name for item in legal],
            },
            new_snapshot=new_snapshot,
        )
        await self._emit_step_completed(result)
        return result

    def _candidate_transitions(
        self,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> list[TransitionSpec]:
        state = self.spec.states[snapshot.current_state]
        names = [*state.transitions, *state.recovery_transitions]
        candidates = []
        for name in names:
            transition = self.spec.transitions[name]
            if transition.trigger.type in (event.type, "*"):
                candidates.append(transition)
        return candidates

    async def _evaluate_guards(
        self,
        transition: TransitionSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> list[DecisionResult]:
        results: list[DecisionResult] = []
        for guard in transition.guards:
            if guard.kind == "deterministic" and guard.ref:
                ctx = self._context(snapshot, event, transition)
                result = _decision_from_result(
                    guard.id,
                    await _invoke(
                        self.definition.bindings.resolve(guard.ref, "guard"),
                        ctx,
                        guard.args,
                    ),
                )
            elif self.guard_evaluator is None:
                result = self._default_guard_result(guard, event)
            else:
                result = await self.guard_evaluator.evaluate_guard(
                    guard,
                    spec=self.spec,
                    snapshot=snapshot,
                    event=event,
                )
            threshold = (
                guard.min_confidence
                if guard.min_confidence is not None
                else self.spec.policy.confidence.min_guard_confidence
            )
            if result.confidence < threshold:
                result = result.model_copy(
                    update={
                        "allowed": False,
                        "uncertainties": [
                            *result.uncertainties,
                            f"confidence below threshold {threshold}",
                        ],
                    }
                )
            results.append(result)
            await self._emit(
                "GuardEvaluated",
                snapshot.machine_id,
                None,
                {
                    "id": result.id,
                    "allowed": result.allowed,
                    "confidence": result.confidence,
                },
            )
        return results

    async def _check_invariants(
        self,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> list[DecisionResult]:
        state = self.spec.states[snapshot.current_state]
        results: list[DecisionResult] = []
        for invariant in state.invariants:
            if invariant.kind == "deterministic" and invariant.ref:
                ctx = self._context(snapshot, event, None)
                result = _decision_from_result(
                    invariant.id,
                    await _invoke(
                        self.definition.bindings.resolve(
                            invariant.ref,
                            "invariant",
                        ),
                        ctx,
                        invariant.args,
                    ),
                )
            elif self.invariant_checker is None:
                result = self._default_invariant_result(invariant, snapshot, event)
            else:
                result = await self.invariant_checker.check_invariant(
                    invariant,
                    spec=self.spec,
                    snapshot=snapshot,
                    event=event,
                )
            threshold = (
                invariant.min_confidence
                if invariant.min_confidence is not None
                else self.spec.policy.confidence.min_invariant_confidence
            )
            if result.confidence < threshold:
                result = result.model_copy(
                    update={
                        "allowed": False,
                        "uncertainties": [
                            *result.uncertainties,
                            f"confidence below threshold {threshold}",
                        ],
                    }
                )
            results.append(result)
            await self._emit(
                "InvariantChecked",
                snapshot.machine_id,
                None,
                {"id": result.id, "allowed": result.allowed},
            )
        return results

    async def _handle_actions(
        self,
        transition: TransitionSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in transition.actions:
            if self.spec.policy.execution == "dry_run" or self.action_executor is None:
                if self.spec.policy.execution == "execute":
                    executed = await self._execute_bound_action(action, snapshot, event)
                    result = executed or ActionResult(
                        action_name=action.name or action.tool or action.ref or "action",
                        tool=action.tool or action.ref or "python",
                        status="planned",
                        message="action planned but no executor/binding was configured",
                    )
                else:
                    result = ActionResult(
                        action_name=action.name or action.tool or action.ref or "action",
                        tool=action.tool or action.ref or "python",
                        status="planned",
                        message="action planned but not executed",
                    )
            else:
                result = await self.action_executor.execute_action(
                    action,
                    spec=self.spec,
                    snapshot=snapshot,
                    event=event,
                )
            results.append(result)
        return results

    def _select_transition(
        self,
        snapshot: MachineSnapshot,
        legal: list[tuple[TransitionSpec, list[DecisionResult]]],
    ) -> tuple[tuple[TransitionSpec, list[DecisionResult]], dict[str, object]]:
        if len(legal) == 1 and legal[0][0].weight is None:
            return legal[0], {}
        if not any(item[0].weight is not None for item in legal):
            return legal[0], {}
        rng = random.Random(snapshot.random_seed)
        for _ in range(snapshot.random_draws):
            rng.random()
        weights = [item[0].weight or 1.0 for item in legal]
        total = sum(weights)
        draw = rng.random()
        threshold = draw * total
        cumulative = 0.0
        selected_index = len(legal) - 1
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold <= cumulative:
                selected_index = index
                break
        record: dict[str, object] = {
            "seed": snapshot.random_seed,
            "draw_index": snapshot.random_draws,
            "draw": draw,
            "weights": {
                legal[index][0].name: weights[index] for index in range(len(legal))
            },
        }
        return legal[selected_index], record

    async def _execute_bound_action(
        self,
        action: ActionSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> ActionResult | None:
        ctx = self._context(snapshot, event, None)
        if action.kind == "deterministic" and action.ref:
            payload = await _invoke(
                self.definition.bindings.resolve(action.ref, "action"),
                ctx,
                action.args,
            )
            return _action_from_result(action, payload)
        if action.kind == "tool" and action.tool and self.definition.bindings.tools:
            arguments = dict(action.arguments)
            arguments_ref = action.args.get("arguments_ref")
            if isinstance(arguments_ref, str):
                produced = await _invoke(
                    self.definition.bindings.resolve(arguments_ref, "action"),
                    ctx,
                    action.args,
                )
                if isinstance(produced, dict):
                    arguments.update(produced)
            output = await ctx.call_tool(action.tool, **arguments)
            return ActionResult(
                action_name=action.name or action.tool,
                tool=action.tool,
                status="executed",
                output=output if isinstance(output, dict) else {"value": output},
            )
        return None

    async def _produce_outputs(
        self,
        transition: TransitionSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> list[OutputRecord]:
        records: list[OutputRecord] = []
        for output in transition.emits:
            payload: dict[str, Any] = {}
            if output.kind == "deterministic" and output.ref:
                produced = await _invoke(
                    self.definition.bindings.resolve(output.ref, "output"),
                    self._context(snapshot, event, transition),
                    output.args,
                )
                if isinstance(produced, OutputRecord):
                    records.append(produced)
                    continue
                if isinstance(produced, dict):
                    payload = produced
                else:
                    payload = {"value": produced}
            elif output.kind == "nl":
                payload = {"instruction": output.text}
            records.append(
                OutputRecord(
                    type=output.type,
                    payload=payload,
                    description=output.description,
                    destination=output.destination,
                )
            )
        return records

    async def _apply_transition(
        self,
        snapshot: MachineSnapshot,
        transition: TransitionSpec,
        event: FSLMEvent,
        result_step_id: str | None,
    ) -> MachineSnapshot:
        variables = dict(snapshot.variables)
        variables.update(_resolve_direct_effects(transition.effects, variables))
        for effect in transition.effect_calls:
            ctx = self._context(snapshot, event, transition, step_id=result_step_id)
            patch = await _invoke(
                self.definition.bindings.resolve(effect.ref, "effect"),
                ctx,
                effect.args,
            )
            if isinstance(patch, dict):
                variables.update(patch)
        return self._advance_snapshot(
            snapshot,
            transition.target,
            variables=variables,
            consumed_random=transition.weight is not None,
        )

    def _context(
        self,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
        transition: TransitionSpec | None,
        *,
        step_id: str | None = None,
    ) -> FSLMContext:
        return FSLMContext(
            spec=self.spec,
            definition=self.definition,
            state=self.spec.states[snapshot.current_state],
            transition=transition,
            snapshot=snapshot,
            event=event,
            vars=snapshot.variables,
            step_id=step_id or "",
        )

    @staticmethod
    def _advance_snapshot(
        snapshot: MachineSnapshot,
        state: str,
        *,
        variables: dict[str, object] | None = None,
        consumed_random: bool = False,
    ) -> MachineSnapshot:
        return snapshot.model_copy(
            update={
                "id": uuid4().hex,
                "current_state": state,
                "variables": variables if variables is not None else snapshot.variables,
                "step_index": snapshot.step_index + 1,
                "random_draws": snapshot.random_draws + (1 if consumed_random else 0),
            }
        )

    @staticmethod
    def _default_guard_result(guard: GuardSpec, event: FSLMEvent) -> DecisionResult:
        if guard.kind == "nl":
            return DecisionResult(
                id=guard.id,
                allowed=False,
                confidence=0.0,
                uncertainties=["no natural-language guard evaluator configured"],
            )
        guard_values = event.payload.get("guards", {})
        allowed = bool(guard_values.get(guard.id, False)) if isinstance(guard_values, dict) else False
        return DecisionResult(id=guard.id, allowed=allowed)

    @staticmethod
    def _default_invariant_result(
        invariant: InvariantSpec,
        snapshot: MachineSnapshot,
        event: FSLMEvent,
    ) -> DecisionResult:
        if invariant.kind == "nl":
            return DecisionResult(
                id=invariant.id,
                allowed=False,
                confidence=0.0,
                uncertainties=["no natural-language invariant checker configured"],
            )
        invariant_values = event.payload.get("invariants")
        if not isinstance(invariant_values, dict):
            invariant_values = snapshot.variables.get("invariants", {})
        allowed = bool(invariant_values.get(invariant.id, True)) if isinstance(invariant_values, dict) else True
        return DecisionResult(id=invariant.id, allowed=allowed)

    async def _emit(
        self,
        type: str,
        machine_id: str,
        step_id: str | None,
        payload: dict[str, object],
    ) -> None:
        if not self.observers:
            return
        record = FSLMEventRecord(
            type=type,
            machine_id=machine_id,
            step_id=step_id,
            payload=payload,
        )
        for observer in self.observers:
            await observer.on_event(record)

    async def _emit_step_completed(self, result: StepResult) -> None:
        await self._emit(
            "StepCompleted",
            result.new_snapshot.machine_id,
            result.step_id,
            {"status": result.status, "state": result.new_snapshot.current_state},
        )


__all__ = [
    "ActionExecutor",
    "FSLMKernel",
    "GuardEvaluator",
    "InvariantChecker",
]


async def _invoke(fn: Any, ctx: FSLMContext, args: dict[str, Any]) -> Any:
    signature = inspect.signature(fn)
    result = fn(ctx, args) if len(signature.parameters) >= 2 else fn(ctx)
    if inspect.isawaitable(result):
        return await result
    return result


def _decision_from_result(id: str, value: Any) -> DecisionResult:
    if isinstance(value, DecisionResult):
        return value
    if isinstance(value, bool):
        return DecisionResult(id=id, allowed=value)
    if isinstance(value, dict):
        return DecisionResult(id=id, allowed=bool(value.get("allowed", True)), metadata=value)
    return DecisionResult(id=id, allowed=bool(value), metadata={"value": value})


def _action_from_result(action: ActionSpec, value: Any) -> ActionResult:
    if isinstance(value, ActionResult):
        return value
    return ActionResult(
        action_name=action.name or action.ref or action.tool or "action",
        tool=action.tool or action.ref or "python",
        status="executed",
        output=value if isinstance(value, dict) else {"value": value},
    )


def _resolve_direct_effects(
    effects: dict[str, Any],
    variables: dict[str, Any],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in effects.items():
        if isinstance(value, dict) and value.get("op") == "increment":
            path = str(value.get("path", key))
            current_key = path.removeprefix("$.snapshot.variables.")
            current = variables.get(current_key, 0)
            resolved[key] = current + value.get("by", 1)
        else:
            resolved[key] = value
    return resolved
