"""Bounded transaction-sequence exploration.

A generated sequence is a plan. It is not a finding, a counterexample, or a
reproduction. Budgets only tighten, including when a planner asks for more.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.parsing.solidity_arguments import (
    ArgumentCandidate,
    StateObservation,
    candidates_for_type,
    combine,
    derive_from_state,
)
from app.parsing.solidity_identity import function_name_for
from app.parsing.solidity_ir import SemanticFunction, SemanticProgram
from app.parsing.solidity_spec import VerificationSpecification, function_signature
from app.parsing.solidity_state_transitions import CandidatePath

MAX_SEQUENCE_LENGTH = 4
MAX_EXECUTIONS = 8
MAX_MUTATIONS = 4
MAX_ARGUMENT_CANDIDATES = 4
MAX_ACTORS = 4
MAX_ARTIFACT_BYTES = 16_000
MAX_SNAPSHOTS = 4
WALL_CLOCK_SECONDS = 60.0

_ACTORS = ("user", "user2", "attacker", "callback")


@dataclass(frozen=True)
class ExplorationBounds:
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    max_executions: int = MAX_EXECUTIONS
    wall_clock_seconds: float = WALL_CLOCK_SECONDS
    max_mutations: int = MAX_MUTATIONS
    max_argument_candidates: int = MAX_ARGUMENT_CANDIDATES
    max_actors: int = MAX_ACTORS
    max_artifact_bytes: int = MAX_ARTIFACT_BYTES
    max_snapshots: int = MAX_SNAPSHOTS


@dataclass(frozen=True)
class PlannedCall:
    function: str
    actor: str
    arguments: tuple[ArgumentCandidate, ...]
    value: str
    direction: str
    assumptions: tuple[str, ...]
    score: int
    contract: str = ""
    function_identity: str = ""


@dataclass(frozen=True)
class PlannedSequence:
    sequence_id: str
    path_id: str
    calls: tuple[PlannedCall, ...]
    status: str
    assumptions: tuple[str, ...]
    origin: str
    project: str = ""
    target: str = ""


def default_bounds() -> ExplorationBounds:
    return ExplorationBounds()


def clamp_bounds(requested: ExplorationBounds | None) -> ExplorationBounds:
    """A planner may only tighten these limits."""
    base = default_bounds()
    if requested is None:
        return base
    return ExplorationBounds(
        max_sequence_length=_tighten(requested.max_sequence_length, base.max_sequence_length),
        max_executions=_tighten(requested.max_executions, base.max_executions),
        wall_clock_seconds=min(float(requested.wall_clock_seconds), base.wall_clock_seconds),
        max_mutations=_tighten(requested.max_mutations, base.max_mutations),
        max_argument_candidates=_tighten(
            requested.max_argument_candidates, base.max_argument_candidates
        ),
        max_actors=_tighten(requested.max_actors, base.max_actors),
        max_artifact_bytes=_tighten(requested.max_artifact_bytes, base.max_artifact_bytes),
        max_snapshots=_tighten(requested.max_snapshots, base.max_snapshots),
    )


def explore_sequences(
    path: CandidatePath,
    spec: VerificationSpecification,
    program: SemanticProgram | None,
    source: str,
    *,
    bounds: ExplorationBounds | None = None,
    observations: tuple[StateObservation, ...] = (),
) -> tuple[PlannedSequence, ...]:
    """Deterministic mutations of one candidate path. Execution count stays capped."""
    active = clamp_bounds(bounds)
    base = _base_sequence(path, spec, program, source, active, observations)
    if base is None:
        return ()
    ranked = _mutations(base, path, spec, program, source, active, observations)
    selected = [base, *ranked[: max(0, active.max_mutations - 1)]]
    selected = selected[: active.max_executions]
    selected.sort(key=lambda item: item.sequence_id)
    return tuple(selected)


def describe_call(
    program: SemanticProgram, source: str, contract: str, name: str
) -> PlannedCall | None:
    """One call with an explicit actor and a recorded value direction."""
    fact = function_signature(source, contract, name, program)
    if fact is None:
        return None
    return _call_from_fact(fact, "user", 0, program)


def explore_sequences_budget(requested_executions: int) -> int:
    """Reject a planner request that would raise the execution cap."""
    return min(max(1, int(requested_executions)), MAX_EXECUTIONS)


def _base_sequence(
    path: CandidatePath,
    spec: VerificationSpecification,
    program: SemanticProgram | None,
    source: str,
    bounds: ExplorationBounds,
    observations: tuple[StateObservation, ...],
) -> PlannedSequence | None:
    names = _function_names(path, program, source, spec.contract)
    if not names or len(names) > bounds.max_sequence_length:
        return None
    calls: list[PlannedCall] = []
    for name in names:
        fact = function_signature(source, spec.contract, name, program)
        if fact is None:
            return None
        call = _call_from_fact(fact, "user", _interest(fact, path), program, observations)
        if call is None:
            return None
        calls.append(call)
    assumptions = (
        "a generated sequence is not a finding",
        "an actor is not proof of authorization",
    )
    return PlannedSequence(
        _sequence_id(path.path_id, tuple(calls), "static-candidate"),
        path.path_id,
        tuple(calls),
        "planned",
        assumptions,
        "static-candidate",
    )


def _mutations(
    base: PlannedSequence,
    path: CandidatePath,
    spec: VerificationSpecification,
    program: SemanticProgram | None,
    source: str,
    bounds: ExplorationBounds,
    observations: tuple[StateObservation, ...],
) -> list[PlannedSequence]:
    found: list[PlannedSequence] = []
    calls = list(base.calls)
    if len(calls) >= 2:
        swapped = [calls[1], calls[0], *calls[2:]]
        found.append(_variant(path, swapped, "order"))
    if len(calls) + 1 <= bounds.max_sequence_length:
        found.append(_variant(path, [calls[0], *calls], "repeat"))
    if len(_ACTORS) > 1:
        actor = "user2" if calls[0].actor == "user" else "user"
        changed = replace(calls[0], actor=actor)
        found.append(_variant(path, [changed, *calls[1:]], "actor"))
    if calls and calls[0].value == "0" and "payable" in calls[0].assumptions:
        valued = replace(calls[0], value="1", direction="inbound")
        found.append(_variant(path, [valued, *calls[1:]], "value"))
    alternative = _alternate_arguments(calls[0], bounds)
    if alternative is not None:
        found.append(_variant(path, [alternative, *calls[1:]], "arguments"))
    prerequisite = _prerequisite(path, spec, program, source, bounds, observations)
    if prerequisite is not None and len(calls) + 1 <= bounds.max_sequence_length:
        found.append(_variant(path, [prerequisite, *calls], "prerequisite"))
    if path.path_id.startswith("reentrancy:"):
        callback = replace(
            calls[0],
            actor="callback",
            assumptions=(*calls[0].assumptions, "callback actor is not an authorization proof"),
        )
        found.append(_variant(path, [callback], "callback"))
    found.sort(key=lambda item: (-_score(item), item.sequence_id))
    return found


def _call_from_fact(
    fact: dict[str, object],
    actor: str,
    score: int,
    program: SemanticProgram | None,
    observations: tuple[StateObservation, ...] = (),
) -> PlannedCall | None:
    del program
    types = fact.get("parameter_types")
    if not isinstance(types, tuple):
        types = ()
    arguments: tuple[ArgumentCandidate, ...] = ()
    if types:
        rows = combine(tuple(str(item) for item in types))
        if rows is None:
            return None
        arguments = rows[0]
        derived = _derived_row(arguments, observations)
        if derived is not None:
            arguments = derived
    payable = bool(fact.get("payable"))
    value = "1" if payable else "0"
    direction = "inbound" if payable else "unknown"
    assumptions = [
        "actor is explicit and is not proof of authorization",
    ]
    if fact.get("uses_sender"):
        assumptions.append("msg.sender is the named actor, not an impersonated privileged account")
    if payable:
        assumptions.append("payable")
        assumptions.append("transaction value is inbound; further economic setup is unsupported")
    elif fact.get("uses_value"):
        return None
    return PlannedCall(
        str(fact.get("name", "")),
        actor if actor in _ACTORS else "user",
        arguments,
        value,
        direction,
        tuple(assumptions),
        score,
        contract=str(fact.get("contract", "")),
        function_identity=str(fact.get("identity", "")),
    )


def _derived_row(
    arguments: tuple[ArgumentCandidate, ...], observations: tuple[StateObservation, ...]
) -> tuple[ArgumentCandidate, ...] | None:
    if not observations or not arguments:
        return None
    derived = derive_from_state(observations[0])
    if derived is None:
        return None
    if arguments[0].type_name.startswith("uint") or arguments[0].type_name.startswith("int"):
        replacement = ArgumentCandidate(
            derived.literal,
            f"state:{derived.observation}:{derived.snapshot_id}",
            arguments[0].type_name,
        )
        return (replacement, *arguments[1:])
    return None


def _alternate_arguments(call: PlannedCall, bounds: ExplorationBounds) -> PlannedCall | None:
    if not call.arguments:
        return None
    type_name = call.arguments[0].type_name
    found = candidates_for_type(type_name, limit=bounds.max_argument_candidates)
    if not found or len(found) < 2:
        return None
    replacement = found[1] if found[1].literal != call.arguments[0].literal else found[0]
    if replacement.literal == call.arguments[0].literal:
        return None
    return replace(call, arguments=(replacement, *call.arguments[1:]))


def _prerequisite(
    path: CandidatePath,
    spec: VerificationSpecification,
    program: SemanticProgram | None,
    source: str,
    bounds: ExplorationBounds,
    observations: tuple[StateObservation, ...],
) -> PlannedCall | None:
    del bounds, observations
    if program is None:
        return None
    wanted = set(path.operation_ids)
    if not wanted:
        return None
    for function in program.functions:
        if function.contract != spec.contract:
            continue
        known = _function_names(path, program, source, spec.contract) or ()
        if function.name in known:
            continue
        writes = {item.operation_id for item in function.access_sites if item.kind != "read"}
        if not writes:
            continue
        fact = function_signature(source, spec.contract, function.name, program)
        if fact is None or fact.get("visibility") not in {"public", "external"}:
            continue
        call = _call_from_fact(fact, "user", 1, program)
        if call is not None:
            return call
    return None


def _function_names(
    path: CandidatePath, program: SemanticProgram | None, source: str, contract: str
) -> tuple[str, ...] | None:
    names: list[str] = []
    for function_id in path.function_ids:
        name = function_name_for(function_id, program=program, source=source, contract=contract)
        if not name:
            return None
        names.append(name)
    return tuple(names)


def _interest(fact: dict[str, object], path: CandidatePath) -> int:
    score = 0
    if fact.get("visibility") in {"public", "external"}:
        score += 1
    if path.operation_ids:
        score += 1
    return score


def _variant(path: CandidatePath, calls: list[PlannedCall], origin: str) -> PlannedSequence:
    return PlannedSequence(
        _sequence_id(path.path_id, tuple(calls), origin),
        path.path_id,
        tuple(calls),
        "planned",
        (
            "a generated sequence is not a finding",
            "an actor is not proof of authorization",
        ),
        origin,
    )


def _score(sequence: PlannedSequence) -> int:
    return sum(item.score for item in sequence.calls)


def _sequence_id(path_id: str, calls: tuple[PlannedCall, ...], origin: str) -> str:
    parts = [path_id, origin]
    for call in calls:
        rendered = ",".join(item.literal for item in call.arguments)
        parts.append(f"{call.actor}:{call.function}:{rendered}:{call.value}:{call.direction}")
    return "|".join(parts)


def _tighten(requested: int, cap: int) -> int:
    return max(1, min(int(requested), cap))


_ECONOMIC_ACTIONS = (
    "deposit",
    "mint",
    "withdraw",
    "redeem",
    "transfer",
    "transferFrom",
    "approve",
    "permit",
    "donate",
    "swap",
    "addLiquidity",
    "removeLiquidity",
    "borrow",
    "repay",
    "liquidate",
    "oracleRead",
    "callback",
    "flashBorrow",
    "flashRepay",
)

_REFINEMENTS = {
    "share_balance_changed": ("withdraw", "redeem"),
    "reserve_changed": ("swap",),
    "attacker_balance_increased": ("withdraw", "redeem", "swap"),
}


def established_actions(
    source: str,
    contract: str,
    program: SemanticProgram | None,
    established: frozenset[str],
) -> tuple[str, ...]:
    """Name an action only when that exact function and semantic are established."""
    found: list[str] = []
    for name in _ECONOMIC_ACTIONS:
        if name not in established:
            continue
        fact = function_signature(source, contract, name, program)
        if fact is None:
            continue
        found.append(name)
    return tuple(found)


def bind_sequence(sequence: PlannedSequence, *, project: str, target: str) -> PlannedSequence:
    return replace(sequence, project=project, target=target)


def sequence_bound(sequence: PlannedSequence, *, project: str, target: str) -> bool:
    return sequence.project == project and sequence.target == target


def refine_actions(observation_kind: str, established: frozenset[str]) -> tuple[str, ...]:
    """Deterministic next actions. This does not call a model."""
    return tuple(name for name in _REFINEMENTS.get(observation_kind, ()) if name in established)


@dataclass(frozen=True)
class AbstractHypothesis:
    """A research note. It is not an executable call."""

    action: str
    reason: str
    executable: bool


def economic_mutations(
    sequence: PlannedSequence,
    *,
    bounds: ExplorationBounds | None,
    established: frozenset[str],
    source: str = "",
    contract: str = "",
    program: SemanticProgram | None = None,
) -> tuple[PlannedSequence, ...]:
    """Bounded mutations. A request for a larger budget is clamped."""
    active = clamp_bounds(bounds)
    if not sequence.calls:
        return ()
    produced: list[PlannedSequence] = []
    first = sequence.calls[0]
    if first.arguments:
        for literal, provenance in (
            ("0", "zero"),
            ("1", "one"),
            ("type(uint256).max", "max"),
            ("1", "boundary"),
        ):
            produced.append(_replace_amount(sequence, literal, provenance))
    if len(sequence.calls) > 1:
        produced.append(_retarget(sequence, tuple(reversed(sequence.calls)), "reverse"))
    actors = ("user", "attacker")
    for actor in actors:
        if actor != first.actor:
            produced.append(
                _retarget(
                    sequence, (replace(first, actor=actor), *sequence.calls[1:]), f"actor-{actor}"
                )
            )
            break
    inserts = (
        ("donate", "donation-before-conversion"),
        ("swap", "swap-before-valuation"),
        ("borrow", "borrow-before-liquidation"),
        ("deposit", "deposit-before-withdraw"),
        ("withdraw", "withdraw-before-deposit"),
        ("approve", "approval-before-transferFrom"),
        ("permit", "permit-before-transferFrom"),
    )
    for name, origin in inserts:
        if name in established and len(sequence.calls) < active.max_sequence_length:
            prefixed = _prefix(sequence, name, origin, source, contract, program)
            if prefixed is not None:
                produced.append(prefixed)
    if "deposit" in established or "mint" in established:
        produced.append(_retarget(sequence, sequence.calls + sequence.calls[:1], "repeat"))
    unique: list[PlannedSequence] = []
    seen: set[str] = set()
    for item in produced:
        if item.sequence_id in seen:
            continue
        seen.add(item.sequence_id)
        unique.append(item)
        if len(unique) >= active.max_mutations:
            break
    return tuple(unique)


def _replace_amount(sequence: PlannedSequence, literal: str, provenance: str) -> PlannedSequence:
    first = sequence.calls[0]
    argument = replace(first.arguments[0], literal=literal, provenance=provenance)
    call = replace(first, arguments=(argument, *first.arguments[1:]))
    return _retarget(sequence, (call, *sequence.calls[1:]), f"amount-{provenance}")


def action_hypothesis(
    name: str, *, source: str, contract: str, program: SemanticProgram | None
) -> AbstractHypothesis:
    """An action name is a hypothesis until a signature and arguments exist."""
    fact = function_signature(source, contract, name, program) if source and contract else None
    if fact is None:
        return AbstractHypothesis(name, "no callable signature was established", False)
    types = fact.get("parameter_types")
    if isinstance(types, tuple) and types and combine(tuple(str(item) for item in types)) is None:
        return AbstractHypothesis(name, "arguments cannot be constructed", False)
    return AbstractHypothesis(name, "signature and arguments are established", True)


def _prefix(
    sequence: PlannedSequence,
    name: str,
    origin: str,
    source: str,
    contract: str,
    program: SemanticProgram | None,
) -> PlannedSequence | None:
    hypothesis = action_hypothesis(name, source=source, contract=contract, program=program)
    if not hypothesis.executable:
        return None
    fact = function_signature(source, contract, name, program)
    if fact is None:
        return None
    call = _call_from_fact(fact, sequence.calls[0].actor, 0, program)
    if call is None:
        return None
    if fact.get("parameter_types") and not call.arguments:
        return None
    call = replace(
        call, contract=contract, function_identity=str(fact.get("identity", call.function_identity))
    )
    return _retarget(sequence, (call, *sequence.calls), origin)


def _retarget(
    sequence: PlannedSequence, calls: tuple[PlannedCall, ...], origin: str
) -> PlannedSequence:
    return replace(
        sequence,
        calls=calls,
        origin=origin,
        sequence_id=_sequence_id(sequence.path_id, calls, origin),
        status="planned",
    )


def function_interest(function: SemanticFunction) -> int:
    """Bias toward state, calls, and token-like callees. A score is not exploitability."""
    score = 0
    if function.writes:
        score += 2
    if function.reads:
        score += 1
    if any(site.external for site in function.call_sites):
        score += 1
    if "msg.sender" in (function.source or ""):
        score += 1
    return score
