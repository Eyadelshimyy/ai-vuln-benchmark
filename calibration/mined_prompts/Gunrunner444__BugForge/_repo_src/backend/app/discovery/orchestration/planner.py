"""Deterministic candidate ranking.

The planner reads typed state and returns one decision. It does not run an
engine, it does not edit a budget, and it never uses a model. Identical state
and inputs always produce the same decision regardless of engine order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from app.discovery.capabilities import EngineAvailability, EngineCapability
from app.discovery.engine import AnalysisRequest, DiscoveryEngine
from app.discovery.orchestration.assess import W_SUGGESTION, actionable
from app.discovery.orchestration.budget import estimate_cost
from app.discovery.orchestration.codec import digest
from app.discovery.orchestration.gate import ExecutionGate, check_gate
from app.discovery.orchestration.model import (
    ACTIONABLE,
    ATTEMPTED,
    CampaignIdentity,
    Candidate,
    CapabilityNeed,
    Complementarity,
    ExecutionPhase,
    NeedStatus,
    ResearchState,
    StopReason,
    Uncertainty,
)
from app.discovery.scheduler import DiscoveryScheduler, capability_for, engine_rank
from app.discovery.sequences import MAX_EXECUTIONS

RESEARCH_CAPABILITY_NAMES: tuple[str, ...] = (
    "bounty_context_analysis",
    "caller_context_analysis",
    "oracle_quality_analysis",
    "proof_binding_analysis",
    "account_abstraction_analysis",
    "accounting_analysis",
    "compiler_advisory_analysis",
    "compiler_differential_validation",
    "vfcs_generation",
)

# Capability -> the request fields the orchestrator sets. Only "mode" is ever set.
CAPABILITY_MODE: dict[str, str] = {
    "static_analysis": "",
    "cross_contract_analysis": "",
    "economic_simulation": "",
    "fuzzing": "fuzz",
    "symbolic_execution": "symbolic",
    "test_execution": "test",
    "property_testing": "invariant",
    "invariant_testing": "invariant",
    "runtime_validation": "local",
    "fork_validation": "fork",
    "differential_validation": "differential",
    **{name: name for name in RESEARCH_CAPABILITY_NAMES},
}
ORCHESTRATABLE = frozenset(CAPABILITY_MODE)

_RUNTIME_FAMILY = frozenset({"runtime_validation", "fork_validation", "differential_validation"})
_NEEDS_SNAPSHOT = frozenset(
    {"cross_contract_analysis", "economic_simulation", "compiler_differential_validation"}
    | _RUNTIME_FAMILY
)
_NEEDS_SOURCES = frozenset(set(RESEARCH_CAPABILITY_NAMES) - {"bounty_context_analysis"})
_NEEDS_TARGET = frozenset(
    {"fuzzing", "symbolic_execution", "test_execution", "property_testing", "invariant_testing"}
)
# Evidence categories from other capabilities that change what a run could learn.
_DEPENDS: dict[str, frozenset[str]] = {
    "fuzzing": frozenset({"finding"}),
    "symbolic_execution": frozenset({"finding"}),
    "test_execution": frozenset({"finding"}),
    "runtime_validation": frozenset({"finding", "economic"}),
    "fork_validation": frozenset({"finding", "economic", "runtime"}),
    "differential_validation": frozenset({"finding", "economic", "runtime"}),
    "vfcs_generation": frozenset({"finding"}),
    "compiler_differential_validation": frozenset({"finding"}),
}
_SEEDED = frozenset({"fuzzing", "symbolic_execution", "test_execution"})
_VOLATILE_EXTRA = frozenset({"campaign_id", "execution_id", "timestamp"})

W_COMPLEMENT = {
    Complementarity.CONTRADICTION_RESOLVING: 30,
    Complementarity.DISCRIMINATIVE: 25,
    Complementarity.NOVEL: 20,
    Complementarity.PREREQUISITE_GENERATING: 18,
    Complementarity.COVERAGE_EXPANDING: 15,
    Complementarity.CORROBORATIVE: 10,
    Complementarity.REDUNDANT: -50,
}
DIVERSITY_BONUS = 8
SUGGESTION_BONUS = 5
CIRCUIT_FAILURES = 2

_BLOCK_PRECEDENCE: tuple[tuple[str, StopReason], ...] = (
    ("gate:safety", StopReason.SAFETY_BLOCKED),
    ("gate:scope", StopReason.SCOPE_BLOCKED),
    ("gate:approval", StopReason.APPROVAL_REQUIRED),
    ("identity:", StopReason.DETERMINISTIC_IDENTITY_MISSING),
    ("budget:", StopReason.BUDGET_EXHAUSTED),
    ("prerequisite:", StopReason.PREREQUISITES_UNAVAILABLE),
    ("scheduler:", StopReason.PREREQUISITES_UNAVAILABLE),
    ("environment:", StopReason.ENVIRONMENT_UNAVAILABLE),
)


@dataclass(frozen=True)
class Suggestion:
    """A request from Cursor. It can add a candidate; it can never bypass a check."""

    capability: str
    engine: str = ""
    reason: str = ""


@dataclass(frozen=True)
class Plan:
    selected: Candidate | None
    considered: tuple[Candidate, ...]
    needs: tuple[CapabilityNeed, ...]
    stop: StopReason | None = None
    rationale: str = ""
    request: AnalysisRequest | None = None
    rejected_suggestions: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    gate_detail: str = ""
    accepted_suggestions: tuple[str, ...] = field(default=())


def build_request(
    base: AnalysisRequest,
    capability: str,
    identity: CampaignIdentity,
    scheduler: DiscoveryScheduler,
) -> AnalysisRequest:
    """The request an engine would receive. Only `mode` and the shared corpus are set."""
    extra = dict(base.extra)
    mode = CAPABILITY_MODE.get(capability, "")
    if mode:
        extra["mode"] = mode
    else:
        extra.pop("mode", None)
    return replace(
        base,
        campaign_id=base.campaign_id or identity.campaign_id,
        difficult=base.difficult or capability == "symbolic_execution",
        corpus=scheduler.corpus,
        extra=extra,
    )


def input_digest(request: AnalysisRequest, capability: str, state: ResearchState) -> str:
    depends = _DEPENDS.get(capability, frozenset())
    upstream = sorted(
        item.evidence_id
        for item in state.evidence
        if item.attrs.get("category") in depends and item.capability != capability
    )
    seeds = sorted(state.corpus_refs) if capability in _SEEDED else []
    extra = {k: v for k, v in request.extra.items() if k not in _VOLATILE_EXTRA}
    return digest(
        {
            "capability": capability,
            "language": request.language,
            "target": request.target,
            "contract": request.contract,
            "function": request.function,
            "source_file": request.source_file,
            "files": sorted(request.files),
            "extra": extra,
            "upstream": upstream,
            "seeds": seeds,
        }
    )


def execution_key(
    identity: CampaignIdentity, engine: str, capability: str, input_hash: str, retries: int = 0
) -> str:
    return f"ex_{digest((identity.target_digest(), engine, capability, input_hash, retries))}"


def prerequisite(capability: str, request: AnalysisRequest, identity: CampaignIdentity) -> str:
    """The first thing missing, or empty. Nothing is invented to satisfy it."""
    extra: Mapping[str, str] = request.extra
    if capability in _NEEDS_TARGET and not (
        request.contract or request.function or request.source_file or request.has_harness
    ):
        return "identity:target"
    if capability in _NEEDS_SOURCES and not (request.files or request.source_file):
        return "identity:target"
    if capability in RESEARCH_CAPABILITY_NAMES and not extra.get("program_context"):
        return "identity:program_context"
    if capability in _NEEDS_SNAPSHOT:
        if not identity.source_snapshot:
            return "identity:source_snapshot"
        if not identity.compiler_configuration:
            return "identity:compiler_configuration"
    if capability == "economic_simulation" and not extra.get("case"):
        return "prerequisite:economic_case"
    if capability in _RUNTIME_FAMILY:
        if not identity.identity_key():
            return "identity:contract_and_function"
        if not extra.get("sequence_id"):
            return "prerequisite:sequence_id"
    if capability == "fork_validation" and not (
        extra.get("chain_id") and extra.get("fork_block") and extra.get("state_snapshot")
    ):
        return "prerequisite:pinned_fork"
    if capability == "differential_validation" and extra.get("state_reset") != "true":
        return "prerequisite:state_reset"
    return ""


def _runs(capability: str, request: AnalysisRequest) -> int:
    if capability not in _RUNTIME_FAMILY:
        return 1
    default = 2 if capability == "differential_validation" else 1
    raw = request.extra.get("executions", str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, min(value, MAX_EXECUTIONS))


def _complementarity(
    capability: str, need: CapabilityNeed, state: ResearchState, engine: str
) -> Complementarity:
    uncertainties = set(need.uncertainties)
    if Uncertainty.CONTRADICTION.value in uncertainties:
        return Complementarity.CONTRADICTION_RESOLVING
    if Uncertainty.SEED_FOLLOWUP.value in uncertainties:
        return Complementarity.COVERAGE_EXPANDING
    if uncertainties & {
        Uncertainty.ECONOMIC_RUNTIME.value,
        Uncertainty.DIVERGENCE.value,
        Uncertainty.REPEATABILITY.value,
        Uncertainty.FORK.value,
    }:
        return Complementarity.DISCRIMINATIVE
    if capability == "cross_contract_analysis":
        return Complementarity.PREREQUISITE_GENERATING
    done = state.completed(capability)
    if done:
        if any(item.engine == engine for item in done):
            return Complementarity.REDUNDANT
        return Complementarity.CORROBORATIVE
    return Complementarity.NOVEL


def validate_suggestions(
    suggestions: tuple[Suggestion, ...], engine_ids: frozenset[str]
) -> tuple[tuple[Suggestion, ...], tuple[str, ...]]:
    accepted: list[Suggestion] = []
    rejected: list[str] = []
    for item in suggestions:
        name = str(item.capability)
        try:
            capability = EngineCapability(name)
        except ValueError:
            rejected.append(f"unknown_capability:{digest(name)[:8]}")
            continue
        if capability.value not in ORCHESTRATABLE:
            rejected.append(f"{capability.value}:not_orchestratable")
        elif item.engine and item.engine not in engine_ids:
            rejected.append(f"{capability.value}:unregistered_engine")
        else:
            accepted.append(item)
    return tuple(accepted), tuple(rejected)


def plan(
    *,
    state: ResearchState,
    needs: tuple[CapabilityNeed, ...],
    scheduler: DiscoveryScheduler,
    request: AnalysisRequest,
    identity: CampaignIdentity,
    gate: ExecutionGate,
    suggestions: tuple[Suggestion, ...] = (),
) -> Plan:
    engines = {engine.engine_id: engine for engine in scheduler.engines}
    accepted, rejected = validate_suggestions(suggestions, frozenset(engines))
    wanted = list(actionable(needs))
    present = {need.capability for need in wanted}
    for item in accepted:
        if item.capability not in present:
            wanted.append(
                CapabilityNeed(
                    item.capability,
                    NeedStatus.LOW_VALUE_FOLLOWUP,
                    W_SUGGESTION,
                    ("external suggestion",),
                    (),
                )
            )
            present.add(item.capability)
    suggested = {(item.capability, item.engine) for item in accepted}

    considered: list[Candidate] = []
    for need in sorted(wanted, key=lambda n: (-n.weight, n.capability)):
        ordered = sorted(
            engines.values(),
            key=lambda e: (engine_rank(e.engine_id, request.language), e.engine_id),
        )
        for engine in ordered:
            if EngineCapability(need.capability) not in engine.capabilities():
                continue
            considered.append(
                _evaluate(
                    engine,
                    need,
                    state=state,
                    scheduler=scheduler,
                    request=request,
                    identity=identity,
                    gate=gate,
                    suggested=(need.capability, engine.engine_id) in suggested
                    or (need.capability, "") in suggested,
                )
            )
    has_support = {
        need.capability
        for need in wanted
        if any(EngineCapability(need.capability) in e.capabilities() for e in engines.values())
    }
    finalized = _finalize(needs, wanted, considered, has_support)
    eligible = [item for item in considered if item.eligible]
    if eligible:
        best = min(
            eligible,
            key=lambda c: (
                -c.score,
                engine_rank(c.engine, request.language),
                c.engine,
                c.capability,
            ),
        )
        built = build_request(request, best.capability, identity, scheduler)
        return Plan(
            selected=best,
            considered=tuple(considered),
            needs=finalized,
            rationale=_rationale(best, wanted),
            request=built,
            rejected_suggestions=rejected,
            prerequisites=(),
            accepted_suggestions=tuple(sorted(f"{s.capability}:{s.engine}" for s in accepted)),
        )
    stop, detail = stop_reason(considered, finalized)
    return Plan(
        selected=None,
        considered=tuple(considered),
        needs=finalized,
        stop=stop,
        rationale=detail,
        rejected_suggestions=rejected,
        prerequisites=tuple(
            sorted(
                {
                    c.rejection
                    for c in considered
                    if c.rejection.startswith(("prerequisite:", "identity:"))
                }
            )
        ),
        accepted_suggestions=tuple(sorted(f"{s.capability}:{s.engine}" for s in accepted)),
    )


def _evaluate(
    engine: DiscoveryEngine,
    need: CapabilityNeed,
    *,
    state: ResearchState,
    scheduler: DiscoveryScheduler,
    request: AnalysisRequest,
    identity: CampaignIdentity,
    gate: ExecutionGate,
    suggested: bool,
) -> Candidate:
    capability = need.capability
    built = build_request(request, capability, identity, scheduler)
    runs = _runs(capability, built)
    cost = estimate_cost(capability, runs=runs)
    digest_in = input_digest(built, capability, state)
    superseded = sum(
        1
        for item in state.executions
        if item.phase is ExecutionPhase.SUPERSEDED
        and item.engine == engine.engine_id
        and item.capability == capability
        and item.input_digest == digest_in
    )
    key = execution_key(identity, engine.engine_id, capability, digest_in, superseded)
    complement = _complementarity(capability, need, state, engine.engine_id)

    def reject(reason: str) -> Candidate:
        return Candidate(
            engine=engine.engine_id,
            capability=capability,
            eligible=False,
            rejection=reason,
            complementarity=complement,
            estimated_cost=cost,
            execution_key=key,
            input_digest=digest_in,
        )

    reason = _first_rejection(engine, capability, state, scheduler, built, identity, key, cost)
    if reason:
        return reject(reason)
    health = next((h for h in state.health if h.engine == engine.engine_id), None)
    components = {
        "need": need.weight,
        "complementarity": W_COMPLEMENT[complement],
        "cost": -_cost_units(cost),
        "engine_rank": -engine_rank(engine.engine_id, request.language),
        "diversity": 0 if engine.engine_id in state.exercised_engines else DIVERSITY_BONUS,
        "health": -(health.consecutive_failures * 5 + health.timeouts * 2) if health else 0,
        "suggestion": SUGGESTION_BONUS if suggested else 0,
    }
    candidate = Candidate(
        engine=engine.engine_id,
        capability=capability,
        eligible=True,
        complementarity=complement,
        score=sum(components.values()),
        components=components,
        estimated_cost=cost,
        execution_key=key,
        input_digest=digest_in,
    )
    verdict = check_gate(gate, candidate=candidate, request=built, state=state)
    if not verdict.allowed:
        reason = {
            StopReason.SCOPE_BLOCKED: "gate:scope",
            StopReason.APPROVAL_REQUIRED: "gate:approval",
        }.get(verdict.stop or StopReason.SAFETY_BLOCKED, "gate:safety")
        return replace(candidate, eligible=False, rejection=reason, score=0)
    return candidate


def _cost_units(cost: Mapping[str, int]) -> int:
    return cost.get("runtime_seconds", 0) // 30 + cost.get("runtime_executions", 0)


def _first_rejection(
    engine: DiscoveryEngine,
    capability: str,
    state: ResearchState,
    scheduler: DiscoveryScheduler,
    built: AnalysisRequest,
    identity: CampaignIdentity,
    key: str,
    cost: dict[str, int],
) -> str:
    if (
        built.language
        and engine.supported_languages
        and built.language not in engine.supported_languages
    ):
        return "environment:language"
    if engine.availability() is not EngineAvailability.AVAILABLE:
        return "environment:engine_unavailable"
    chosen = capability_for(engine, built)
    if chosen.value != capability:
        return "prerequisite:capability_not_selectable"
    missing = prerequisite(capability, built, identity)
    if missing:
        return missing
    decision = scheduler.decide(engine.engine_id, built)
    if decision.action != "run":
        return f"scheduler:{decision.code or 'skip'}"
    health = next((h for h in state.health if h.engine == engine.engine_id), None)
    if health and health.consecutive_failures >= CIRCUIT_FAILURES:
        return "environment:circuit_open"
    previous = [item for item in state.executions if item.execution_key == key]
    if any(item.phase is ExecutionPhase.COMPLETED for item in previous):
        return "duplicate:execution_key"
    if any(item.phase in ATTEMPTED for item in previous):
        return "environment:failed_attempt"
    if any(
        item.phase
        in {ExecutionPhase.UNAVAILABLE, ExecutionPhase.UNSUPPORTED, ExecutionPhase.BLOCKED}
        for item in previous
    ):
        return "duplicate:refused_before"
    shortfall = state.budget.shortfall(cost)
    if shortfall:
        return f"budget:{shortfall}"
    return ""


def _finalize(
    needs: tuple[CapabilityNeed, ...],
    wanted: list[CapabilityNeed],
    considered: list[Candidate],
    supported: set[str],
) -> tuple[CapabilityNeed, ...]:
    """Name why an actionable need cannot run. A need that can run is unchanged."""
    by_capability: dict[str, list[Candidate]] = {}
    for item in considered:
        by_capability.setdefault(item.capability, []).append(item)
    merged = {need.capability: need for need in needs}
    for need in wanted:
        merged.setdefault(need.capability, need)
    result: list[CapabilityNeed] = []
    for need in merged.values():
        if need.status not in ACTIONABLE:
            result.append(need)
            continue
        group = by_capability.get(need.capability, [])
        if any(item.eligible for item in group):
            result.append(need)
        elif need.capability not in supported or not group:
            result.append(replace(need, status=NeedStatus.UNAVAILABLE))
        elif all(item.rejection.startswith("duplicate:") for item in group):
            result.append(replace(need, status=NeedStatus.REDUNDANT))
        elif all(item.rejection.startswith("environment:") for item in group):
            result.append(replace(need, status=NeedStatus.UNAVAILABLE))
        else:
            result.append(replace(need, status=NeedStatus.BLOCKED))
    return tuple(sorted(result, key=lambda n: (-n.weight, n.capability)))


def stop_reason(
    considered: list[Candidate], needs: tuple[CapabilityNeed, ...]
) -> tuple[StopReason, str]:
    blocked = [n for n in needs if n.status in {NeedStatus.BLOCKED, NeedStatus.UNAVAILABLE}]
    reasons = [c.rejection for c in considered if not c.eligible]
    if blocked:
        names = {n.capability for n in blocked}
        relevant = [c.rejection for c in considered if c.capability in names and not c.eligible]
        for prefix, stop in _BLOCK_PRECEDENCE:
            if any(item.startswith(prefix) for item in relevant):
                return stop, f"blocked: {sorted(set(relevant))[:6]}"
        return StopReason.ENVIRONMENT_UNAVAILABLE, "no engine supports a needed capability"
    if any(n.status is NeedStatus.REDUNDANT for n in needs) or any(
        item.startswith("duplicate:") for item in reasons
    ):
        return StopReason.NO_USEFUL_CAPABILITY, "remaining capabilities would repeat evidence"
    return StopReason.ALL_CAPABILITIES_EXERCISED, "no capability need remains"


def _rationale(best: Candidate, wanted: list[CapabilityNeed]) -> str:
    need = next((n for n in wanted if n.capability == best.capability), None)
    reasons = "; ".join(need.reasons) if need else ""
    return (
        f"{best.capability} via {best.engine} ({best.complementarity.value}, "
        f"score {best.score}): {reasons}"
    )[:400]
