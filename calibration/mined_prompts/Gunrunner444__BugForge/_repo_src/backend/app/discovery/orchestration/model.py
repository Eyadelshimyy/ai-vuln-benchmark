"""Typed state, taxonomy, and state machine for Phase 49 orchestration.

Nothing here executes an engine and nothing here can mark a finding verified.
There is no "verified" evidence quality on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from app.discovery.orchestration.codec import digest, to_jsonable

SCHEMA_VERSION = 1
ORCHESTRATOR_VERSION = "phase49.1"


class OrchestrationError(Exception):
    """Base class. Each subclass is a distinct failure kind, not a generic failure."""


class InvalidDecisionError(OrchestrationError):
    """A request or suggestion is malformed or outside what the campaign allows."""


class IllegalTransitionError(OrchestrationError):
    """The state machine does not allow this move."""


class PersistenceError(OrchestrationError):
    """State could not be saved or loaded."""


class ResumeError(OrchestrationError):
    """A persisted campaign could not be restored safely. The status says why."""

    def __init__(self, status: ResumeStatus, detail: str = "") -> None:
        super().__init__(f"{status.value}: {detail}" if detail else status.value)
        self.status = status
        self.detail = detail


class OrchestratorState(StrEnum):
    CREATED = "created"
    BASELINE = "baseline"
    PLANNING = "planning"
    READY = "ready"
    BLOCKED = "blocked"
    EXECUTING = "executing"
    OBSERVING = "observing"
    CORRELATING = "correlating"
    CONTINUE = "continue"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    INCONCLUSIVE = "inconclusive"
    FAILED = "failed"


_S = OrchestratorState
TRANSITIONS: dict[OrchestratorState, frozenset[OrchestratorState]] = {
    _S.CREATED: frozenset({_S.BASELINE, _S.FAILED}),
    _S.BASELINE: frozenset({_S.PLANNING, _S.FAILED}),
    _S.PLANNING: frozenset({_S.READY, _S.BLOCKED, _S.STOPPING, _S.FAILED}),
    _S.READY: frozenset({_S.EXECUTING, _S.BLOCKED, _S.PLANNING, _S.FAILED}),
    _S.BLOCKED: frozenset({_S.PLANNING, _S.STOPPING, _S.FAILED}),
    _S.EXECUTING: frozenset({_S.OBSERVING, _S.FAILED}),
    _S.OBSERVING: frozenset({_S.CORRELATING, _S.FAILED}),
    _S.CORRELATING: frozenset({_S.CONTINUE, _S.STOPPING, _S.FAILED}),
    _S.CONTINUE: frozenset({_S.PLANNING, _S.STOPPING, _S.FAILED}),
    _S.STOPPING: frozenset({_S.STOPPED, _S.COMPLETED, _S.INCONCLUSIVE, _S.FAILED}),
    _S.STOPPED: frozenset({_S.PLANNING}),
    _S.COMPLETED: frozenset(),
    _S.INCONCLUSIVE: frozenset(),
    _S.FAILED: frozenset(),
}
TERMINAL = frozenset({_S.COMPLETED, _S.INCONCLUSIVE, _S.FAILED})


def transition(
    current: OrchestratorState, target: OrchestratorState, *, explicit_resume: bool = False
) -> OrchestratorState:
    """Return the new state or raise. STOPPED only moves on an explicit resume."""
    if target not in TRANSITIONS[current]:
        raise IllegalTransitionError(f"{current.value} -> {target.value}")
    if current is _S.STOPPED and not explicit_resume:
        raise IllegalTransitionError("stopped -> planning needs an explicit resume")
    return target


class StopReason(StrEnum):
    BUDGET_EXHAUSTED = "budget_exhausted"
    ROUNDS_EXHAUSTED = "rounds_exhausted"
    EXECUTION_CAP_EXHAUSTED = "execution_cap_exhausted"
    NO_USEFUL_CAPABILITY = "no_useful_capability"
    ALL_CAPABILITIES_EXERCISED = "all_capabilities_exercised"
    REDUNDANT_EVIDENCE = "redundant_evidence"
    PREREQUISITES_UNAVAILABLE = "prerequisites_unavailable"
    SCOPE_BLOCKED = "scope_blocked"
    SAFETY_BLOCKED = "safety_blocked"
    APPROVAL_REQUIRED = "approval_required"
    DETERMINISTIC_IDENTITY_MISSING = "deterministic_identity_missing"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
    UNRESOLVED_UNCERTAINTY = "unresolved_uncertainty"
    COMPLETED_BOUNDED_RESEARCH = "completed_bounded_research"
    # Operator control (Phase 51 hardening). A pause is resumable; a stop is final.
    OPERATOR_PAUSED = "operator_paused"
    OPERATOR_STOPPED = "operator_stopped"


# Stop reason -> terminal-ish machine state. Only these may be resumed explicitly.
STOP_STATE: dict[StopReason, OrchestratorState] = {
    StopReason.BUDGET_EXHAUSTED: _S.STOPPED,
    StopReason.ROUNDS_EXHAUSTED: _S.STOPPED,
    StopReason.EXECUTION_CAP_EXHAUSTED: _S.STOPPED,
    StopReason.NO_USEFUL_CAPABILITY: _S.COMPLETED,
    StopReason.ALL_CAPABILITIES_EXERCISED: _S.COMPLETED,
    StopReason.REDUNDANT_EVIDENCE: _S.COMPLETED,
    StopReason.PREREQUISITES_UNAVAILABLE: _S.STOPPED,
    StopReason.SCOPE_BLOCKED: _S.STOPPED,
    StopReason.SAFETY_BLOCKED: _S.STOPPED,
    StopReason.APPROVAL_REQUIRED: _S.STOPPED,
    StopReason.DETERMINISTIC_IDENTITY_MISSING: _S.INCONCLUSIVE,
    StopReason.ENVIRONMENT_UNAVAILABLE: _S.STOPPED,
    StopReason.UNRESOLVED_UNCERTAINTY: _S.INCONCLUSIVE,
    StopReason.COMPLETED_BOUNDED_RESEARCH: _S.COMPLETED,
    StopReason.OPERATOR_PAUSED: _S.STOPPED,
    StopReason.OPERATOR_STOPPED: _S.STOPPED,
}
RESUMABLE_STOPS = frozenset(
    {
        StopReason.APPROVAL_REQUIRED,
        StopReason.ENVIRONMENT_UNAVAILABLE,
        StopReason.PREREQUISITES_UNAVAILABLE,
        StopReason.OPERATOR_PAUSED,
    }
)
# Stops only an operator may request. Neither raises a budget or grants anything.
OPERATOR_STOPS = frozenset({StopReason.OPERATOR_PAUSED, StopReason.OPERATOR_STOPPED})


class NeedStatus(StrEnum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    REDUNDANT = "redundant"
    HIGH_VALUE_FOLLOWUP = "high_value_followup"
    LOW_VALUE_FOLLOWUP = "low_value_followup"


ACTIONABLE = frozenset(
    {
        NeedStatus.MISSING,
        NeedStatus.PARTIAL,
        NeedStatus.HIGH_VALUE_FOLLOWUP,
        NeedStatus.LOW_VALUE_FOLLOWUP,
    }
)


class EvidenceQuality(StrEnum):
    """What a result establishes. There is deliberately no verified level."""

    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"
    OBSERVATION = "observation"
    CANDIDATE = "candidate"
    CORROBORATED = "corroborated"


class ExecutionPhase(StrEnum):
    PLANNED = "planned"
    STARTED = "started"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    SUPERSEDED = "superseded"


ATTEMPTED = frozenset(
    {
        ExecutionPhase.STARTED,
        ExecutionPhase.COMPLETED,
        ExecutionPhase.TIMED_OUT,
        ExecutionPhase.FAILED,
        ExecutionPhase.UNKNOWN,
    }
)


class Complementarity(StrEnum):
    NOVEL = "novel"
    CORROBORATIVE = "corroborative"
    DISCRIMINATIVE = "discriminative"
    PREREQUISITE_GENERATING = "prerequisite_generating"
    COVERAGE_EXPANDING = "coverage_expanding"
    CONTRADICTION_RESOLVING = "contradiction_resolving"
    REDUNDANT = "redundant"


class FailureClass(StrEnum):
    NONE = "none"
    INFRASTRUCTURE = "infrastructure"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    IDENTITY = "identity"
    POLICY = "policy"
    TOOL = "tool"
    UNKNOWN_OUTCOME = "unknown_outcome"


class Uncertainty(StrEnum):
    BASELINE = "baseline"
    UNEXERCISED = "unexercised_candidate"
    REACHABILITY = "reachability"
    CROSS_CONTRACT = "cross_contract"
    ECONOMIC = "economic"
    ECONOMIC_RUNTIME = "economic_runtime"
    RUNTIME = "runtime"
    FORK = "fork"
    REPEATABILITY = "repeatability"
    PROPERTY = "property"
    SEED_FOLLOWUP = "seed_followup"
    DIVERGENCE = "divergence"
    CONTRADICTION = "contradiction"
    PROGRAM_CONTEXT = "program_context"
    SEMANTIC_RESEARCH = "semantic_research"
    COMPILER = "compiler"


class ResumeStatus(StrEnum):
    NEW = "new"
    RESTORED = "restored"
    STALE_IDENTITY = "stale_identity"
    CORRUPT = "corrupt"
    UNSUPPORTED_VERSION = "unsupported_version"
    MIGRATION_REQUIRED = "migration_required"


class DecisionSource(StrEnum):
    PLANNER = "planner"
    EXTERNAL = "external_suggestion"
    RETRY = "retry"
    RECOVERY = "recovery"
    OPERATOR = "operator"


# ---- identity -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignIdentity:
    campaign_id: str
    project: str = ""
    source_snapshot: str = ""
    compiler_configuration: str = ""
    repository_root: str = ""
    language: str = ""
    target: str = ""
    contract: str = ""
    function: str = ""
    source_file: str = ""
    deployment: str = ""
    fork_reference: str = ""
    program_context: str = ""

    def target_digest(self) -> str:
        """Identity of what is investigated, without the campaign id.

        An empty program context is left out so identities and persisted campaigns
        from before bounty programs existed keep their digests.
        """
        base = replace(self, campaign_id="")
        if base.program_context:
            return digest(base)
        data = to_jsonable(base)
        data.pop("program_context", None)
        return digest(data)

    def identity_key(self) -> str:
        """Contract and function together, or empty. A bare name is not an identity."""
        if self.contract and self.function:
            return f"{self.contract}.{self.function}"
        return ""


# ---- budget ---------------------------------------------------------------------------------

DIMENSIONS = (
    "attempts",
    "engines",
    "rounds",
    "executions",
    "runtime_executions",
    "runtime_seconds",
    "fuzz_runs",
    "symbolic_runs",
    "test_runs",
    "economic_runs",
    "protocol_runs",
)


@dataclass
class BudgetLedger:
    """Limits only tighten. Consumption only grows. A planner cannot edit either."""

    limits: dict[str, int] = field(default_factory=dict)
    consumed: dict[str, int] = field(default_factory=dict)
    reserved: dict[str, int] = field(default_factory=dict)
    overrun: bool = False

    def remaining(self, dimension: str) -> int:
        limit = self.limits.get(dimension)
        if limit is None:
            return 0
        return limit - self.consumed.get(dimension, 0) - self.reserved.get(dimension, 0)

    def shortfall(self, cost: dict[str, int]) -> str:
        """The first dimension (in fixed order) the cost cannot fit. Empty when it fits."""
        for dimension in DIMENSIONS:
            need = cost.get(dimension, 0)
            if need > 0 and need > self.remaining(dimension):
                return dimension
        for dimension in sorted(set(cost) - set(DIMENSIONS)):
            return dimension if cost[dimension] > 0 else ""
        return ""

    def reserve(self, cost: dict[str, int]) -> None:
        missing = self.shortfall(cost)
        if missing:
            raise InvalidDecisionError(f"budget:{missing}")
        for dimension, amount in cost.items():
            self.reserved[dimension] = self.reserved.get(dimension, 0) + amount

    def release(self, cost: dict[str, int]) -> None:
        for dimension, amount in cost.items():
            self.reserved[dimension] = max(0, self.reserved.get(dimension, 0) - amount)

    def settle(self, reserved: dict[str, int], actual: dict[str, int]) -> None:
        """Release the reservation and consume what was actually used.

        Work that never started consumes less than it reserved. Use above the
        reservation, or above a limit, is kept and flagged as an overrun.
        """
        self.release(reserved)
        for dimension in sorted(set(reserved) | set(actual)):
            amount = actual.get(dimension, 0)
            self.consumed[dimension] = self.consumed.get(dimension, 0) + amount
            if amount > reserved.get(dimension, 0) or self.consumed[dimension] > self.limits.get(
                dimension, 0
            ):
                self.overrun = True

    def exhausted(self) -> str:
        for dimension in ("rounds", "executions", "engines", "attempts"):
            if dimension in self.limits and self.remaining(dimension) <= 0:
                return dimension
        return ""

    def merge_restored(self, saved: BudgetLedger) -> None:
        """Restore without raising any limit or lowering any consumption."""
        for dimension, limit in saved.limits.items():
            current = self.limits.get(dimension)
            self.limits[dimension] = limit if current is None else min(limit, current)
        for dimension, amount in saved.consumed.items():
            self.consumed[dimension] = max(self.consumed.get(dimension, 0), amount)
        self.reserved = {}
        self.overrun = self.overrun or saved.overrun


# ---- evidence -------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    quality: EvidenceQuality
    engine: str
    capability: str
    identity_key: str = ""
    execution_key: str = ""
    polarity: str = ""
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Contradiction:
    contradiction_id: str
    kind: str
    identity_key: str
    evidence_ids: tuple[str, ...]
    engines: tuple[str, ...]
    discriminators: tuple[str, ...]
    status: str = "open"


@dataclass(frozen=True)
class NegativeEvidence:
    kind: str
    capability: str
    engine: str
    identity_key: str
    statement: str
    proves_safety: bool = False


@dataclass(frozen=True)
class EvidenceDelta:
    new_evidence_ids: tuple[str, ...] = ()
    new_finding_ids: tuple[str, ...] = ()
    new_protocol_paths: tuple[str, ...] = ()
    new_state_transitions: tuple[str, ...] = ()
    new_sequences: tuple[str, ...] = ()
    new_runtime_observations: tuple[str, ...] = ()
    new_economic_observations: tuple[str, ...] = ()
    new_differential_observations: tuple[str, ...] = ()
    new_corpus_seeds: tuple[str, ...] = ()
    new_contradictions: tuple[str, ...] = ()
    new_negative_evidence: tuple[str, ...] = ()
    new_coverage: bool = False
    uncertainty_added: tuple[str, ...] = ()
    uncertainty_removed: tuple[str, ...] = ()

    @property
    def informative(self) -> bool:
        """Evidence-based novelty. A run that only removes an uncertainty is not informative."""
        return bool(
            self.new_finding_ids
            or self.new_protocol_paths
            or self.new_state_transitions
            or self.new_sequences
            or self.new_runtime_observations
            or self.new_economic_observations
            or self.new_differential_observations
            or self.new_corpus_seeds
            or self.new_contradictions
            or self.new_coverage
        )


# ---- execution ledger -----------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionRecord:
    execution_key: str
    decision_id: str
    engine: str
    capability: str
    input_digest: str
    digest_after: str
    phase: ExecutionPhase
    status: str
    executed: bool
    failure_class: FailureClass
    consumed: dict[str, int] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    round: int = 0
    retries: int = 0
    retryable: bool = False


@dataclass(frozen=True)
class EngineHealth:
    engine: str
    consecutive_failures: int = 0
    timeouts: int = 0
    successes: int = 0
    unavailable: int = 0


# ---- planning -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityNeed:
    capability: str
    status: NeedStatus
    weight: int = 0
    reasons: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()


@dataclass(frozen=True)
class Candidate:
    engine: str
    capability: str
    eligible: bool
    rejection: str = ""
    complementarity: Complementarity = Complementarity.NOVEL
    score: int = 0
    components: dict[str, int] = field(default_factory=dict)
    estimated_cost: dict[str, int] = field(default_factory=dict)
    execution_key: str = ""
    input_digest: str = ""


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    campaign_id: str
    sequence: int
    round: int
    source: DecisionSource
    phase: ExecutionPhase
    previous_state: str
    state_hash_before: str
    evidence_summary: dict[str, int]
    uncertainties: tuple[str, ...]
    missing_capability: str
    needs: tuple[CapabilityNeed, ...]
    considered: tuple[Candidate, ...]
    selected_capability: str
    selected_engine: str
    rationale: str
    prerequisites: tuple[str, ...]
    budget_before: dict[str, int]
    reserved_cost: dict[str, int]
    consumed_cost: dict[str, int]
    execution_id: str
    input_digest: str
    result_status: str
    evidence_delta: EvidenceDelta
    next_capability: str
    stop_reason: str
    fork_reference: dict[str, str]
    differential: dict[str, str]
    identity_digest: str
    source_snapshot: str
    compiler_configuration: str
    orchestrator_version: str
    created_at: str = ""
    record_hash: str = ""

    def sealed(self) -> DecisionRecord:
        return replace(self, record_hash=digest(replace(self, record_hash="")))


@dataclass
class ResearchState:
    identity: CampaignIdentity
    schema_version: int = SCHEMA_VERSION
    orchestrator_version: str = ORCHESTRATOR_VERSION
    state: OrchestratorState = OrchestratorState.CREATED
    round: int = 0
    decision_number: int = 0
    resume_status: ResumeStatus = ResumeStatus.NEW
    resume_count: int = 0
    uncertainties: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    contradictions: tuple[Contradiction, ...] = ()
    negatives: tuple[NegativeEvidence, ...] = ()
    executions: tuple[ExecutionRecord, ...] = ()
    health: tuple[EngineHealth, ...] = ()
    budget: BudgetLedger = field(default_factory=BudgetLedger)
    corpus_refs: tuple[str, ...] = ()
    sequence_refs: tuple[str, ...] = ()
    protocol_refs: tuple[str, ...] = ()
    runtime_refs: tuple[str, ...] = ()
    economic_refs: tuple[str, ...] = ()
    differential_refs: tuple[str, ...] = ()
    decisions: tuple[DecisionRecord, ...] = ()
    next_capability: str = ""
    next_engine: str = ""
    rationale: str = ""
    stop_reason: str = ""
    recommend_verification_review: bool = False
    feedback: dict[str, str] = field(default_factory=dict)
    state_hash: str = ""

    def seal(self) -> ResearchState:
        self.state_hash = digest(replace(self, state_hash=""), length=32)
        return self

    def hash_now(self) -> str:
        return digest(replace(self, state_hash=""), length=32)

    @property
    def last_decision(self) -> DecisionRecord | None:
        return self.decisions[-1] if self.decisions else None

    def attempted_keys(self) -> frozenset[str]:
        return frozenset(item.execution_key for item in self.executions if item.phase in ATTEMPTED)

    def completed(self, capability: str) -> tuple[ExecutionRecord, ...]:
        return tuple(
            item
            for item in self.executions
            if item.capability == capability
            and item.phase is ExecutionPhase.COMPLETED
            and item.executed
        )

    @property
    def exercised_capabilities(self) -> tuple[str, ...]:
        return tuple(sorted({item.capability for item in self.executions if _exercised(item)}))

    @property
    def exercised_engines(self) -> tuple[str, ...]:
        return tuple(sorted({item.engine for item in self.executions if _exercised(item)}))

    @property
    def failed_engines(self) -> tuple[str, ...]:
        failed = (ExecutionPhase.FAILED, ExecutionPhase.TIMED_OUT, ExecutionPhase.UNKNOWN)
        return tuple(sorted({i.engine for i in self.executions if i.phase in failed}))

    @property
    def unavailable_engines(self) -> tuple[str, ...]:
        return tuple(
            sorted({i.engine for i in self.executions if i.phase is ExecutionPhase.UNAVAILABLE})
        )

    @property
    def skipped_engines(self) -> tuple[str, ...]:
        skipped = (ExecutionPhase.UNSUPPORTED, ExecutionPhase.BLOCKED)
        return tuple(sorted({i.engine for i in self.executions if i.phase in skipped}))

    @property
    def successful_engines(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    i.engine
                    for i in self.executions
                    if i.phase is ExecutionPhase.COMPLETED and i.executed
                }
            )
        )


def _exercised(record: ExecutionRecord) -> bool:
    """Only an execution that ran to a usable result counts. Planning, refusals, and failures do not."""
    return record.executed and record.phase is ExecutionPhase.COMPLETED
