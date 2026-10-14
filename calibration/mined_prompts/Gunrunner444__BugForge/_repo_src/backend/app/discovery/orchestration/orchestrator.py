"""Adaptive, bounded research orchestration.

One engine runs per round. Each round assesses what evidence is missing, ranks
the eligible engines deterministically, reserves budget, runs exactly one
engine through the scheduler, and folds the result back into typed state. The
orchestrator calls no model, raises no limit, grants no approval, and never
marks a finding verified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from app.discovery.engine import AnalysisRequest
from app.discovery.orchestration.assess import (
    AssessContext,
    actionable,
    assess,
    uncertainties_of,
)
from app.discovery.orchestration.budget import actual_cost, new_ledger
from app.discovery.orchestration.codec import digest
from app.discovery.orchestration.evidence import (
    RUNTIME_FAMILY,
    compute_delta,
    corroborate,
    detect_contradictions,
    identity_matches,
    items_from_result,
    negative_from,
    outcome_of,
)
from app.discovery.orchestration.gate import DefaultGate, ExecutionGate
from app.discovery.orchestration.model import (
    OPERATOR_STOPS,
    RESUMABLE_STOPS,
    STOP_STATE,
    TERMINAL,
    BudgetLedger,
    CampaignIdentity,
    Candidate,
    CapabilityNeed,
    DecisionRecord,
    DecisionSource,
    EngineHealth,
    EvidenceDelta,
    EvidenceItem,
    EvidenceQuality,
    ExecutionPhase,
    ExecutionRecord,
    FailureClass,
    IllegalTransitionError,
    InvalidDecisionError,
    OrchestrationError,
    OrchestratorState,
    PersistenceError,
    ResearchState,
    ResumeError,
    ResumeStatus,
    StopReason,
    transition,
)
from app.discovery.orchestration.planner import Plan, Suggestion, plan
from app.discovery.orchestration.report import build_report
from app.discovery.orchestration.store import ResearchStore
from app.discovery.results import DynamicResult
from app.discovery.scheduler import DiscoveryScheduler
from app.discovery.sequences import MAX_EXECUTIONS

MAX_SUPERSEDED = 2
MAX_PENDING_SUGGESTIONS = 8
_RETRYABLE_PHASES = frozenset(
    {ExecutionPhase.TIMED_OUT, ExecutionPhase.FAILED, ExecutionPhase.UNKNOWN}
)
_RETRYABLE_FAILURES = frozenset(
    {FailureClass.TIMEOUT, FailureClass.INFRASTRUCTURE, FailureClass.UNKNOWN_OUTCOME}
)
_S = OrchestratorState


def identity_from_request(request: AnalysisRequest, *, campaign_id: str = "") -> CampaignIdentity:
    """Derive the campaign identity from caller-supplied fields only."""
    extra = request.extra
    base = CampaignIdentity(
        campaign_id="",
        project=extra.get("project_id", "") or str(request.repo_root),
        source_snapshot=extra.get("source_snapshot", ""),
        compiler_configuration=extra.get("compiler_configuration", ""),
        repository_root=str(request.repo_root),
        language=request.language,
        target=request.target,
        contract=request.contract,
        function=request.function,
        source_file=request.source_file,
        deployment=extra.get("deployment_address", ""),
        fork_reference=extra.get("fork_reference", ""),
        program_context=extra.get("program_context", ""),
    )
    chosen = campaign_id or request.campaign_id or f"cp_{base.target_digest()}"
    return replace(base, campaign_id=chosen)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Orchestrator:
    def __init__(
        self,
        scheduler: DiscoveryScheduler,
        request: AnalysisRequest,
        *,
        identity: CampaignIdentity | None = None,
        store: ResearchStore | None = None,
        gate: ExecutionGate | None = None,
        max_rounds: int | None = None,
        caps: dict[str, int] | None = None,
        runtime_seconds: int | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.request = request
        self.identity = identity or identity_from_request(request)
        self.store = store
        self.gate: ExecutionGate = gate or DefaultGate()
        self._clock = clock or _now
        self._budget_args = (max_rounds, caps, runtime_seconds)
        self._suggestions: list[Suggestion] = []
        self._started = False
        self.state = ResearchState(identity=self.identity, budget=self._new_ledger())

    # ---- public API ------------------------------------------------------------------------

    def suggest(self, capability: str, *, engine: str = "", reason: str = "") -> bool:
        """Queue a request from Cursor. It is validated like any other candidate."""
        if len(self._suggestions) >= MAX_PENDING_SUGGESTIONS:
            return False
        self._suggestions.append(Suggestion(capability, engine, reason[:120]))
        return True

    def start(self, *, explicit_resume: bool = False) -> ResearchState:
        saved = self.store.load(self.identity.campaign_id) if self.store else None
        if saved is None:
            self.state = ResearchState(identity=self.identity, budget=self._new_ledger())
            self._save()
        else:
            self.state = self._restore(saved, explicit_resume=explicit_resume)
            self._save()
        self._started = True
        return self.state

    def run(self) -> ResearchState:
        if not self._started:
            self.start()
        while self.step():
            pass
        return self.state

    def step(self) -> bool:
        """Advance by at most one engine run. False means nothing more will happen now."""
        if not self._started:
            self.start()
        if self.state.state in TERMINAL or self.state.state is _S.STOPPED:
            return False
        try:
            return self._step()
        except (ResumeError, PersistenceError):
            raise
        except OrchestrationError as exc:
            self._fail(exc)
            return False

    def grant_retry(self, execution_key: str) -> bool:
        """Allow one explicit re-run of an unknown or infrastructure-failed execution.

        The retry still needs budget, a gate pass, and a free execution key.
        """
        state = self.state
        if state.state in TERMINAL:
            return False
        record = next((r for r in state.executions if r.execution_key == execution_key), None)
        if (
            record is None
            or record.phase not in _RETRYABLE_PHASES
            or not record.retryable
            or record.failure_class not in _RETRYABLE_FAILURES
        ):
            return False
        used = sum(
            1
            for item in state.executions
            if item.phase is ExecutionPhase.SUPERSEDED
            and (item.engine, item.capability, item.input_digest)
            == (record.engine, record.capability, record.input_digest)
        )
        if used >= MAX_SUPERSEDED:
            return False
        state.executions = tuple(
            replace(item, phase=ExecutionPhase.SUPERSEDED, retryable=False)
            if item.execution_key == execution_key
            else item
            for item in state.executions
        )
        if state.state is _S.STOPPED and state.stop_reason in {
            reason.value for reason in RESUMABLE_STOPS
        }:
            self._move(_S.PLANNING, explicit_resume=True)
            state.stop_reason = ""
        self._save()
        return True

    def report(self) -> dict[str, object]:
        return dict(build_report(self.state))

    def halt(self, reason: StopReason, *, detail: str = "") -> bool:
        """Operator pause or stop through the normal state machine.

        A pause (``OPERATOR_PAUSED``) is a resumable stop; a stop
        (``OPERATOR_STOPPED``) is final. The call is only valid between rounds,
        which callers guarantee by holding the campaign lock that ``step`` runs
        under. The decision is recorded like any other stop and persisted. A
        terminal campaign, or one already stopped for a non-operator reason, is
        left unchanged and ``False`` is returned.
        """
        if reason not in OPERATOR_STOPS:
            raise InvalidDecisionError(f"not an operator stop: {reason.value}")
        if not self._started:
            self.start()
        state = self.state
        if state.state in TERMINAL:
            return False
        if state.state is _S.STOPPED:
            current = state.stop_reason
            if current == StopReason.OPERATOR_STOPPED.value:
                return False
            if current == reason.value:
                return False
            resumable = {item.value for item in RESUMABLE_STOPS}
            if current not in resumable:
                return False
            # Re-enter planning (an operator action) only to record the new stop.
            self._move(_S.PLANNING, explicit_resume=True)
            state.stop_reason = ""
        elif state.state is _S.CREATED:
            self._move(_S.BASELINE)
            self._move(_S.PLANNING)
        elif state.state not in {_S.PLANNING, _S.CONTINUE, _S.BLOCKED}:
            raise IllegalTransitionError(f"cannot halt from {state.state.value}")
        self._suggestions.clear()
        needs = assess(state, self._context())
        self._stop(reason, needs, None, detail[:400] or f"{reason.value} by operator")
        return True

    # ---- stepping --------------------------------------------------------------------------

    def _step(self) -> bool:
        state = self.state
        if state.state is _S.CREATED:
            self._move(_S.BASELINE)
            self._move(_S.PLANNING)
        elif state.state in {_S.CONTINUE, _S.BLOCKED}:
            self._move(_S.PLANNING)
        elif state.state is not _S.PLANNING:
            raise IllegalTransitionError(f"cannot step from {state.state.value}")
        needs = assess(state, self._context())
        state.uncertainties = uncertainties_of(needs)
        suggestions = tuple(self._suggestions)
        self._suggestions.clear()
        if not actionable(needs) and not suggestions:
            self._stop(self._no_need_reason(), needs, None, "no capability need remains")
            return False
        exhausted = state.budget.exhausted()
        if exhausted:
            reason = (
                StopReason.ROUNDS_EXHAUSTED
                if exhausted == "rounds"
                else StopReason.EXECUTION_CAP_EXHAUSTED
            )
            self._stop(reason, needs, None, f"{exhausted} budget is spent")
            return False
        chosen = plan(
            state=state,
            needs=needs,
            scheduler=self.scheduler,
            request=self.request,
            identity=self.identity,
            gate=self.gate,
            suggestions=suggestions,
        )
        if chosen.selected is None or chosen.request is None:
            self._stop(chosen.stop or StopReason.NO_USEFUL_CAPABILITY, chosen.needs, chosen, "")
            return False
        return self._run_round(chosen, needs)

    def _no_need_reason(self) -> StopReason:
        state = self.state
        if state.contradictions:
            return StopReason.UNRESOLVED_UNCERTAINTY
        if state.executions and state.budget.remaining("rounds") <= 0:
            return StopReason.COMPLETED_BOUNDED_RESEARCH
        return StopReason.ALL_CAPABILITIES_EXERCISED

    def _context(self) -> AssessContext:
        request = self.request
        return AssessContext(
            language=request.language,
            extra=dict(request.extra),
            difficult=request.difficult,
            has_harness=request.has_harness,
            framework=request.framework,
            targeted=bool(request.function or request.contract),
        )

    # ---- one round -------------------------------------------------------------------------

    def _run_round(self, chosen: Plan, needs_before: tuple[CapabilityNeed, ...]) -> bool:
        state = self.state
        candidate = chosen.selected
        request = chosen.request
        assert candidate is not None and request is not None
        hash_before = state.hash_now()
        sequence = state.decision_number + 1
        decision_id = f"dc_{digest((self.identity.campaign_id, sequence))}"
        budget_before = _remaining(state.budget)
        evidence_before = state.evidence
        contradictions_before = state.contradictions
        refs_before = set(state.corpus_refs)
        self._move(_S.READY)
        try:
            state.budget.reserve(candidate.estimated_cost)
        except InvalidDecisionError:
            self._move(_S.PLANNING)
            self._stop(StopReason.BUDGET_EXHAUSTED, needs_before, chosen, "reservation refused")
            return False
        self._move(_S.EXECUTING)
        state.round += 1
        state.executions = (
            *state.executions,
            ExecutionRecord(
                execution_key=candidate.execution_key,
                decision_id=decision_id,
                engine=candidate.engine,
                capability=candidate.capability,
                input_digest=candidate.input_digest,
                digest_after="",
                phase=ExecutionPhase.STARTED,
                status="started",
                executed=False,
                failure_class=FailureClass.NONE,
                consumed=dict(candidate.estimated_cost),
                round=state.round,
            ),
        )
        self._save()
        result = self._call(candidate.engine, request)
        self._move(_S.OBSERVING)
        record, items, attributable = self._observe(candidate, request, result, decision_id)
        state.executions = tuple(
            item
            for item in state.executions
            if not (
                item.execution_key == candidate.execution_key
                and item.phase is ExecutionPhase.STARTED
            )
        ) + (record,)
        state.budget.settle(candidate.estimated_cost, record.consumed)
        state.health = _health(state.health, candidate.engine, record.phase)
        merged = {item.evidence_id: item for item in (*state.evidence, *items)}
        state.evidence = corroborate(tuple(sorted(merged.values(), key=lambda i: i.evidence_id)))
        state.contradictions = detect_contradictions(
            state.evidence, frozenset(state.exercised_capabilities)
        )
        negatives_added: tuple[str, ...] = ()
        if result is not None and attributable:
            negative = negative_from(
                result,
                identity=self.identity,
                capability=candidate.capability,
                phase=record.phase,
            )
            if negative is not None and negative not in state.negatives:
                state.negatives = (*state.negatives, negative)
                negatives_added = (f"{negative.engine}:{negative.capability}",)
        state.corpus_refs = self._corpus_refs()
        needs_after = assess(state, self._context())
        state.uncertainties = uncertainties_of(needs_after)
        delta = compute_delta(
            evidence_before,
            state.evidence,
            new_seeds=tuple(sorted(set(state.corpus_refs) - refs_before)),
            new_coverage=bool(
                result is not None
                and attributable
                and result.coverage.get("new_coverage", result.coverage.get("new", "")) == "true"
            ),
            contradictions_before=contradictions_before,
            contradictions_after=state.contradictions,
            negatives_added=negatives_added,
            uncertainty_before=uncertainties_of(needs_before),
            uncertainty_after=state.uncertainties,
        )
        self._move(_S.CORRELATING)
        state.recommend_verification_review = _recommend(state)
        state.decision_number = sequence
        state.decisions = (
            *state.decisions,
            self._decision(
                sequence=sequence,
                decision_id=decision_id,
                hash_before=hash_before,
                budget_before=budget_before,
                chosen=chosen,
                needs=chosen.needs,
                record=record,
                delta=delta,
                result=result,
                request=request,
                next_capability=_first_need(needs_after),
                previous_state=_S.PLANNING.value,
            ),
        )
        state.next_capability = _first_need(needs_after)
        if self._redundant():
            self._stop(
                StopReason.REDUNDANT_EVIDENCE, needs_after, None, "two repeats added nothing"
            )
            return False
        self._move(_S.CONTINUE)
        self._save()
        return True

    def _call(self, engine_id: str, request: AnalysisRequest) -> DynamicResult | None:
        try:
            result = self.scheduler.run_engine(engine_id, request)
        except Exception:
            return None
        return result if isinstance(result, DynamicResult) else None

    def _observe(
        self,
        candidate: Candidate,
        request: AnalysisRequest,
        result: DynamicResult | None,
        decision_id: str,
    ) -> tuple[ExecutionRecord, tuple[EvidenceItem, ...], bool]:
        base = {
            "execution_key": candidate.execution_key,
            "decision_id": decision_id,
            "engine": candidate.engine,
            "capability": candidate.capability,
            "input_digest": candidate.input_digest,
            "round": self.state.round,
            "retries": 0,
        }
        if result is None:
            return (
                ExecutionRecord(
                    **base,  # type: ignore[arg-type]
                    digest_after="",
                    phase=ExecutionPhase.UNKNOWN,
                    status="unknown",
                    executed=False,
                    failure_class=FailureClass.UNKNOWN_OUTCOME,
                    consumed=dict(candidate.estimated_cost),
                    retryable=True,
                ),
                (),
                False,
            )
        attributable = (
            result.engine == candidate.engine
            and result.language in {"", request.language}
            and (
                not result.campaign_id
                or not request.campaign_id
                or result.campaign_id == request.campaign_id
            )
            and identity_matches(result, self.identity)
        )
        if not attributable:
            phase, failure = ExecutionPhase.FAILED, FailureClass.IDENTITY
        else:
            phase, failure = outcome_of(result)
        items: tuple[EvidenceItem, ...] = ()
        if attributable:
            items = items_from_result(
                result,
                identity=self.identity,
                capability=candidate.capability,
                execution_key=candidate.execution_key,
                phase=phase,
            )
        runs = 1
        if candidate.capability in RUNTIME_FAMILY:
            try:
                runs = max(1, min(int(result.metadata.get("executions", "1")), MAX_EXECUTIONS))
            except ValueError:
                runs = 1
        consumed = actual_cost(candidate.capability, executed=result.executed, runs=runs)
        record = ExecutionRecord(
            **base,  # type: ignore[arg-type]
            digest_after=digest(sorted(item.evidence_id for item in items)) if items else "",
            phase=phase,
            status=result.status.value,
            executed=result.executed,
            failure_class=failure,
            consumed=consumed,
            evidence_ids=tuple(sorted(item.evidence_id for item in items)),
            retryable=failure in _RETRYABLE_FAILURES and phase in _RETRYABLE_PHASES,
        )
        return record, items, attributable

    def _redundant(self) -> bool:
        recent = [
            r
            for r in self.state.decisions
            if r.phase is ExecutionPhase.COMPLETED and r.selected_capability
        ]
        if len(recent) < 2:
            return False
        last, previous = recent[-1], recent[-2]
        return (
            last.selected_capability == previous.selected_capability
            and not last.evidence_delta.informative
            and not previous.evidence_delta.informative
        )

    def _corpus_refs(self) -> tuple[str, ...]:
        identity = self.identity
        refs = {
            f"{seed.source.value}:{seed.content_sha256[:16]}"
            for seed in self.scheduler.corpus.seeds
            if seed.source_snapshot == identity.source_snapshot
            and seed.compiler_configuration == identity.compiler_configuration
            and seed.target == self.request.target
        }
        return tuple(sorted(refs))

    # ---- decisions and stopping ------------------------------------------------------------

    def _decision(
        self,
        *,
        sequence: int,
        decision_id: str,
        hash_before: str,
        budget_before: dict[str, int],
        chosen: Plan,
        needs: tuple[CapabilityNeed, ...],
        record: ExecutionRecord,
        delta: EvidenceDelta,
        result: DynamicResult | None,
        request: AnalysisRequest,
        next_capability: str,
        previous_state: str,
        stop_reason: str = "",
        source: DecisionSource = DecisionSource.PLANNER,
    ) -> DecisionRecord:
        state = self.state
        candidate = chosen.selected
        rationale = chosen.rationale
        if chosen.rejected_suggestions:
            rationale = (
                f"{rationale}; rejected suggestions: {', '.join(chosen.rejected_suggestions)}"[:400]
            )
        extra = request.extra
        fork: dict[str, str] = {}
        differential: dict[str, str] = {}
        if record.capability in {"fork_validation", "differential_validation"}:
            pinned = bool(
                extra.get("chain_id") and extra.get("fork_block") and extra.get("state_snapshot")
            )
            fork = {
                "chain_id": extra.get("chain_id", ""),
                "block": extra.get("fork_block", ""),
                "state_snapshot": extra.get("state_snapshot", ""),
                "pinned": str(pinned).lower(),
            }
        if record.capability == "differential_validation":
            differential = {
                "state_reset": extra.get("state_reset", ""),
                "classification": str(result.metadata.get("classification", "")) if result else "",
            }
        return DecisionRecord(
            decision_id=decision_id,
            campaign_id=self.identity.campaign_id,
            sequence=sequence,
            round=record.round,
            source=source,
            phase=record.phase,
            previous_state=previous_state,
            state_hash_before=hash_before,
            evidence_summary=_summary(state),
            uncertainties=state.uncertainties,
            missing_capability=_first_need(needs),
            needs=needs,
            considered=chosen.considered,
            selected_capability=candidate.capability if candidate else "",
            selected_engine=candidate.engine if candidate else "",
            rationale=rationale,
            prerequisites=chosen.prerequisites,
            budget_before=budget_before,
            reserved_cost=dict(candidate.estimated_cost) if candidate else {},
            consumed_cost=dict(record.consumed),
            execution_id=record.execution_key,
            input_digest=record.input_digest,
            result_status=record.status,
            evidence_delta=delta,
            next_capability=next_capability,
            stop_reason=stop_reason,
            fork_reference=fork,
            differential=differential,
            identity_digest=self.identity.target_digest(),
            source_snapshot=self.identity.source_snapshot,
            compiler_configuration=self.identity.compiler_configuration,
            orchestrator_version=state.orchestrator_version,
            created_at=self._clock(),
        ).sealed()

    def _stop(
        self,
        reason: StopReason,
        needs: tuple[CapabilityNeed, ...],
        chosen: Plan | None,
        detail: str,
    ) -> None:
        state = self.state
        hash_before = state.hash_now()
        previous = state.state
        if previous is _S.PLANNING and reason in RESUMABLE_STOPS:
            self._move(_S.BLOCKED)
        self._move(_S.STOPPING)
        state.stop_reason = reason.value
        state.rationale = (detail or (chosen.rationale if chosen else ""))[:400]
        state.next_capability = _first_need(needs)
        sequence = state.decision_number + 1
        plan_view = chosen or Plan(selected=None, considered=(), needs=needs)
        blocked = STOP_STATE[reason] in {_S.STOPPED, _S.INCONCLUSIVE}
        record = ExecutionRecord(
            execution_key="",
            decision_id=f"dc_{digest((self.identity.campaign_id, sequence))}",
            engine="",
            capability="",
            input_digest="",
            digest_after="",
            phase=ExecutionPhase.BLOCKED if blocked else ExecutionPhase.COMPLETED,
            status="stopped",
            executed=False,
            failure_class=FailureClass.POLICY if blocked else FailureClass.NONE,
            round=state.round,
        )
        state.decision_number = sequence
        state.decisions = (
            *state.decisions,
            self._decision(
                sequence=sequence,
                decision_id=record.decision_id,
                hash_before=hash_before,
                budget_before=_remaining(state.budget),
                chosen=plan_view,
                needs=needs,
                record=record,
                delta=EvidenceDelta(),
                result=None,
                request=self.request,
                next_capability=state.next_capability,
                previous_state=previous.value,
                stop_reason=reason.value,
                source=DecisionSource.OPERATOR
                if reason in OPERATOR_STOPS
                else DecisionSource.PLANNER,
            ),
        )
        state.recommend_verification_review = _recommend(state)
        self._move(STOP_STATE[reason])
        self._save()

    def _fail(self, error: OrchestrationError) -> None:
        state = self.state
        state.rationale = f"{type(error).__name__}: {error}"[:400]
        if state.state not in TERMINAL:
            try:
                state.state = transition(state.state, _S.FAILED)
            except IllegalTransitionError:
                state.state = _S.FAILED
        self._save()

    def _move(self, target: OrchestratorState, *, explicit_resume: bool = False) -> None:
        self.state.state = transition(self.state.state, target, explicit_resume=explicit_resume)

    def _save(self) -> None:
        if self.store is not None:
            self.store.save(self.state)

    def _new_ledger(self) -> BudgetLedger:
        max_rounds, caps, runtime_seconds = self._budget_args
        ledger = new_ledger(
            self.scheduler, max_rounds=max_rounds, caps=caps, runtime_seconds=runtime_seconds
        )
        started = self.scheduler.engines_started
        for name in ("engines", "executions"):
            ledger.consumed[name] = max(ledger.consumed.get(name, 0), started)
        return ledger

    # ---- resume ----------------------------------------------------------------------------

    def _restore(self, saved: ResearchState, *, explicit_resume: bool) -> ResearchState:
        """Revalidate a stored campaign. Nothing restored can widen authority or budget."""
        if (
            saved.identity.campaign_id != self.identity.campaign_id
            or saved.identity.target_digest() != self.identity.target_digest()
        ):
            raise ResumeError(ResumeStatus.STALE_IDENTITY, "target, snapshot, or compiler changed")
        state = saved
        ledger = self._new_ledger()
        ledger.merge_restored(saved.budget)
        recovered: list[ExecutionRecord] = []
        kept: list[ExecutionRecord] = []
        for item in state.executions:
            if item.phase in {ExecutionPhase.STARTED, ExecutionPhase.PLANNED}:
                unknown = replace(
                    item,
                    phase=ExecutionPhase.UNKNOWN,
                    status="unknown",
                    failure_class=FailureClass.UNKNOWN_OUTCOME,
                    retryable=True,
                    executed=False,
                )
                recovered.append(unknown)
                kept.append(unknown)
                for name, amount in item.consumed.items():
                    ledger.consumed[name] = ledger.consumed.get(name, 0) + amount
                    if ledger.consumed[name] > ledger.limits.get(name, 0):
                        ledger.overrun = True
            else:
                kept.append(item)
        state.executions = tuple(kept)
        state.budget = ledger
        state.resume_status = ResumeStatus.RESTORED
        state.resume_count += 1
        state.next_engine = ""
        state.feedback = {}
        self.state = state
        for item in recovered:
            self._record_recovery(item)
        self._normalize()
        if state.state is _S.STOPPED:
            self._maybe_resume(explicit_resume)
        state.recommend_verification_review = _recommend(state)
        return state

    def _record_recovery(self, record: ExecutionRecord) -> None:
        state = self.state
        sequence = state.decision_number + 1
        state.decision_number = sequence
        state.decisions = (
            *state.decisions,
            self._decision(
                sequence=sequence,
                decision_id=f"dc_{digest((self.identity.campaign_id, sequence))}",
                hash_before=state.hash_now(),
                budget_before=_remaining(state.budget),
                chosen=Plan(selected=None, considered=(), needs=()),
                needs=(),
                record=record,
                delta=EvidenceDelta(),
                result=None,
                request=self.request,
                next_capability="",
                previous_state=state.state.value,
                source=DecisionSource.RECOVERY,
            ),
        )

    def _normalize(self) -> None:
        """Bring a state that was interrupted mid-round back to a point that can plan."""
        state = self.state
        walk = {
            _S.EXECUTING: (_S.OBSERVING, _S.CORRELATING, _S.CONTINUE),
            _S.OBSERVING: (_S.CORRELATING, _S.CONTINUE),
            _S.CORRELATING: (_S.CONTINUE,),
            _S.READY: (_S.PLANNING,),
            _S.BASELINE: (_S.PLANNING,),
        }
        for target in walk.get(state.state, ()):
            self._move(target)
        if state.state is _S.STOPPING:
            try:
                reason = StopReason(state.stop_reason)
            except ValueError:
                self._move(_S.INCONCLUSIVE)
            else:
                self._move(STOP_STATE[reason])

    def _maybe_resume(self, explicit_resume: bool) -> None:
        state = self.state
        try:
            reason = StopReason(state.stop_reason)
        except ValueError:
            return
        if not explicit_resume or reason not in RESUMABLE_STOPS:
            return
        available = {
            engine.engine_id
            for engine in self.scheduler.engines
            if engine.availability().value == "available"
        }
        state.executions = tuple(
            replace(item, phase=ExecutionPhase.SUPERSEDED)
            if item.phase is ExecutionPhase.UNAVAILABLE and item.engine in available
            else item
            for item in state.executions
        )
        self._move(_S.PLANNING, explicit_resume=True)
        state.stop_reason = ""


def _remaining(ledger: BudgetLedger) -> dict[str, int]:
    return {name: ledger.remaining(name) for name in sorted(ledger.limits)}


def _first_need(needs: tuple[CapabilityNeed, ...]) -> str:
    live = actionable(needs)
    return live[0].capability if live else ""


def _summary(state: ResearchState) -> dict[str, int]:
    counts: dict[str, int] = {"total": len(state.evidence)}
    for item in state.evidence:
        counts[item.quality.value] = counts.get(item.quality.value, 0) + 1
    counts["contradictions"] = len(state.contradictions)
    counts["negatives"] = len(state.negatives)
    return dict(sorted(counts.items()))


def _recommend(state: ResearchState) -> bool:
    """A recommendation for Cursor to review. It changes no finding and verifies nothing."""
    conflicted = {item.identity_key for item in state.contradictions if item.status != "resolved"}
    return any(
        item.quality is EvidenceQuality.CORROBORATED and item.identity_key not in conflicted
        for item in state.evidence
    )


def _health(
    current: tuple[EngineHealth, ...], engine: str, phase: ExecutionPhase
) -> tuple[EngineHealth, ...]:
    existing = next((item for item in current if item.engine == engine), EngineHealth(engine))
    if phase is ExecutionPhase.COMPLETED:
        updated = replace(existing, successes=existing.successes + 1, consecutive_failures=0)
    elif phase in {ExecutionPhase.FAILED, ExecutionPhase.UNKNOWN}:
        updated = replace(existing, consecutive_failures=existing.consecutive_failures + 1)
    elif phase is ExecutionPhase.TIMED_OUT:
        updated = replace(
            existing,
            consecutive_failures=existing.consecutive_failures + 1,
            timeouts=existing.timeouts + 1,
        )
    elif phase is ExecutionPhase.UNAVAILABLE:
        updated = replace(existing, unavailable=existing.unavailable + 1)
    else:
        updated = existing
    rest = tuple(item for item in current if item.engine != engine)
    return tuple(sorted((*rest, updated), key=lambda item: item.engine))


__all__ = ["Orchestrator", "identity_from_request"]
