from __future__ import annotations

from dataclasses import replace

import pytest

from app.discovery.capabilities import EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest
from app.discovery.orchestration import (
    MemoryStore,
    OrchestratorState,
    StopReason,
)
from app.discovery.orchestration import orchestrator as orchestrator_module
from app.discovery.orchestration.model import (
    EngineHealth,
    EvidenceDelta,
    ExecutionPhase,
    InvalidDecisionError,
    PersistenceError,
)
from app.discovery.orchestration.orchestrator import identity_from_request
from app.discovery.orchestration.planner import plan
from app.discovery.results import DynamicResult
from tests.test_phase49.phase49_support import (
    FakeEngine,
    fuzz_engine,
    make_orchestrator,
    make_request,
    make_scheduler,
    ok,
    static_clean,
    static_engine,
)

C = EngineCapability


def _bare(tmp_path) -> AnalysisRequest:
    return AnalysisRequest(
        repo_root=tmp_path,
        language="solidity",
        extra={"project_id": "p", "source_snapshot": "s", "compiler_configuration": "c"},
    )


def test_untargeted_research_ends_when_the_baseline_is_done(tmp_path) -> None:
    orch = make_orchestrator([static_engine(static_clean), fuzz_engine()], _bare(tmp_path))
    state = orch.run()
    assert state.stop_reason == StopReason.ALL_CAPABILITIES_EXERCISED.value
    assert state.state is OrchestratorState.COMPLETED
    assert [r.selected_capability for r in state.decisions if r.selected_capability] == [
        "static_analysis"
    ]


def test_a_spent_round_budget_with_nothing_left_is_bounded_completion(tmp_path) -> None:
    orch = make_orchestrator([static_engine(static_clean)], _bare(tmp_path), max_rounds=1)
    state = orch.run()
    assert state.stop_reason == StopReason.COMPLETED_BOUNDED_RESEARCH.value
    assert state.state is OrchestratorState.COMPLETED


def test_a_spent_round_budget_with_work_left_is_not_completion(tmp_path) -> None:
    orch = make_orchestrator(
        [static_engine(static_clean), fuzz_engine()], make_request(tmp_path), max_rounds=1
    )
    state = orch.run()
    assert state.stop_reason == StopReason.ROUNDS_EXHAUSTED.value
    assert state.state is OrchestratorState.STOPPED


def test_the_shared_engine_budget_stops_the_campaign(tmp_path) -> None:
    orch = make_orchestrator(
        [static_engine(static_clean), fuzz_engine()], make_request(tmp_path), max_engines=1
    )
    state = orch.run()
    assert state.stop_reason == StopReason.EXECUTION_CAP_EXHAUSTED.value
    assert orch.scheduler.engines_started == 1


def test_unavailable_engines_do_not_consume_rounds_or_engines(tmp_path) -> None:
    down = static_engine()
    down.available = False
    orch = make_orchestrator([down], make_request(tmp_path))
    state = orch.run()
    assert sum(state.budget.consumed.values()) == 0
    assert state.round == 0


def test_an_engine_that_cannot_run_the_capability_is_recorded_unsupported_and_not_retried(
    tmp_path,
) -> None:
    def refuse(req):
        return DynamicResult(
            engine="ityfuzz",
            language=req.language,
            target=req.target,
            status=ResultStatus.NOT_IMPLEMENTED,
            executed=False,
        )

    orch = make_orchestrator(
        [static_engine(static_clean), fuzz_engine(refuse)], make_request(tmp_path)
    )
    state = orch.run()
    refused = [r for r in state.executions if r.engine == "ityfuzz"]
    assert len(refused) == 1 and refused[0].phase is ExecutionPhase.UNSUPPORTED
    assert refused[0].consumed == {"attempts": 1}
    assert state.budget.consumed.get("fuzz_runs", 0) == 0
    assert "fuzzing" not in state.exercised_capabilities
    assert state.evidence and all(
        i.engine != "ityfuzz" or i.quality.value == "unsupported" for i in state.evidence
    )


def test_a_result_naming_another_engine_or_language_is_not_trusted(tmp_path) -> None:
    def impostor(req):
        return ok("slither", req, findings=())

    orch = make_orchestrator(
        [static_engine(static_clean), fuzz_engine(impostor)], make_request(tmp_path)
    )
    state = orch.run()
    record = next(r for r in state.executions if r.engine == "ityfuzz")
    assert record.phase is ExecutionPhase.FAILED and record.failure_class.value == "identity"
    assert not [i for i in state.evidence if i.engine == "slither"]

    def wrong_language(req):
        return ok("ityfuzz", req, language="rust")

    other = make_orchestrator(
        [static_engine(static_clean), fuzz_engine(wrong_language)], make_request(tmp_path)
    )
    record = next(r for r in other.run().executions if r.engine == "ityfuzz")
    assert record.failure_class.value == "identity"

    def wrong_campaign(req):
        return ok("ityfuzz", req, campaign_id="someone-else")

    third = make_orchestrator(
        [static_engine(static_clean), fuzz_engine(wrong_campaign)], make_request(tmp_path)
    )
    record = next(r for r in third.run().executions if r.engine == "ityfuzz")
    assert record.failure_class.value == "identity"


def test_an_engine_that_raises_is_an_unknown_outcome_that_keeps_its_cost(tmp_path) -> None:
    def explode(req):
        raise RuntimeError("tool crashed")

    orch = make_orchestrator(
        [static_engine(static_clean), fuzz_engine(explode)], make_request(tmp_path)
    )
    state = orch.run()
    record = next(r for r in state.executions if r.engine == "ityfuzz")
    assert record.phase is ExecutionPhase.UNKNOWN and record.executed is False
    assert state.budget.consumed["fuzz_runs"] == 1
    assert state.health[-1].consecutive_failures == 1
    assert state.stop_reason == StopReason.ENVIRONMENT_UNAVAILABLE.value
    assert "tool crashed" not in repr(state)


def test_an_open_circuit_blocks_an_engine_until_it_recovers(tmp_path) -> None:
    request = make_request(tmp_path)
    scheduler = make_scheduler([static_engine()])
    from app.discovery.orchestration.budget import new_ledger
    from app.discovery.orchestration.model import ResearchState

    identity = identity_from_request(request)
    state = ResearchState(identity=identity, budget=new_ledger(scheduler))
    state.health = (EngineHealth("bugforge-static", consecutive_failures=2),)
    from app.discovery.orchestration.assess import AssessContext, assess
    from app.discovery.orchestration.gate import DefaultGate

    needs = assess(state, AssessContext(language="solidity", extra=request.extra))
    result = plan(
        state=state,
        needs=needs,
        scheduler=scheduler,
        request=request,
        identity=identity,
        gate=DefaultGate(),
    )
    assert result.considered[0].rejection == "environment:circuit_open"
    state.health = (EngineHealth("bugforge-static", consecutive_failures=1, timeouts=1),)
    healthy = plan(
        state=state,
        needs=needs,
        scheduler=scheduler,
        request=request,
        identity=identity,
        gate=DefaultGate(),
    )
    assert healthy.selected is not None
    assert healthy.selected.components["health"] == -7


def test_an_engine_that_would_not_run_the_requested_capability_is_not_selectable(tmp_path) -> None:
    class Stubborn(FakeEngine):
        def selected_capability(self, request):  # always symbolic, whatever was asked
            return C.SYMBOLIC_EXECUTION

    engine = Stubborn("ityfuzz", {C.FUZZING, C.SYMBOLIC_EXECUTION})
    orch = make_orchestrator([static_engine(static_clean), engine], make_request(tmp_path))
    state = orch.run()
    fuzz = [
        c
        for r in state.decisions
        for c in r.considered
        if c.engine == "ityfuzz" and c.capability == "fuzzing"
    ]
    assert fuzz and all(c.rejection == "prerequisite:capability_not_selectable" for c in fuzz)
    assert engine.calls == []


def test_blocked_is_visited_only_for_resumable_stops(tmp_path, monkeypatch) -> None:
    visited: list[OrchestratorState] = []
    original = orchestrator_module.transition

    def spy(current, target, **kwargs):
        visited.append(target)
        return original(current, target, **kwargs)

    monkeypatch.setattr(orchestrator_module, "transition", spy)
    down = static_engine()
    down.available = False
    make_orchestrator([down], make_request(tmp_path)).run()
    assert OrchestratorState.BLOCKED in visited
    assert visited[-2:] == [OrchestratorState.STOPPING, OrchestratorState.STOPPED]
    visited.clear()
    make_orchestrator([static_engine(static_clean)], _bare(tmp_path)).run()
    assert OrchestratorState.BLOCKED not in visited
    assert visited[-1] is OrchestratorState.COMPLETED


def test_an_internal_error_fails_the_campaign_closed_and_is_remembered(
    tmp_path, monkeypatch
) -> None:
    store = MemoryStore()

    def broken(**_kwargs):
        raise InvalidDecisionError("bad plan")

    monkeypatch.setattr(orchestrator_module, "plan", broken)
    orch = make_orchestrator([static_engine()], make_request(tmp_path), store=store)
    state = orch.run()
    assert state.state is OrchestratorState.FAILED
    assert "bad plan" in state.rationale
    assert store.load(orch.identity.campaign_id).state is OrchestratorState.FAILED  # type: ignore[union-attr]
    assert orch.step() is False
    resumed = make_orchestrator([static_engine()], make_request(tmp_path), store=store)
    assert resumed.start().state is OrchestratorState.FAILED
    assert resumed.step() is False


def test_persistence_errors_are_not_swallowed(tmp_path) -> None:
    class Failing(MemoryStore):
        def save(self, state):
            if state.round >= 1:
                raise PersistenceError("disk full")
            super().save(state)

    orch = make_orchestrator(
        [static_engine(), fuzz_engine()], make_request(tmp_path), store=Failing()
    )
    with pytest.raises(PersistenceError):
        orch.run()


def test_two_unproductive_repeats_of_one_capability_stop_as_redundant(tmp_path) -> None:
    orch = make_orchestrator([static_engine(static_clean)], make_request(tmp_path))
    orch.run()
    state = orch.state
    base = state.decisions[0]
    fuzz = replace(base, selected_capability="fuzzing", evidence_delta=EvidenceDelta())
    state.decisions = (replace(fuzz, sequence=1), replace(fuzz, sequence=2))
    assert orch._redundant() is True
    informative = replace(fuzz, evidence_delta=EvidenceDelta(new_corpus_seeds=("a",)))
    state.decisions = (fuzz, informative)
    assert orch._redundant() is False
    other = replace(fuzz, selected_capability="symbolic_execution")
    state.decisions = (fuzz, other)
    assert orch._redundant() is False


def test_stalled_fuzzing_is_detected_only_from_a_completed_uninformative_round(tmp_path) -> None:
    from app.discovery.orchestration.assess import stalled

    orch = make_orchestrator([static_engine(static_clean), fuzz_engine()], make_request(tmp_path))
    orch.run()
    assert stalled(orch.state) is True
    fresh = make_orchestrator([static_engine()], make_request(tmp_path))
    assert stalled(fresh.state) is False


def test_the_decision_record_explains_the_choice(tmp_path) -> None:
    orch = make_orchestrator([static_engine(static_clean), fuzz_engine()], make_request(tmp_path))
    state = orch.run()
    record = state.decisions[1]
    assert record.selected_capability == "fuzzing" and record.selected_engine == "ityfuzz"
    assert record.missing_capability == "fuzzing"
    assert record.needs and record.considered
    assert record.budget_before["rounds"] == orch.state.budget.limits["rounds"] - 1
    assert record.reserved_cost["fuzz_runs"] == 1
    assert record.consumed_cost["fuzz_runs"] == 1
    assert record.previous_state == "planning"
    assert record.identity_digest == orch.identity.target_digest()
    assert record.source_snapshot == "snap1" and record.compiler_configuration == "solc-0.8.24"
    assert record.input_digest and record.execution_id
    assert record.orchestrator_version == state.orchestrator_version
    assert record.record_hash == record.sealed().record_hash
    stop = state.decisions[-1]
    assert stop.stop_reason == state.stop_reason
    assert stop.previous_state == "planning"
    assert [r.sequence for r in state.decisions] == list(range(1, len(state.decisions) + 1))


def test_a_refused_reservation_stops_cleanly_instead_of_failing(tmp_path, monkeypatch) -> None:
    from app.discovery.orchestration.model import BudgetLedger

    def refuse(self, cost):
        raise InvalidDecisionError("budget:attempts")

    monkeypatch.setattr(BudgetLedger, "reserve", refuse)
    orch = make_orchestrator([static_engine()], make_request(tmp_path))
    state = orch.run()
    assert state.state is OrchestratorState.STOPPED
    assert state.stop_reason == StopReason.BUDGET_EXHAUSTED.value
    assert state.executions == ()
