"""End-to-end scenarios A to K. Every engine is scripted and no tool runs."""

from __future__ import annotations

import pytest

from app.discovery.capabilities import ResultStatus
from app.discovery.orchestration import (
    DefaultGate,
    MemoryStore,
    OrchestratorState,
    ResumeError,
    ResumeStatus,
    StopReason,
)
from app.discovery.orchestration.model import (
    EvidenceQuality,
    ExecutionPhase,
    FailureClass,
)
from app.discovery.results import DynamicResult
from tests.test_phase49.phase49_runtime import (
    FORK_EXTRA,
    RUNTIME_EXTRA,
    economic_engine,
    protocol_engine,
    runtime_engine,
    runtime_result,
)
from tests.test_phase49.phase49_support import (
    finding,
    fuzz_engine,
    make_orchestrator,
    make_request,
    ok,
    static_clean,
    static_engine,
    static_with_finding,
    symbolic_engine,
)


def _capabilities(orch) -> list[str]:
    return [r.selected_capability for r in orch.state.decisions if r.selected_capability]


def test_a_static_candidate_leads_to_fuzzing_then_stops_for_the_missing_capability(
    tmp_path,
) -> None:
    request = make_request(tmp_path)
    orch = make_orchestrator([static_engine(), fuzz_engine()], request)
    state = orch.run()
    assert _capabilities(orch)[:2] == ["static_analysis", "fuzzing"]
    # a reentrancy candidate needs protocol evidence and none can be produced here
    assert state.stop_reason == StopReason.ENVIRONMENT_UNAVAILABLE.value
    assert state.state is OrchestratorState.STOPPED
    assert orch.report()["verified"] is False
    assert all(item.quality is not None for item in state.evidence)


def test_a_complete_research_with_all_capabilities_available(tmp_path) -> None:
    request = make_request(tmp_path)
    orch = make_orchestrator(
        [static_engine(), protocol_engine(), fuzz_engine(), symbolic_engine()], request
    )
    state = orch.run()
    assert _capabilities(orch) == [
        "static_analysis",
        "fuzzing",
        "symbolic_execution",
        "cross_contract_analysis",
    ]
    assert state.state in {OrchestratorState.COMPLETED, OrchestratorState.INCONCLUSIVE}
    assert state.stop_reason in {
        StopReason.ALL_CAPABILITIES_EXERCISED.value,
        StopReason.COMPLETED_BOUNDED_RESEARCH.value,
    }
    paths = [item for item in state.evidence if item.kind == "protocol_path"]
    assert len(paths) == 2


def test_b_stalled_fuzzing_triggers_symbolic_and_the_seed_feeds_a_fuzz_follow_up(tmp_path) -> None:
    request = make_request(tmp_path)
    holder: dict[str, object] = {}

    def symbolic_script(req):
        from tests.test_phase49.phase49_support import add_symbolic_seed

        add_symbolic_seed(holder["orch"].scheduler, req)  # type: ignore[attr-defined]
        return ok("halmos", req, metadata={"seed_source": "symbolic"})

    engines = [static_engine(static_clean), fuzz_engine(), symbolic_engine(symbolic_script)]
    orch = make_orchestrator(engines, request, max_rounds=8, max_engines=8)
    holder["orch"] = orch
    orch.run()
    order = _capabilities(orch)
    assert order[:3] == ["static_analysis", "fuzzing", "symbolic_execution"]
    assert order[3] == "fuzzing"
    follow_up = [
        r for r in orch.state.decisions if r.sequence and r.selected_capability == "fuzzing"
    ]
    assert follow_up[0].input_digest != follow_up[1].input_digest
    assert follow_up[1].evidence_delta.new_corpus_seeds == ()
    assert any(ref.startswith("symbolic_execution:") for ref in orch.state.corpus_refs)


def test_c_economic_observation_triggers_a_runtime_follow_up(tmp_path) -> None:
    request = make_request(tmp_path, economic="true", case="erc20", **RUNTIME_EXTRA)
    request.extra.pop("runtime")
    engines = [static_engine(static_clean), economic_engine(), runtime_engine(), fuzz_engine()]
    orch = make_orchestrator(engines, request)
    orch.run()
    order = _capabilities(orch)
    assert "economic_simulation" in order and "runtime_validation" in order
    runtime = [r for r in orch.state.decisions if r.selected_capability == "runtime_validation"]
    assert order.index("economic_simulation") < len(order) - 1 - order[::-1].index(
        "runtime_validation"
    )
    assert len(runtime) == 2
    assert runtime[0].input_digest != runtime[1].input_digest
    assert "economic observation" in runtime[1].rationale
    assert runtime[1].missing_capability == "runtime_validation"


def test_d_fork_needs_approval_and_a_pinned_fork_and_never_invents_either(tmp_path) -> None:
    request = make_request(tmp_path, fork="true", sequence_id="seq-1", **FORK_EXTRA)
    engines = [static_engine(static_clean), fuzz_engine(), runtime_engine()]
    refused = make_orchestrator(engines, request)
    state = refused.run()
    assert state.stop_reason == StopReason.APPROVAL_REQUIRED.value
    assert state.state is OrchestratorState.STOPPED
    assert "fork_validation" not in state.exercised_capabilities
    assert [c.extra["mode"] for c in engines[2].calls] == ["local"]

    approved_engines = [static_engine(static_clean), fuzz_engine(), runtime_engine()]
    approved = make_orchestrator(
        approved_engines, request, gate=DefaultGate(approvals=frozenset({"fork_validation"}))
    )
    approved.run()
    assert "fork_validation" in approved.state.exercised_capabilities
    call = next(c for c in approved_engines[2].calls if c.extra["mode"] == "fork")
    assert call.extra["fork_block"] == FORK_EXTRA["fork_block"]
    assert "state_reset" not in call.extra


def test_d_unpinned_fork_is_a_missing_prerequisite_not_a_run(tmp_path) -> None:
    request = make_request(tmp_path, fork="true", sequence_id="seq-1")
    engines = [static_engine(static_clean), fuzz_engine(), runtime_engine()]
    orch = make_orchestrator(
        engines, request, gate=DefaultGate(approvals=frozenset({"fork_validation"}))
    )
    state = orch.run()
    assert state.stop_reason == StopReason.PREREQUISITES_UNAVAILABLE.value
    assert "fork" not in [c.extra["mode"] for c in engines[2].calls]
    blocked = state.decisions[-1]
    assert "prerequisite:pinned_fork" in blocked.prerequisites


def test_e_divergence_creates_a_contradiction_that_needs_a_discriminator(tmp_path) -> None:
    request = make_request(
        tmp_path, differential="true", state_reset="true", **RUNTIME_EXTRA, **FORK_EXTRA
    )

    def script(req):
        mode = req.extra.get("mode", "local")
        if mode == "differential":
            return runtime_result(req, mode=mode, classification="deterministic divergence")
        return runtime_result(req, mode=mode, classification="deterministic same result")

    engines = [
        static_engine(static_clean),
        fuzz_engine(),
        symbolic_engine(),
        runtime_engine(script),
    ]
    gate = DefaultGate(approvals=frozenset({"fork_validation"}))
    orch = make_orchestrator(engines, request, gate=gate, max_rounds=8, max_engines=8)
    state = orch.run()
    assert state.contradictions
    kinds = {item.kind for item in state.contradictions}
    assert "runtime_divergence" in kinds
    assert "fork_validation" in state.exercised_capabilities
    # neither side is chosen and nothing is called resolved
    assert all(item.status in {"open", "unresolved"} for item in state.contradictions)
    assert state.stop_reason == StopReason.UNRESOLVED_UNCERTAINTY.value
    assert state.state is OrchestratorState.INCONCLUSIVE
    assert state.recommend_verification_review is False


def test_f_unavailable_engine_falls_back_to_another_for_the_same_capability(tmp_path) -> None:
    request = make_request(tmp_path)
    missing = static_engine(engine_id="slither")
    missing.available = False
    present = static_engine(static_clean, "bugforge-static")
    orch = make_orchestrator([missing, present, fuzz_engine()], request)
    orch.run()
    assert missing.calls == []
    assert orch.state.executions[0].engine == "bugforge-static"
    first = orch.state.decisions[0]
    rejected = {c.engine: c.rejection for c in first.considered if not c.eligible}
    assert rejected["slither"] == "environment:engine_unavailable"
    assert "slither" not in orch.state.successful_engines


def test_f_no_engine_available_stops_without_fabricating_evidence(tmp_path) -> None:
    request = make_request(tmp_path)
    only = static_engine()
    only.available = False
    orch = make_orchestrator([only], request)
    state = orch.run()
    assert state.stop_reason == StopReason.ENVIRONMENT_UNAVAILABLE.value
    assert state.evidence == ()
    assert state.executions == ()
    assert state.budget.consumed.get("engines", 0) == 0


def test_g_round_budget_is_a_hard_cap_and_resume_cannot_raise_it(tmp_path) -> None:
    request = make_request(tmp_path)
    store = MemoryStore()
    orch = make_orchestrator([static_engine(), fuzz_engine()], request, store=store, max_rounds=1)
    state = orch.run()
    assert state.round == 1
    assert state.stop_reason == StopReason.ROUNDS_EXHAUSTED.value
    assert state.state is OrchestratorState.STOPPED
    roomy = make_orchestrator([static_engine(), fuzz_engine()], request, store=store, max_rounds=6)
    resumed = roomy.start(explicit_resume=True)
    assert resumed.budget.limits["rounds"] == 1
    assert resumed.state is OrchestratorState.STOPPED
    assert roomy.step() is False


def test_h_a_crash_mid_execution_is_unknown_and_never_re_run(tmp_path) -> None:
    request = make_request(tmp_path)
    store = MemoryStore()
    static = static_engine(static_clean)

    class CrashError(Exception):
        pass

    def exploding(req):
        raise CrashError

    orch = make_orchestrator([static, fuzz_engine(exploding)], request, store=store)
    # the engine raising is an unknown outcome, and its budget stays spent
    state = orch.run()
    unknown = [r for r in state.executions if r.phase is ExecutionPhase.UNKNOWN]
    assert unknown and unknown[0].failure_class is FailureClass.UNKNOWN_OUTCOME
    assert unknown[0].retryable
    assert state.budget.consumed["fuzz_runs"] == 1
    again = make_orchestrator([static_engine(static_clean), fuzz_engine()], request, store=store)
    restored = again.start()
    assert restored.resume_status is ResumeStatus.RESTORED
    assert "fuzzing" not in restored.exercised_capabilities
    assert restored.budget.consumed["fuzz_runs"] >= 1


def test_h_stale_identity_is_refused_not_reused(tmp_path) -> None:
    store = MemoryStore()
    first = make_orchestrator([static_engine()], make_request(tmp_path), store=store)
    first.run()
    changed = make_request(tmp_path, source_snapshot="snap2")
    second = make_orchestrator([static_engine()], changed, store=store)
    second.identity = type(first.identity)(
        **{**first.identity.__dict__, "source_snapshot": "snap2"}
    )
    with pytest.raises(ResumeError) as excinfo:
        second.start()
    assert excinfo.value.status is ResumeStatus.STALE_IDENTITY


def test_i_timeouts_open_the_engine_circuit_and_a_retry_needs_a_grant(tmp_path) -> None:
    request = make_request(tmp_path)

    def timing_out(req):
        return DynamicResult(
            engine="ityfuzz",
            language=req.language,
            target=req.target,
            status=ResultStatus.TIMEOUT,
            executed=True,
        )

    orch = make_orchestrator([static_engine(static_clean), fuzz_engine(timing_out)], request)
    state = orch.run()
    timed = [r for r in state.executions if r.phase is ExecutionPhase.TIMED_OUT]
    assert len(timed) == 1
    assert timed[0].failure_class is FailureClass.TIMEOUT
    health = next(h for h in state.health if h.engine == "ityfuzz")
    assert health.timeouts == 1 and health.consecutive_failures == 1
    assert [i.quality for i in state.evidence if i.engine == "ityfuzz"] == [
        EvidenceQuality.INCOMPLETE
    ]
    assert [n for n in state.negatives if n.engine == "ityfuzz"] == []
    calls_before = len(orch.scheduler.engines[1].calls)  # type: ignore[attr-defined]
    assert orch.grant_retry(timed[0].execution_key) is True
    assert orch.grant_retry(timed[0].execution_key) is False
    assert calls_before == 1


def test_j_missing_snapshot_identity_blocks_protocol_and_runtime_research(tmp_path) -> None:
    request = make_request(tmp_path, protocol="true")
    request.extra.pop("source_snapshot")
    engines = [static_engine(static_clean), protocol_engine(), fuzz_engine()]
    orch = make_orchestrator(engines, request)
    state = orch.run()
    assert state.stop_reason == StopReason.DETERMINISTIC_IDENTITY_MISSING.value
    assert state.state is OrchestratorState.INCONCLUSIVE
    assert engines[1].calls == []


def test_k_two_engines_corroborate_a_candidate_and_it_still_is_not_verified(tmp_path) -> None:
    request = make_request(tmp_path)

    def fuzz_hit(req):
        return ok(
            "ityfuzz",
            req,
            findings=(finding(),),
            metadata={"evidence_class": "fuzzing"},
        )

    engines = [static_engine(static_with_finding), protocol_engine(), fuzz_engine(fuzz_hit)]
    orch = make_orchestrator(engines, request)
    state = orch.run()
    corroborated = [i for i in state.evidence if i.quality is EvidenceQuality.CORROBORATED]
    assert {i.engine for i in corroborated} == {"bugforge-static", "ityfuzz"}
    assert state.recommend_verification_review is True
    report = orch.report()
    assert report["verified"] is False
    assert all(i.attrs.get("verified", "false") == "false" for i in state.evidence)
    assert not hasattr(state, "verified")


def test_k_no_finding_is_negative_evidence_and_never_safety(tmp_path) -> None:
    request = make_request(tmp_path)
    orch = make_orchestrator([static_engine(static_clean), fuzz_engine()], request)
    state = orch.run()
    assert state.negatives
    assert all(item.proves_safety is False for item in state.negatives)
    assert all("not evidence of safety" in item.statement for item in state.negatives)


def test_planner_never_runs_a_candidate_twice(tmp_path) -> None:
    request = make_request(tmp_path)
    engines = [static_engine(), fuzz_engine(), protocol_engine(), symbolic_engine()]
    orch = make_orchestrator(engines, request, max_rounds=12, max_engines=12)
    orch.run()
    keys = [r.execution_key for r in orch.state.executions]
    assert len(keys) == len(set(keys))
    for engine in engines:
        assert len(engine.calls) <= 2


def test_terminal_state_is_idempotent(tmp_path) -> None:
    request = make_request(tmp_path)
    orch = make_orchestrator([static_engine(static_clean)], request)
    first = orch.run()
    snapshot = (first.state, first.decision_number, first.round, first.hash_now())
    again = orch.run()
    assert (again.state, again.decision_number, again.round, again.hash_now()) == snapshot
    assert orch.step() is False


def test_engine_reporting_the_wrong_identity_is_not_attributed(tmp_path) -> None:
    request = make_request(tmp_path)

    def wrong_snapshot(req):
        return ok(
            "bugforge-static",
            req,
            status=ResultStatus.INGESTED,
            findings=(finding(),),
            metadata={"source_snapshot": "other-snapshot"},
        )

    orch = make_orchestrator([static_engine(wrong_snapshot)], request)
    state = orch.run()
    record = state.executions[0]
    assert record.phase is ExecutionPhase.FAILED
    assert record.failure_class is FailureClass.IDENTITY
    assert state.evidence == ()


def test_d_approval_is_never_stored_so_resume_needs_the_caller_to_grant_it_again(tmp_path) -> None:
    request = make_request(tmp_path, fork="true", sequence_id="seq-1", **FORK_EXTRA)
    store = MemoryStore()
    engines = [static_engine(static_clean), fuzz_engine(), runtime_engine()]
    first = make_orchestrator(engines, request, store=store)
    assert first.run().stop_reason == StopReason.APPROVAL_REQUIRED.value
    assert '"approvals"' not in store.documents[first.identity.campaign_id]

    no_approval = make_orchestrator(engines, request, store=store)
    resumed = no_approval.start(explicit_resume=True)
    assert resumed.state is OrchestratorState.PLANNING
    assert no_approval.run().stop_reason == StopReason.APPROVAL_REQUIRED.value

    granted = make_orchestrator(
        engines, request, store=store, gate=DefaultGate(approvals=frozenset({"fork_validation"}))
    )
    granted.start(explicit_resume=True)
    final = granted.run()
    assert "fork_validation" in final.exercised_capabilities
    assert final.resume_count == 2
