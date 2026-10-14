from __future__ import annotations

import itertools
from dataclasses import replace

from app.discovery.capabilities import EngineCapability
from app.discovery.orchestration import DefaultGate, GateVerdict, StopReason
from app.discovery.orchestration.assess import AssessContext, assess
from app.discovery.orchestration.model import (
    CapabilityNeed,
    ExecutionPhase,
    ExecutionRecord,
    FailureClass,
    NeedStatus,
    ResearchState,
)
from app.discovery.orchestration.orchestrator import identity_from_request
from app.discovery.orchestration.planner import (
    CAPABILITY_MODE,
    Suggestion,
    build_request,
    input_digest,
    plan,
    prerequisite,
    validate_suggestions,
)
from app.discovery.scheduler import engine_rank
from tests.test_phase49.phase49_runtime import FORK_EXTRA, runtime_engine
from tests.test_phase49.phase49_support import (
    FakeEngine,
    fuzz_engine,
    make_request,
    make_scheduler,
    static_engine,
)

C = EngineCapability


def _state(request, scheduler) -> tuple[ResearchState, object]:
    from app.discovery.orchestration.budget import new_ledger

    identity = identity_from_request(request)
    return ResearchState(identity=identity, budget=new_ledger(scheduler)), identity


def _plan(engines, request, *, state=None, gate=None, suggestions=(), needs=None):
    scheduler = make_scheduler(list(engines))
    base, identity = _state(request, scheduler)
    state = state or base
    ctx = AssessContext(
        language=request.language, extra=request.extra, targeted=bool(request.function)
    )
    return plan(
        state=state,
        needs=needs if needs is not None else assess(state, ctx),
        scheduler=scheduler,
        request=request,
        identity=identity,
        gate=gate or DefaultGate(),
        suggestions=suggestions,
    )


def test_baseline_static_is_chosen_first_and_recorded(tmp_path) -> None:
    result = _plan([fuzz_engine(), static_engine()], make_request(tmp_path))
    assert result.selected is not None
    assert (result.selected.engine, result.selected.capability) == (
        "bugforge-static",
        "static_analysis",
    )
    assert result.selected.components["need"] == 100
    assert result.selected.score == sum(result.selected.components.values())
    assert result.request is not None and "mode" not in result.request.extra
    assert result.rationale


def test_ranking_is_independent_of_engine_registration_order(tmp_path) -> None:
    request = make_request(tmp_path)
    engines = [
        static_engine(),
        static_engine(engine_id="slither"),
        fuzz_engine(),
        fuzz_engine(engine_id="echidna"),
    ]
    outcomes = set()
    for ordering in itertools.permutations(engines):
        result = _plan(ordering, request)
        outcomes.add(
            (
                result.selected.engine if result.selected else "",
                tuple((c.engine, c.capability, c.score) for c in result.considered),
            )
        )
    assert len(outcomes) == 1


def test_ties_break_by_engine_rank_then_id(tmp_path) -> None:
    request = make_request(tmp_path)
    twin_a = static_engine(engine_id="zeta-static")
    twin_b = static_engine(engine_id="alpha-static")
    result = _plan([twin_a, twin_b], request)
    assert result.selected is not None
    assert result.selected.engine == "alpha-static"
    assert engine_rank("bugforge-static", "solidity") < engine_rank("slither", "solidity")
    assert engine_rank("unknown", "solidity") > engine_rank("wake", "solidity")


def test_unavailable_and_unsupported_engines_are_rejected_with_a_reason(tmp_path) -> None:
    down = static_engine(engine_id="slither")
    down.available = False
    wrong_language = FakeEngine(
        "bugforge-static", {C.STATIC_ANALYSIS}, languages=frozenset({"python"})
    )
    result = _plan([down, wrong_language], make_request(tmp_path))
    reasons = {c.engine: c.rejection for c in result.considered}
    assert reasons == {
        "slither": "environment:engine_unavailable",
        "bugforge-static": "environment:language",
    }
    assert result.selected is None
    assert result.stop is StopReason.ENVIRONMENT_UNAVAILABLE


def test_capability_the_engine_would_not_actually_run_is_not_selectable(tmp_path) -> None:
    both = FakeEngine("ityfuzz", {C.FUZZING, C.SYMBOLIC_EXECUTION})
    request = make_request(tmp_path)
    assert (
        build_request(
            request, "fuzzing", identity_from_request(request), make_scheduler([both])
        ).extra["mode"]
        == "fuzz"
    )
    result = _plan([static_engine(), both], request)
    assert result.selected is not None and result.selected.capability == "static_analysis"
    only_property = FakeEngine("ityfuzz", {C.FUZZING, C.TEST_EXECUTION})
    # test_execution picks the test capability with mode "test", so fuzzing stays selectable
    assert (
        _plan(
            [only_property], request, needs=(CapabilityNeed("fuzzing", NeedStatus.MISSING, 70),)
        ).selected
        is not None
    )


def test_the_orchestrator_only_ever_sets_mode(tmp_path) -> None:
    request = make_request(tmp_path, network="none", state_reset="false")
    identity = identity_from_request(request)
    scheduler = make_scheduler([])
    for capability in CAPABILITY_MODE:
        built = build_request(request, capability, identity, scheduler)
        changed = {
            k for k in {*built.extra, *request.extra} if built.extra.get(k) != request.extra.get(k)
        }
        assert changed <= {"mode"}
        assert built.extra.get("network") == "none"
        assert built.extra.get("state_reset") == "false"


def test_prerequisites_are_reported_and_never_invented(tmp_path) -> None:
    request = make_request(tmp_path)
    identity = identity_from_request(request)
    assert prerequisite("static_analysis", request, identity) == ""
    assert prerequisite("fuzzing", request, identity) == ""
    assert prerequisite("economic_simulation", request, identity) == "prerequisite:economic_case"
    assert prerequisite("runtime_validation", request, identity) == "prerequisite:sequence_id"
    with_sequence = replace(request, extra={**request.extra, "sequence_id": "s"})
    assert prerequisite("runtime_validation", with_sequence, identity) == ""
    assert prerequisite("fork_validation", with_sequence, identity) == "prerequisite:pinned_fork"
    pinned = replace(with_sequence, extra={**with_sequence.extra, **FORK_EXTRA})
    assert prerequisite("fork_validation", pinned, identity) == ""
    assert prerequisite("differential_validation", pinned, identity) == "prerequisite:state_reset"
    reset = replace(pinned, extra={**pinned.extra, "state_reset": "true"})
    assert prerequisite("differential_validation", reset, identity) == ""
    bare = replace(request, contract="", function="", source_file="", target="")
    assert prerequisite("fuzzing", bare, identity_from_request(bare)) == "identity:target"
    no_snapshot = replace(request, extra={"project_id": "p"})
    assert prerequisite(
        "cross_contract_analysis", no_snapshot, identity_from_request(no_snapshot)
    ) == ("identity:source_snapshot")
    half = replace(request, extra={"project_id": "p", "source_snapshot": "s"})
    assert prerequisite("runtime_validation", half, identity_from_request(half)) == (
        "identity:compiler_configuration"
    )
    function_only = replace(with_sequence, contract="")
    assert prerequisite(
        "runtime_validation", function_only, identity_from_request(function_only)
    ) == ("identity:contract_and_function")


def test_a_completed_execution_key_is_never_proposed_again(tmp_path) -> None:
    request = make_request(tmp_path)
    scheduler = make_scheduler([static_engine()])
    state, identity = _state(request, scheduler)
    first = _plan([static_engine()], request, state=state)
    assert first.selected is not None
    record = ExecutionRecord(
        execution_key=first.selected.execution_key,
        decision_id="d",
        engine="bugforge-static",
        capability="static_analysis",
        input_digest=first.selected.input_digest,
        digest_after="",
        phase=ExecutionPhase.COMPLETED,
        status="executed",
        executed=True,
        failure_class=FailureClass.NONE,
    )
    state.executions = (record,)
    second = _plan(
        [static_engine()],
        request,
        state=state,
        needs=(CapabilityNeed("static_analysis", NeedStatus.MISSING, 100),),
    )
    assert second.selected is None
    assert second.considered[0].rejection == "duplicate:execution_key"
    assert second.stop is StopReason.NO_USEFUL_CAPABILITY
    assert second.needs[0].status is NeedStatus.REDUNDANT


def test_a_failed_attempt_is_not_repeated_without_an_explicit_grant(tmp_path) -> None:
    request = make_request(tmp_path)
    state, _identity = _state(request, make_scheduler([static_engine()]))
    first = _plan([static_engine()], request, state=state)
    assert first.selected is not None
    for phase, failure in (
        (ExecutionPhase.TIMED_OUT, FailureClass.TIMEOUT),
        (ExecutionPhase.UNKNOWN, FailureClass.UNKNOWN_OUTCOME),
    ):
        state.executions = (
            ExecutionRecord(
                execution_key=first.selected.execution_key,
                decision_id="d",
                engine="bugforge-static",
                capability="static_analysis",
                input_digest=first.selected.input_digest,
                digest_after="",
                phase=phase,
                status="x",
                executed=False,
                failure_class=failure,
                retryable=True,
            ),
        )
        again = _plan(
            [static_engine()],
            request,
            state=state,
            needs=(CapabilityNeed("static_analysis", NeedStatus.MISSING, 100),),
        )
        assert again.selected is None
        assert again.considered[0].rejection == "environment:failed_attempt"
    state.executions = (replace(state.executions[0], phase=ExecutionPhase.SUPERSEDED),)
    granted = _plan(
        [static_engine()],
        request,
        state=state,
        needs=(CapabilityNeed("static_analysis", NeedStatus.MISSING, 100),),
    )
    assert granted.selected is not None
    assert granted.selected.execution_key != first.selected.execution_key


def test_a_changed_input_gets_a_new_digest_and_may_run_again(tmp_path) -> None:
    request = make_request(tmp_path)
    scheduler = make_scheduler([fuzz_engine()])
    state, identity = _state(request, scheduler)
    built = build_request(request, "fuzzing", identity, scheduler)
    before = input_digest(built, "fuzzing", state)
    assert input_digest(built, "fuzzing", state) == before
    state.corpus_refs = ("symbolic_execution:abc",)
    assert input_digest(built, "fuzzing", state) != before
    static_before = input_digest(built, "static_analysis", state)
    state.corpus_refs = ()
    assert input_digest(built, "static_analysis", state) == static_before
    noisy = replace(built, extra={**built.extra, "execution_id": "random", "timestamp": "now"})
    assert input_digest(noisy, "fuzzing", state) == before


def test_budget_shortfall_makes_a_candidate_ineligible_without_consuming(tmp_path) -> None:
    request = make_request(tmp_path)
    scheduler = make_scheduler([static_engine()])
    state, _identity = _state(request, scheduler)
    state.budget.limits["rounds"] = 0
    result = _plan([static_engine()], request, state=state)
    assert result.selected is None
    assert result.considered[0].rejection == "budget:rounds"
    assert result.stop is StopReason.BUDGET_EXHAUSTED
    assert state.budget.consumed == {} and state.budget.reserved == {}


def test_the_gate_decides_and_a_failing_gate_refuses(tmp_path) -> None:
    request = make_request(tmp_path)

    class Boom:
        def check(self, **_kwargs):
            raise RuntimeError("boom")

    class Silent:
        def check(self, **_kwargs):
            return None

    class Refuse:
        def check(self, **_kwargs):
            return GateVerdict(False)

    for gate, expected in (
        (Boom(), StopReason.SAFETY_BLOCKED),
        (Silent(), StopReason.SAFETY_BLOCKED),
        (Refuse(), StopReason.SAFETY_BLOCKED),
        (DefaultGate(scope_allowed=False), StopReason.SCOPE_BLOCKED),
    ):
        result = _plan([static_engine()], request, gate=gate)
        assert result.selected is None
        assert result.stop is expected


def test_network_requests_outside_approved_fork_validation_are_a_safety_stop(tmp_path) -> None:
    request = make_request(tmp_path, network="controlled-fork")
    result = _plan([static_engine()], request)
    assert result.stop is StopReason.SAFETY_BLOCKED
    assert result.selected is None


def test_fork_approval_comes_only_from_the_gate(tmp_path) -> None:
    request = make_request(tmp_path, fork="true", sequence_id="s", **FORK_EXTRA)
    state, _ = _state(request, make_scheduler([runtime_engine()]))
    needs = (CapabilityNeed("fork_validation", NeedStatus.MISSING, 45),)
    refused = _plan([runtime_engine()], request, needs=needs, state=state)
    assert refused.stop is StopReason.APPROVAL_REQUIRED
    allowed = _plan(
        [runtime_engine()],
        request,
        needs=needs,
        state=state,
        gate=DefaultGate(approvals=frozenset({"fork_validation"})),
    )
    assert allowed.selected is not None and allowed.selected.capability == "fork_validation"


def test_external_suggestions_are_validated_and_cannot_bypass_checks(tmp_path) -> None:
    accepted, rejected = validate_suggestions(
        (
            Suggestion("fuzzing"),
            Suggestion("not_a_capability"),
            Suggestion("results_ingestion"),
            Suggestion("fuzzing", engine="ghost"),
        ),
        frozenset({"ityfuzz"}),
    )
    assert [s.capability for s in accepted] == ["fuzzing"]
    assert len(rejected) == 3
    request = make_request(tmp_path, sequence_id="s")
    suggested = _plan(
        [static_engine(), runtime_engine()],
        request,
        suggestions=(Suggestion("runtime_validation", reason="please"),),
        needs=(CapabilityNeed("static_analysis", NeedStatus.SATISFIED, 0),),
    )
    # a suggestion only adds a low-value candidate; it is still gated and budgeted
    assert suggested.selected is not None and suggested.selected.capability == "runtime_validation"
    assert suggested.selected.components["suggestion"] == 5
    refused = _plan(
        [static_engine(), runtime_engine()],
        request,
        gate=DefaultGate(scope_allowed=False),
        suggestions=(Suggestion("runtime_validation"),),
        needs=(CapabilityNeed("static_analysis", NeedStatus.SATISFIED, 0),),
    )
    assert refused.selected is None and refused.stop is StopReason.SCOPE_BLOCKED
    unregistered = _plan(
        [static_engine()],
        request,
        suggestions=(Suggestion("fork_validation"),),
        needs=(CapabilityNeed("static_analysis", NeedStatus.SATISFIED, 0),),
    )
    assert unregistered.selected is None
    assert unregistered.stop is StopReason.ENVIRONMENT_UNAVAILABLE


def test_assessment_follows_the_evidence(tmp_path) -> None:
    from app.discovery.orchestration.evidence import items_from_result
    from tests.test_phase49.phase49_support import finding

    request = make_request(tmp_path)
    state, identity = _state(request, make_scheduler([]))
    ctx = AssessContext(language="solidity", extra=request.extra, targeted=True)
    first = {n.capability: n for n in assess(state, ctx)}
    assert first["static_analysis"].status is NeedStatus.MISSING
    assert first["fuzzing"].weight == 55
    assert "symbolic_execution" not in first and "runtime_validation" not in first
    from tests.test_phase49.test_evidence import _result

    items = items_from_result(
        _result(engine="bugforge-static", findings=(finding(),)),
        identity=identity,
        capability="static_analysis",
        execution_key="k",
        phase=ExecutionPhase.COMPLETED,
    )
    state.evidence = items
    state.executions = (
        ExecutionRecord(
            execution_key="k",
            decision_id="d",
            engine="bugforge-static",
            capability="static_analysis",
            input_digest="i",
            digest_after="",
            phase=ExecutionPhase.COMPLETED,
            status="e",
            executed=True,
            failure_class=FailureClass.NONE,
        ),
    )
    second = {n.capability: n for n in assess(state, ctx)}
    assert second["static_analysis"].status is NeedStatus.SATISFIED
    assert second["fuzzing"].weight == 70
    assert second["cross_contract_analysis"].status is NeedStatus.MISSING
