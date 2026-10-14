"""Report pack determinism and the end-to-end deterministic bounty campaign."""

from __future__ import annotations

import json

from app.discovery.bounty import report_pack
from app.discovery.bounty.advisories import match_advisories
from app.discovery.bounty.compiler_diff import NoCompilerBackend, run_differential
from app.discovery.bounty.engine import (
    BugforgeCompilerDifferentialEngine,
    BugforgeResearchEngine,
    load_sources,
)
from app.discovery.bounty.findings import build_findings
from app.discovery.bounty.gate import BountyGate
from app.discovery.bounty.priority import prioritize
from app.discovery.bounty.report_pack import SECTION_TITLES, build_report_pack
from app.discovery.bounty.vfcs import FeedbackSignal, SequenceIdentity, generate, minimize, mutate
from app.discovery.capabilities import EngineCapability, ResultStatus
from app.discovery.engine import AnalysisRequest
from app.discovery.orchestration import MemoryStore, Orchestrator
from app.discovery.orchestration.model import EvidenceQuality, ExecutionPhase
from app.discovery.results import DynamicResult
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase49.phase49_runtime import runtime_result
from tests.test_phase49.phase49_support import FakeEngine, finding, make_scheduler, ok
from tests.test_phase50.phase50_support import FIXTURES, manifest

C = EngineCapability
CALLER = "caller_context_vulnerable.sol"
FUNCTION = "multicall(bytes[])"


def _setup(**manifest_overrides):
    mani = manifest(**manifest_overrides)
    identity = mani.to_campaign_identity(
        contract="SelfTrustMulticall", function=FUNCTION, source_file=CALLER
    )
    return mani, identity


def _request(mani, identity, **extra):
    return AnalysisRequest(
        repo_root=FIXTURES,
        language="solidity",
        target="SelfTrustMulticall.multicall",
        contract="SelfTrustMulticall",
        function=FUNCTION,
        source_file=CALLER,
        files=(CALLER,),
        campaign_id=identity.campaign_id,
        extra={"project_id": "proj", **mani.request_extra(), **extra},
    )


def _sequence_for(mani, identity):
    sources = load_sources(
        AnalysisRequest(repo_root=FIXTURES, language="solidity", files=(CALLER,))
    )
    model = build_research_model(sources)
    seq_identity = SequenceIdentity(
        campaign_id=identity.campaign_id,
        source_snapshot=mani.source_commit,
        compiler_configuration=mani.compiler.fingerprint(),
        fork_reference=mani.fork_reference(),
        program_context=mani.identity_digest(),
    )
    result = generate(model, run_suite(model).candidates, seq_identity)
    chosen = next(s for s in result.sequences if s.template == "nested-dispatch")
    return model, chosen, result


# ---- scripted engines for the non-research capabilities -------------------------------------


def _static_engine() -> FakeEngine:
    def script(request: AnalysisRequest) -> DynamicResult:
        return ok(
            "bugforge-static",
            request,
            status=ResultStatus.INGESTED,
            findings=(finding("reentrancy-eth", "SelfTrustMulticall", FUNCTION),),
            provenance="bugforge_static",
            metadata={"program_context": request.extra.get("program_context", "")},
        )

    return FakeEngine("bugforge-static", {C.STATIC_ANALYSIS, C.RESULTS_INGESTION}, script)


def _runtime_engine(outcome: str) -> FakeEngine:
    def script(request: AnalysisRequest) -> DynamicResult:
        mode = request.extra.get("mode", "local")
        result = runtime_result(
            request,
            mode=mode,
            outcome=outcome,
            classification="deterministic same result" if mode in {"fork", "differential"} else "",
        )
        result.metadata["program_context"] = request.extra.get("program_context", "")
        return result

    return FakeEngine(
        "bugforge-runtime",
        {C.RUNTIME_VALIDATION, C.FORK_VALIDATION, C.DIFFERENTIAL_VALIDATION},
        script,
    )


def _fuzz_and_symbolic() -> list[FakeEngine]:
    return [FakeEngine("ityfuzz", {C.FUZZING}), FakeEngine("halmos", {C.SYMBOLIC_EXECUTION})]


def _run(mani, identity, request, extra_engines, *, approvals=frozenset(), max_rounds=14):
    research = BugforgeResearchEngine(mani)
    diff = BugforgeCompilerDifferentialEngine(mani, NoCompilerBackend())
    scheduler = make_scheduler(
        [research, diff, *extra_engines], max_engines=14, max_rounds=max_rounds
    )
    orch = Orchestrator(
        scheduler,
        request,
        identity=identity,
        gate=BountyGate(mani, approvals=approvals),
        store=MemoryStore(),
        clock=lambda: "2026-01-01T00:00:00+00:00",
    )
    return orch, orch.run()


# ---- end to end -----------------------------------------------------------------------------


def test_end_to_end_campaign_from_manifest_to_report_package() -> None:
    mani, identity = _setup()
    _model, sequence, _all = _sequence_for(mani, identity)
    request = _request(
        mani,
        identity,
        runtime="true",
        sequence_id=sequence.sequence_id,
        runtime_configuration="pinned-local",
        fork="true",
    )
    runtime = _runtime_engine("reverted")
    others = [_static_engine(), *_fuzz_and_symbolic(), runtime]
    orch, state = _run(mani, identity, request, others, approvals=frozenset({"fork_validation"}))

    capabilities = [d.selected_capability for d in state.decisions if d.selected_capability]
    for expected in (
        "bounty_context_analysis",
        "static_analysis",
        "caller_context_analysis",
        "vfcs_generation",
        "fuzzing",
        "runtime_validation",
    ):
        assert expected in capabilities, capabilities
    assert len(capabilities) == len(set(capabilities)), "no capability repeats without new evidence"

    # identity and budget are preserved through every engine call
    for engine in (*others, *[]):
        for call in engine.calls:
            assert call.extra["program_context"] == mani.identity_digest()
            assert call.extra["source_snapshot"] == "abc1234"
            assert call.extra["compiler_configuration"] == mani.compiler.fingerprint()
    assert state.identity.program_context == mani.identity_digest()
    assert state.budget.limits["engines"] == 14
    assert state.budget.limits["rounds"] <= 14
    assert sum(1 for e in state.executions if e.phase is ExecutionPhase.COMPLETED) <= 14

    # the fork ran only because scope, mode, pinned fork, and approval all held
    fork_calls = [c for c in runtime.calls if c.extra.get("mode") == "fork"]
    for call in fork_calls:
        assert call.extra["chain_id"] == "1" and call.extra["fork_block"] == "19000000"

    # missing tools do not fabricate results
    diff_records = [e for e in state.executions if e.engine == "bugforge-compiler-diff"]
    assert not any(e.phase is ExecutionPhase.COMPLETED for e in diff_records)
    assert not [i for i in state.evidence if i.engine == "bugforge-compiler-diff"]

    # nothing is verified, and a revert contradicts the static candidate visibly
    assert all(i.attrs.get("verified", "false") == "false" for i in state.evidence)
    assert all(i.quality is not None for i in state.evidence)
    assert orch.report()["verified"] is False
    assert state.contradictions, "a reverted replay of a static candidate must stay visible"

    # minimize the planned sequence with a scripted replay evaluator, then qualify and report
    keep = sequence.calls[0].identity
    minimized = {
        sequence.sequence_id: minimize(
            sequence,
            lambda calls: any(c.identity == keep for c in calls),
            evaluator_name="scripted-replay",
        )
    }
    suite = run_suite(build_research_model(load_sources(request)))
    findings = build_findings(
        suite.candidates,
        manifest=mani,
        identity=identity,
        state=state,
        sequences=_all.sequences,
        minimized=minimized,
    )
    elevation = next(f for f in findings if f.detector == "caller_context.self_call_elevation")
    assert elevation.scope_status == "in_scope"
    assert elevation.sequence_ids and elevation.minimization
    assert elevation.verified is False and elevation.confirmed_severity == "unconfirmed"
    assert elevation.open_contradictions

    advisories = match_advisories(mani.compiler, load_sources(request))
    differential = run_differential(load_sources(request), backend=NoCompilerBackend())
    pack = build_report_pack(
        manifest=mani,
        identity=identity,
        findings=findings,
        sequences=_all.sequences,
        minimized=minimized,
        state=state,
        advisories=advisories,
        differential=differential,
        priority=prioritize(
            build_research_model(load_sources(request)), manifest=mani, candidates=suite.candidates
        ),
        tools={"solc": "unavailable", "forge": "unavailable", "bugforge-research": "available"},
    )
    assert pack.verified is False and pack.submitted is False
    text = pack.markdown
    assert mani.identity_digest() in text
    assert "ethereum/solidity" in text, "advisory provenance is in the report"
    assert "unavailable" in text and "Not verified" in text and "Not submitted" in text
    assert elevation.open_contradictions[0] in text


def test_out_of_scope_target_never_reaches_an_engine() -> None:
    mani, _ = _setup()
    identity = mani.to_campaign_identity(contract="BalanceAsDeposit", function="creditDeposit()")
    request = AnalysisRequest(
        repo_root=FIXTURES,
        language="solidity",
        target="BalanceAsDeposit.creditDeposit",
        contract="BalanceAsDeposit",
        function="creditDeposit()",
        source_file="accounting_vulnerable.sol",
        files=("accounting_vulnerable.sol",),
        campaign_id=identity.campaign_id,
        extra={"project_id": "proj", **mani.request_extra()},
    )
    spy = _static_engine()
    _orch, state = _run(mani, identity, request, [spy])
    assert state.stop_reason == "scope_blocked"
    assert spy.calls == []
    assert not [i for i in state.evidence if i.quality is EvidenceQuality.CANDIDATE]


def test_resume_cannot_raise_the_budget() -> None:
    mani, identity = _setup()
    request = _request(mani, identity)
    store = MemoryStore()
    first_scheduler = make_scheduler([BugforgeResearchEngine(mani)], max_engines=3, max_rounds=3)
    first = Orchestrator(
        first_scheduler, request, identity=identity, gate=BountyGate(mani), store=store
    )
    done = first.run()
    spent = done.budget.consumed.get("engines", 0)
    bigger = make_scheduler([BugforgeResearchEngine(mani)], max_engines=12, max_rounds=12)
    second = Orchestrator(bigger, request, identity=identity, gate=BountyGate(mani), store=store)
    second.start(explicit_resume=True)
    assert second.state.budget.limits["engines"] == 3
    assert second.state.budget.limits["rounds"] == 3
    assert spent <= 3
    again = second.run()
    assert again.budget.consumed.get("engines", 0) <= again.budget.limits["engines"]
    assert not any(e.phase is ExecutionPhase.FAILED for e in again.executions)


def test_campaign_run_is_deterministic() -> None:
    mani, identity = _setup()
    outcomes = []
    for _ in range(2):
        request = _request(mani, identity)
        _orch, state = _run(mani, identity, request, [_static_engine()])
        outcomes.append(
            (
                [d.selected_capability for d in state.decisions],
                sorted(i.evidence_id for i in state.evidence),
                state.stop_reason,
            )
        )
    assert outcomes[0] == outcomes[1]


# ---- report pack ----------------------------------------------------------------------------


def _pack(mani=None, **kwargs):
    mani, identity = (
        (mani, mani.to_campaign_identity(contract="SelfTrustMulticall")) if mani else _setup()
    )
    sources = load_sources(
        AnalysisRequest(
            repo_root=FIXTURES, language="solidity", files=(CALLER, "accounting_vulnerable.sol")
        )
    )
    model = build_research_model(sources)
    suite = run_suite(model)
    sequences = generate(
        model, suite.candidates, SequenceIdentity(campaign_id=identity.campaign_id)
    ).sequences
    findings = build_findings(
        suite.candidates, manifest=mani, identity=identity, sequences=sequences
    )
    return build_report_pack(
        manifest=mani, identity=identity, findings=findings, sequences=sequences, **kwargs
    ), findings


def test_report_pack_has_sixteen_sections_per_finding() -> None:
    pack, findings = _pack()
    assert len(SECTION_TITLES) == 16
    assert pack.reports and len(pack.reports) == min(len(findings), report_pack.MAX_FINDING_REPORTS)
    for item in pack.reports:
        assert [title for title, _ in item.sections] == list(SECTION_TITLES)
        assert all(body for _, body in item.sections)
        assert item.verified is False and item.submitted is False


def test_report_pack_is_deterministic() -> None:
    first, _ = _pack()
    second, _ = _pack()
    assert first.markdown == second.markdown
    assert first.pack_id == second.pack_id
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_report_never_claims_verification_or_submission() -> None:
    pack, _ = _pack()
    lowered = pack.markdown.lower()
    assert "not verified" in lowered and "not submitted" in lowered
    assert "confirmed severity" in lowered or "confirmed unconfirmed" in lowered
    assert "verified: true" not in lowered and "has been submitted" not in lowered
    assert pack.to_dict()["verified"] is False and pack.to_dict()["submitted"] is False


def test_known_issue_and_scope_wording_is_not_a_safety_claim() -> None:
    pack, findings = _pack()
    known = next(f for f in findings if f.known_issue)
    report = next(r for r in pack.reports if r.finding_id == known.finding_id)
    body = dict(report.sections)["Known-issue and duplicate status"]
    assert "KI-1" in body and "says nothing about whether the code is safe" in body
    scope = dict(report.sections)["Scope status"]
    assert scope.startswith("in_scope")


def test_missing_program_policy_never_invents_a_severity() -> None:
    bare = manifest(impact_categories=[], rules_version="unknown", poc_requirement="unknown")
    pack, findings = _pack(bare)
    assert {f.severity_candidate for f in findings} == {"unknown"}
    text = pack.markdown
    assert "no severity is proposed" in text
    assert "manifest_gaps" in text and "impact_categories" in text


def test_report_is_size_bounded() -> None:
    pack, _ = _pack()
    assert len(pack.markdown) <= report_pack.MAX_PACK_CHARS + 100
    for item in pack.reports:
        assert all(len(body) <= report_pack.MAX_SECTION_CHARS for _, body in item.sections)


def test_mutation_feedback_from_the_campaign_stays_bounded() -> None:
    mani, identity = _setup()
    _model, sequence, _ = _sequence_for(mani, identity)
    children = mutate([sequence], [FeedbackSignal("caller_context_mismatch", sequence.sequence_id)])
    assert all(len(c.calls) <= 4 for c in children)
