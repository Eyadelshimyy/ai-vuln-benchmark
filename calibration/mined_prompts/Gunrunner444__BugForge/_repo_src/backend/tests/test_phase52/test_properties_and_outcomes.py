"""Phase 52 hardening, Slice A: properties, oracles, structured outcomes (no forge needed)."""

from __future__ import annotations

from dataclasses import replace

from app.discovery.bounty.findings import execution_summary, is_strong
from app.discovery.bounty.properties import (
    OracleKind,
    OracleObservation,
    PropertyDeclaration,
    PropertyFamily,
    PropertyVerdict,
    build_property,
    evaluate,
)
from app.discovery.bounty.stateful import (
    IdentityContext,
    Outcome,
    StatefulExecutor,
    ToolStatus,
    _Base,
    abi_type,
    bind_identity,
    build_harness,
    classify,
    interpret,
    target_contract,
)
from app.discovery.bounty.vfcs import SequenceIdentity, Vfcs, VfcsCall, stamp_source_identity
from app.parsing.solidity_research import SemanticCandidate
from tests.test_phase52.phase52_support import (
    RAW_VAULT,
    VULNERABLE_INIT,
    deposit_sequence,
    event_log,
    forge_json,
    init_sequence,
    model_of,
)

TOOLS = ToolStatus("available", "available", "1.0.0", "0.8.30")


# ---- property construction ---------------------------------------------------------------------


def test_initializer_property_is_built_from_code_evidence_with_provenance() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    spec = build_property(init_sequence(), model_of(sources))
    assert spec.declaration is PropertyDeclaration.PROPERTY_UNDER_TEST
    assert spec.family is PropertyFamily.INITIALIZATION
    assert spec.oracle.kind is OracleKind.SECOND_CALL_MUST_FAIL
    assert spec.oracle.probe_role == "reinitialize"
    # the alternate formulation reads the owner the first call wrote
    assert [a.kind for a in spec.alternates] == [OracleKind.STATE_UNCHANGED]
    assert spec.alternates[0].getter == "owner()"
    prov = spec.provenance
    assert prov.file == "Acct.sol" and prov.contract == "Acct"
    assert prov.line_start > 0 and prov.line_end >= prov.line_start
    assert prov.variables == ("owner",)
    assert prov.candidate.startswith("aa.unprotected_account_initializer@")
    assert spec.status == "hypothesis" and spec.as_dict()["verified"] is False


def test_credit_property_has_a_solvency_alternate_and_fixture() -> None:
    spec = build_property(deposit_sequence(), model_of({"V.sol": RAW_VAULT}))
    assert spec.declaration is PropertyDeclaration.PROPERTY_UNDER_TEST
    assert spec.oracle.kind is OracleKind.CREDIT_LE_RECEIVED
    assert spec.oracle.token_fixture == "fee_on_transfer"
    assert spec.oracle.getter == "balances(address)"
    assert [a.kind for a in spec.alternates] == [OracleKind.CREDIT_LE_HOLDINGS]
    unchecked = build_property(
        deposit_sequence("accounting.unchecked_token_return"), model_of({"V.sol": RAW_VAULT})
    )
    assert unchecked.oracle.token_fixture == "false_return"


def test_unsupported_and_no_property_are_declared_not_guessed() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    model = model_of(sources)
    oracle_seq = replace(
        init_sequence(),
        template="oracle update→valuation→borrow",
        derived_from="oracle.stale_source_accepted@Acct.initialize(address)",
    )
    spec = build_property(oracle_seq, model)
    assert spec.declaration is PropertyDeclaration.PROPERTY_UNSUPPORTED
    assert "price" in spec.unsupported_reason
    assert not spec.executable
    boundary = replace(
        init_sequence(),
        template="boundary-batch",
        derived_from="arithmetic.batch_accumulation_overflow@Acct.initialize(address)",
    )
    spec = build_property(boundary, model)
    assert spec.declaration is PropertyDeclaration.NO_PROPERTY_AVAILABLE
    missing = replace(init_sequence(), derived_from="aa.x@Nope.f()")
    spec = build_property(missing, model)
    assert spec.declaration is PropertyDeclaration.PROPERTY_UNSUPPORTED
    assert spec.unsupported_reason == "candidate function unresolved"


# ---- oracle evaluation -------------------------------------------------------------------------


def test_evaluate_violated_held_and_unknown() -> None:
    seq = init_sequence()
    oracle = build_property(seq, model_of({"Acct.sol": VULNERABLE_INIT})).oracle
    assert (
        evaluate(seq, oracle, OracleObservation((1, 1))).verdict
        is PropertyVerdict.PROPERTY_VIOLATED
    )
    assert evaluate(seq, oracle, OracleObservation((1, 0))).verdict is PropertyVerdict.PROPERTY_HELD
    # the first initialization reverted: the property cannot be judged
    pre = evaluate(seq, oracle, OracleObservation((0, 1)))
    assert pre.verdict is PropertyVerdict.NOT_EVALUATED and not pre.precondition_ok
    # mutation reordered the calls: the precondition no longer precedes the probe
    swapped = replace(seq, calls=(seq.calls[1], seq.calls[0]))
    assert (
        evaluate(swapped, oracle, OracleObservation((1, 1))).verdict
        is PropertyVerdict.NOT_EVALUATED
    )
    # minimization dropped the probe
    alone = replace(seq, calls=(seq.calls[0],))
    assert evaluate(alone, oracle, OracleObservation((1,))).verdict is PropertyVerdict.NOT_EVALUATED


def test_state_and_credit_oracles() -> None:
    seq = init_sequence()
    alt = build_property(seq, model_of({"Acct.sol": VULNERABLE_INIT})).alternates[0]
    changed = OracleObservation((1, 1), before="0x01", after="0x02")
    assert evaluate(seq, alt, changed).verdict is PropertyVerdict.PROPERTY_VIOLATED
    same = OracleObservation((1, 1), before="0x01", after="0x01")
    assert evaluate(seq, alt, same).verdict is PropertyVerdict.PROPERTY_HELD
    unreadable = OracleObservation((1, 1), readable=False)
    assert evaluate(seq, alt, unreadable).verdict is PropertyVerdict.NOT_EVALUATED
    dep = deposit_sequence()
    spec = build_property(dep, model_of({"V.sol": RAW_VAULT}))
    over = OracleObservation((1,), received=99, credited=100, holdings_after=99, credit_after=100)
    assert evaluate(dep, spec.oracle, over).verdict is PropertyVerdict.PROPERTY_VIOLATED
    assert evaluate(dep, spec.alternates[0], over).verdict is PropertyVerdict.PROPERTY_VIOLATED
    fine = OracleObservation((1,), received=100, credited=100, holdings_after=100, credit_after=100)
    assert evaluate(dep, spec.oracle, fine).verdict is PropertyVerdict.PROPERTY_HELD
    reverted = OracleObservation((0,), received=0, credited=0)
    assert evaluate(dep, spec.oracle, reverted).verdict is PropertyVerdict.NOT_EVALUATED


def test_no_oracle_runs_are_execution_only_and_reverts_are_not_findings() -> None:
    seq = init_sequence()
    none = build_property(
        replace(seq, template="boundary-batch"), model_of({"Acct.sol": VULNERABLE_INIT})
    ).oracle
    outcome, evaluation, _ = classify(seq, none, OracleObservation((1, 1)))
    assert outcome is Outcome.SEQUENCE_EXECUTED_NO_ORACLE
    assert evaluation.verdict is PropertyVerdict.EXECUTION_ONLY_OBSERVATION
    outcome, _, reason = classify(seq, none, OracleObservation((1, 0)))
    assert outcome is Outcome.SEQUENCE_REVERTED and "not a finding" in reason


# ---- structured forge results ------------------------------------------------------------------


def _base(seq: Vfcs, sources: dict[str, str]) -> tuple[_Base, object]:
    model = model_of(sources)
    seq = stamp_source_identity(seq, sources)
    spec = build_property(seq, model)
    identity = bind_identity(seq, model, sources, IdentityContext(), property_id=spec.property_id)
    harness = build_harness(seq, model, oracle=spec.oracle, sources=sources)
    return _Base(seq, spec, spec.oracle, "default", identity, TOOLS), harness


def test_interpret_reads_structured_events_only_from_the_harness() -> None:
    seq = init_sequence()
    base, harness = _base(seq, {"Acct.sol": VULNERABLE_INIT})
    logs = [event_log(1, 0, 1), event_log(2, 0, 1), event_log(2, 1, 1), event_log(9, 2, 1)]
    obs = interpret(base, harness, 0, forge_json(logs), "")  # type: ignore[arg-type]
    assert obs.outcome is Outcome.PROPERTY_VIOLATED
    assert obs.result is not None and obs.result.call_results == (1, 1)
    assert obs.result.compiler_status == "ok" and obs.result.property_status == "violated"
    assert obs.strong_candidate and obs.as_dict()["verified"] is False
    # the analyzed contract emitting look-alike events cannot forge an observation
    forged = [event_log(k, i, v, address="0x" + "ab" * 20) for k, i, v in ((1, 0, 1), (9, 2, 1))]
    obs = interpret(base, harness, 0, forge_json(forged), "")  # type: ignore[arg-type]
    assert obs.outcome is Outcome.EXECUTION_FAILED
    assert obs.reason_code == "incomplete_run" and not obs.strong_candidate


def test_interpret_never_collapses_failures() -> None:
    seq = init_sequence()
    base, harness = _base(seq, {"Acct.sol": VULNERABLE_INIT})
    failed = interpret(base, harness, 1, forge_json([], "Failure", "revert"), "")  # type: ignore[arg-type]
    assert failed.outcome is Outcome.EXECUTION_FAILED
    assert "could not be deployed" in failed.reason
    missing = interpret(base, harness, 1, "not json at all", "boom")  # type: ignore[arg-type]
    assert missing.outcome is Outcome.EXECUTION_FAILED and missing.reason_code == "no_result"
    held = [event_log(1, 0, 1), event_log(2, 0, 1), event_log(2, 1, 0), event_log(9, 2, 1)]
    obs = interpret(base, harness, 0, forge_json(held), "")  # type: ignore[arg-type]
    assert obs.outcome is Outcome.PROPERTY_HELD and obs.reverted_index == 1
    pre = [event_log(1, 0, 1), event_log(2, 0, 0), event_log(2, 1, 1), event_log(9, 2, 1)]
    obs = interpret(base, harness, 0, forge_json(pre), "")  # type: ignore[arg-type]
    assert obs.outcome is Outcome.INCONCLUSIVE and obs.reason_code == "precondition_failed"


# ---- harness synthesis and identity ------------------------------------------------------------


def test_harness_imports_the_declaring_file_not_a_generic_first_file() -> None:
    sources = {
        "a_first.sol": "pragma solidity ^0.8.20;\ncontract Other {}\n",
        "z/Acct.sol": VULNERABLE_INIT,
    }
    model = model_of(sources)
    seq = init_sequence()
    harness = build_harness(seq, model, oracle=build_property(seq, model).oracle, sources=sources)
    assert harness.buildable and 'import "src/z/Acct.sol";' in harness.source
    # an expectation of a different file is an identity mismatch, never silently used
    wrong = build_harness(seq, model, sources=sources, source_file="a_first.sol")
    assert not wrong.buildable and wrong.reason_code == "identity_mismatch"


def test_target_contract_comes_from_the_candidate_not_a_primitive_call() -> None:
    seq = Vfcs(
        "vf_x",
        "approve→transferFrom",
        "template",
        (
            VfcsCall("token", "approve(address,uint256)", "approve", "victim", (), True, "why"),
            VfcsCall("Router", "execute(address,bytes)", "execute", "attacker"),
        ),
        "p",
        "caller_context.x@Router.execute(address,bytes)",
        SequenceIdentity(),
    )
    assert target_contract(seq) == "Router"


def test_abi_types_are_canonical_for_selectors() -> None:
    model = model_of({"V.sol": RAW_VAULT})
    assert abi_type(model, "IERC20") == "address"
    assert abi_type(model, "uint") == "uint256"
    assert abi_type(model, "bytes[] calldata") == "bytes[]"
    assert abi_type(model, "address payable") == "address"
    assert abi_type(model, "SomeStruct") is None
    harness = build_harness(deposit_sequence(), model, sources={"V.sol": RAW_VAULT})
    assert '"deposit(address,uint256)"' in harness.source


def test_unknown_constructor_arguments_are_never_invented() -> None:
    text = (
        "pragma solidity ^0.8.20;\ncontract Acct {\n    address public owner;\n"
        "    address public admin;\n    constructor(address a) { admin = a; }\n"
        "    function initialize(address newOwner) external { owner = newOwner; }\n}\n"
    )
    model = model_of({"Acct.sol": text})
    harness = build_harness(init_sequence(), model, sources={"Acct.sol": text})
    assert not harness.buildable and harness.reason_code == "unknown_constructor_argument"


def test_identity_mismatches_fail_closed_before_anything_runs() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    model = model_of(sources)
    seq = init_sequence(identity=SequenceIdentity(campaign_id="cp_a", deployment="1:0xabc"))
    executor = StatefulExecutor(tools=TOOLS)
    changed = IdentityContext(expected_hashes={"Acct.sol": "0" * 64})
    obs = executor.execute(seq, model, sources, context=changed)
    assert obs.outcome is Outcome.IDENTITY_MISMATCH and "changed" in obs.reason
    other = IdentityContext(campaign_id="cp_b")
    assert executor.execute(seq, model, sources, context=other).outcome is Outcome.IDENTITY_MISMATCH
    elsewhere = IdentityContext(deployment="1:0xdef")
    assert (
        executor.execute(seq, model, sources, context=elsewhere).outcome
        is Outcome.IDENTITY_MISMATCH
    )
    # the declaring file is missing from the source set
    obs = executor.execute(seq, model, {"Other.sol": "contract X {}"})
    assert obs.outcome is Outcome.IDENTITY_MISMATCH
    assert executor.runs == 0  # nothing reached forge


def test_unavailable_tools_stay_unavailable() -> None:
    sources = {"Acct.sol": VULNERABLE_INIT}
    executor = StatefulExecutor(tools=ToolStatus("unavailable", "available"))
    missing = executor.execute(init_sequence(), model_of(sources), sources)
    assert missing.outcome is Outcome.IDENTITY_MISMATCH
    assert "not recorded" in missing.reason
    assert executor.runs == 0
    seq = stamp_source_identity(init_sequence(), sources)
    obs = executor.execute(seq, model_of(sources), sources)
    assert obs.outcome is Outcome.UNAVAILABLE and obs.reason_code == "tool_missing"
    assert not obs.executed and executor.runs == 0


# ---- findings: execution never upgrades without an oracle --------------------------------------


def _candidate() -> SemanticCandidate:
    return SemanticCandidate(
        detector="aa.unprotected_account_initializer",
        family="aa",
        title="t",
        contract="Acct",
        function="initialize(address)",
        file="Acct.sol",
        line=6,
        summary="s",
    )


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "derived_from": "aa.unprotected_account_initializer@Acct.initialize(address)",
        "outcome": "property_violated",
        "declaration": "property_under_test",
        "verdict": "property_violated",
        "oracle_kind": "second_call_must_fail",
        "identity_status": "bound",
        "corroboration": "single_path",
        "property_id": "pr_1",
        "bundle_id": "rb_1",
    }
    base.update(overrides)
    return base


def test_no_oracle_execution_cannot_become_a_strong_candidate() -> None:
    forged = _record(
        outcome="sequence_executed_no_oracle",
        verdict="execution_only_observation",
        oracle_kind="none",
        strong_candidate=True,  # a stored flag alone is never trusted
    )
    assert not is_strong(forged)
    summary = execution_summary(_candidate(), (forged,))
    assert summary.strength == "weak_execution_only"
    assert summary.status == "sequence_executed_no_oracle"


def test_mechanism_and_identity_mismatch_are_not_violations() -> None:
    mechanism = _record(
        outcome="mechanism_observed",
        declaration="mechanism_observation",
        verdict="mechanism_observed",
        oracle_kind="mechanism_probe",
    )
    assert not is_strong(mechanism)
    summary = execution_summary(_candidate(), (mechanism,))
    assert summary.status == "mechanism_observed"
    assert summary.strength == "mechanism_observation"
    assert any("unproven" in note for note in summary.uncertainties)
    mismatch = _record(
        outcome="identity_mismatch",
        verdict="not_evaluated",
        identity_status="identity_mismatch",
        oracle_kind="second_call_must_fail",
    )
    assert not is_strong(mismatch)
    mismatched = execution_summary(_candidate(), (mismatch,))
    assert mismatched.status == "identity_mismatch"
    assert mismatched.strength == "static_candidate"
    held = _record(outcome="property_held", verdict="property_held")
    assert execution_summary(_candidate(), (held,)).status == "property_held"
    assert not is_strong(held)


def test_strength_requires_bound_fresh_oracle_violation() -> None:
    assert execution_summary(_candidate(), (_record(),)).strength == "strong_candidate"
    corroborated = _record(corroboration="corroborated_candidate")
    assert execution_summary(_candidate(), (corroborated,)).strength == "corroborated_candidate"
    assert not is_strong(_record(identity_status="identity_mismatch"))
    assert not is_strong(_record(stale=True))
    assert not is_strong(_record(declaration="no_property_available"))
    disagreement = execution_summary(_candidate(), (_record(corroboration="disagreement"),))
    assert disagreement.corroboration == "disagreement"
    assert any("disagrees" in note for note in disagreement.uncertainties)
    assert execution_summary(_candidate(), ()).strength == "static_candidate"
