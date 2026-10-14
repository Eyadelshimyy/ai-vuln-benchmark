from __future__ import annotations

from dataclasses import replace

from app.discovery.capabilities import ResultStatus
from app.discovery.orchestration.evidence import (
    attribute,
    compute_delta,
    corroborate,
    detect_contradictions,
    identity_matches,
    items_from_result,
    negative_from,
    outcome_of,
    polarity_of,
    quality_of,
)
from app.discovery.orchestration.model import (
    CampaignIdentity,
    EvidenceQuality,
    ExecutionPhase,
    FailureClass,
)
from app.discovery.results import DynamicResult
from tests.test_phase49.phase49_runtime import runtime_result
from tests.test_phase49.phase49_support import finding, make_request

IDENTITY = CampaignIdentity(
    "c",
    source_snapshot="snap1",
    compiler_configuration="solc-0.8.24",
    contract="Vault",
    function="withdraw",
)


def _result(**overrides: object) -> DynamicResult:
    values: dict[str, object] = {
        "engine": "e",
        "language": "solidity",
        "target": "t",
        "status": ResultStatus.EXECUTED,
        "executed": True,
    }
    values.update(overrides)
    return DynamicResult(**values)  # type: ignore[arg-type]


def test_outcome_mapping_is_truthful_about_what_ran() -> None:
    cases = {
        ResultStatus.UNAVAILABLE: ExecutionPhase.UNAVAILABLE,
        ResultStatus.NOT_IMPLEMENTED: ExecutionPhase.UNSUPPORTED,
        ResultStatus.UNSUPPORTED: ExecutionPhase.UNSUPPORTED,
        ResultStatus.PLANNED: ExecutionPhase.UNSUPPORTED,
        ResultStatus.TIMEOUT: ExecutionPhase.TIMED_OUT,
        ResultStatus.TOOL_FAILURE: ExecutionPhase.FAILED,
        ResultStatus.FAILED: ExecutionPhase.FAILED,
        ResultStatus.EXECUTED: ExecutionPhase.COMPLETED,
        ResultStatus.INGESTED: ExecutionPhase.COMPLETED,
        ResultStatus.INTERESTING: ExecutionPhase.COMPLETED,
    }
    for status, phase in cases.items():
        assert outcome_of(_result(status=status, executed=True))[0] is phase
    assert outcome_of(_result(status=ResultStatus.EXECUTED, executed=False))[0] is (
        ExecutionPhase.UNKNOWN
    )
    assert outcome_of(_result(status=ResultStatus.FAILED, executed=False))[1] is (
        FailureClass.INFRASTRUCTURE
    )
    assert outcome_of(_result(provenance="budget_exhausted", executed=False))[0] is (
        ExecutionPhase.BLOCKED
    )


def test_a_failed_test_with_an_assertion_is_an_observation_not_a_tool_failure() -> None:
    result = _result(status=ResultStatus.FAILED, assertion="invariant broke")
    assert outcome_of(result)[0] is ExecutionPhase.COMPLETED
    assert quality_of(result, ExecutionPhase.COMPLETED) is EvidenceQuality.CANDIDATE


def test_quality_levels() -> None:
    clean = _result()
    assert quality_of(clean, ExecutionPhase.COMPLETED) is EvidenceQuality.OBSERVATION
    assert quality_of(_result(findings=(finding(),)), ExecutionPhase.COMPLETED) is (
        EvidenceQuality.CANDIDATE
    )
    assert quality_of(clean, ExecutionPhase.TIMED_OUT) is EvidenceQuality.INCOMPLETE
    assert quality_of(clean, ExecutionPhase.UNAVAILABLE) is EvidenceQuality.UNAVAILABLE
    assert quality_of(clean, ExecutionPhase.UNSUPPORTED) is EvidenceQuality.UNSUPPORTED
    unknown = _result(metadata={"observation_status": "unknown"})
    assert quality_of(unknown, ExecutionPhase.COMPLETED) is EvidenceQuality.UNKNOWN
    unattributed = _result(metadata={"document_attributed": "false"})
    assert quality_of(unattributed, ExecutionPhase.COMPLETED) is EvidenceQuality.UNKNOWN


def test_polarity() -> None:
    assert polarity_of(_result(findings=(finding(),)), ExecutionPhase.COMPLETED) == "positive"
    assert polarity_of(_result(), ExecutionPhase.COMPLETED) == "neutral"
    assert polarity_of(_result(), ExecutionPhase.FAILED) == ""
    reverted = _result(metadata={"transaction_outcome": "reverted"})
    assert polarity_of(reverted, ExecutionPhase.COMPLETED) == "negative"
    divergent = _result(metadata={"classification": "deterministic divergence"})
    assert polarity_of(divergent, ExecutionPhase.COMPLETED) == "divergent"


def test_attribution_requires_contract_and_function_and_respects_overloads() -> None:
    assert attribute("", "withdraw", IDENTITY) == ""
    assert attribute("Vault", "", IDENTITY) == ""
    assert attribute("Vault", "withdraw", IDENTITY) == "Vault.withdraw"
    assert attribute("", "Vault.sol::Vault.withdraw(uint256)", IDENTITY) == "Vault.withdraw"
    assert attribute("Vault", "deposit", IDENTITY) == "Vault.deposit"
    typed = replace(IDENTITY, function="withdraw(uint256)")
    assert attribute("Vault", "withdraw", typed) == ""
    assert attribute("Vault", "withdraw(address)", typed) == ""
    assert attribute("Vault", "withdraw(uint256)", typed) == "Vault.withdraw(uint256)"


def test_a_mismatched_snapshot_is_not_attributed_to_the_campaign() -> None:
    other = _result(metadata={"source_snapshot": "other"})
    assert identity_matches(other, IDENTITY) is False
    assert identity_matches(_result(metadata={"source_snapshot": "snap1"}), IDENTITY)
    assert identity_matches(_result(), IDENTITY)
    items = items_from_result(
        _result(findings=(finding(),), metadata={"source_snapshot": "other"}),
        identity=IDENTITY,
        capability="static_analysis",
        execution_key="k",
        phase=ExecutionPhase.COMPLETED,
    )
    assert [i.quality for i in items] == [EvidenceQuality.UNKNOWN]
    assert items[0].attrs["identity"] == "mismatch"
    assert items[0].polarity == ""


def test_evidence_ids_are_content_based_and_bounded() -> None:
    many = tuple(finding(detector=f"d{n}", function=f"f{n}") for n in range(60))
    result = _result(findings=many)
    first = items_from_result(
        result,
        identity=IDENTITY,
        capability="fuzzing",
        execution_key="k1",
        phase=ExecutionPhase.COMPLETED,
    )
    second = items_from_result(
        result,
        identity=IDENTITY,
        capability="fuzzing",
        execution_key="k2",
        phase=ExecutionPhase.COMPLETED,
    )
    assert [i.evidence_id for i in first] == [i.evidence_id for i in second]
    assert len(first) <= 32
    assert len({i.evidence_id for i in first}) == len(first)


def test_raw_tool_output_is_never_copied_into_evidence() -> None:
    result = _result(stdout="SECRET-OUTPUT " * 50, stderr="stderr text", findings=(finding(),))
    items = items_from_result(
        result,
        identity=IDENTITY,
        capability="fuzzing",
        execution_key="k",
        phase=ExecutionPhase.COMPLETED,
    )
    assert "SECRET-OUTPUT" not in repr(items)
    assert "stderr text" not in repr(items)


def _finding_item(engine: str, capability: str, *, function: str = "withdraw", family="reentrancy"):
    result = _result(
        engine=engine,
        findings=(finding(detector=f"{family}-eth", function=function),),
    )
    items = items_from_result(
        result,
        identity=IDENTITY,
        capability=capability,
        execution_key=engine,
        phase=ExecutionPhase.COMPLETED,
    )
    return next(i for i in items if i.kind == "finding")


def test_corroboration_needs_distinct_engines_and_distinct_capabilities() -> None:
    a = _finding_item("slither", "static_analysis")
    b = _finding_item("echidna", "fuzzing")
    same_capability = _finding_item("bugforge-static", "static_analysis")
    pair = corroborate((a, b))
    assert {i.quality for i in pair} == {EvidenceQuality.CORROBORATED}
    assert {i.quality for i in corroborate((a, same_capability))} == {EvidenceQuality.CANDIDATE}
    assert {i.quality for i in corroborate((a, replace(b, engine="slither")))} == {
        EvidenceQuality.CANDIDATE
    }
    assert corroborate(corroborate((a, b))) == corroborate((a, b))


def test_a_missing_identity_or_other_function_never_corroborates() -> None:
    a = _finding_item("slither", "static_analysis")
    other_function = _finding_item("echidna", "fuzzing", function="deposit")
    anonymous = replace(_finding_item("medusa", "fuzzing"), identity_key="")
    assert {i.quality for i in corroborate((a, other_function))} == {EvidenceQuality.CANDIDATE}
    assert {i.quality for i in corroborate((a, anonymous))} == {EvidenceQuality.CANDIDATE}
    other_family = _finding_item("echidna", "fuzzing", family="delegatecall")
    assert {i.quality for i in corroborate((a, other_family))} == {EvidenceQuality.CANDIDATE}


def test_candidate_not_reproduced_is_a_contradiction_neither_side_is_chosen() -> None:
    request = make_request(__import__("pathlib").Path("."))
    positive = _finding_item("slither", "static_analysis")
    reverted = items_from_result(
        runtime_result(request, mode="local", outcome="reverted"),
        identity=IDENTITY,
        capability="runtime_validation",
        execution_key="rt",
        phase=ExecutionPhase.COMPLETED,
    )[0]
    assert reverted.polarity == "negative"
    found = detect_contradictions((positive, reverted), frozenset())
    assert [c.kind for c in found] == ["candidate_not_reproduced"]
    contradiction = found[0]
    assert contradiction.status == "open"
    assert set(contradiction.evidence_ids) == {positive.evidence_id, reverted.evidence_id}
    assert "runtime_validation" not in contradiction.discriminators
    assert "differential_validation" in contradiction.discriminators
    exercised = detect_contradictions((positive, reverted), frozenset({"differential_validation"}))
    assert exercised[0].status == "unresolved"
    assert exercised[0].contradiction_id == contradiction.contradiction_id


def test_evidence_for_different_identities_never_conflicts() -> None:
    positive = _finding_item("slither", "static_analysis", function="deposit")
    request = make_request(__import__("pathlib").Path("."))
    reverted = items_from_result(
        runtime_result(request, mode="local", outcome="reverted"),
        identity=IDENTITY,
        capability="runtime_validation",
        execution_key="rt",
        phase=ExecutionPhase.COMPLETED,
    )[0]
    assert detect_contradictions((positive, reverted), frozenset()) == ()


def test_negative_evidence_never_claims_safety_and_needs_a_completed_run() -> None:
    clean = _result()
    negative = negative_from(
        clean, identity=IDENTITY, capability="fuzzing", phase=ExecutionPhase.COMPLETED
    )
    assert negative is not None and negative.proves_safety is False
    assert (
        negative_from(
            _result(findings=(finding(),)),
            identity=IDENTITY,
            capability="fuzzing",
            phase=ExecutionPhase.COMPLETED,
        )
        is None
    )
    for phase in (ExecutionPhase.TIMED_OUT, ExecutionPhase.UNAVAILABLE, ExecutionPhase.FAILED):
        assert negative_from(clean, identity=IDENTITY, capability="fuzzing", phase=phase) is None
    mismatched = _result(metadata={"source_snapshot": "other"})
    assert (
        negative_from(
            mismatched, identity=IDENTITY, capability="fuzzing", phase=ExecutionPhase.COMPLETED
        )
        is None
    )


def test_delta_reports_only_what_is_new() -> None:
    first = _finding_item("slither", "static_analysis")
    second = _finding_item("echidna", "fuzzing")
    delta = compute_delta(
        (first,),
        (first, second),
        new_seeds=("crash:abc",),
        new_coverage=False,
        contradictions_before=(),
        contradictions_after=(),
        negatives_added=(),
        uncertainty_before=("unexercised_candidate", "baseline"),
        uncertainty_after=("unexercised_candidate", "reachability"),
    )
    assert delta.new_evidence_ids == (second.evidence_id,)
    assert delta.new_finding_ids == (second.evidence_id,)
    assert delta.new_corpus_seeds == ("crash:abc",)
    assert delta.uncertainty_added == ("reachability",)
    assert delta.uncertainty_removed == ("baseline",)
    assert delta.informative
    nothing = compute_delta(
        (first,),
        (first,),
        new_seeds=(),
        new_coverage=False,
        contradictions_before=(),
        contradictions_after=(),
        negatives_added=("x",),
        uncertainty_before=("a",),
        uncertainty_after=(),
    )
    assert nothing.informative is False
    assert nothing.uncertainty_removed == ("a",)
