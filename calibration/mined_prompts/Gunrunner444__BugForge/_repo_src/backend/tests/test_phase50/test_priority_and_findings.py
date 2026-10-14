"""Program-aware triage, scope, known-issue suppression, and finding qualification."""

from __future__ import annotations

from app.discovery.bounty.campaign import BountyManifest, PocRequirement
from app.discovery.bounty.findings import build_findings, match_known_issue, severity_candidate
from app.discovery.bounty.priority import DISCLAIMER, prioritize, signals_of
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase50.phase50_support import FIXTURES, manifest, read

NAMES = ("caller_context_vulnerable.sol", "accounting_vulnerable.sol")


def _model():
    model = build_research_model({n: read(n) for n in NAMES})
    return model, run_suite(model)


def test_triage_is_not_an_exploitability_claim() -> None:
    model, suite = _model()
    result = prioritize(model, manifest=manifest(), candidates=suite.candidates)
    assert result.disclaimer == DISCLAIMER
    assert "exploitab" in DISCLAIMER and "not" in DISCLAIMER
    assert all(entry.disclaimer == DISCLAIMER for entry in result.ranked)


def test_signals_carry_evidence_and_ignore_function_names() -> None:
    source = """
    pragma solidity ^0.8.20;
    interface IERC20 { function transfer(address, uint256) external returns (bool);
        function balanceOf(address) external view returns (uint256); }
    contract A {
        IERC20 public t;
        function harmless(address to, uint256 amount) external {
            require(t.transfer(to, amount), "x");
        }
        function withdrawAllFunds() external view returns (uint256) { return 1; }
    }
    """
    model = build_research_model({"A.sol": source})
    sig = {f.name: {s.name for s in signals_of(model, f)} for f in model.all_functions()}
    assert "value_transfer" in sig["harmless"]
    assert "value_transfer" not in sig["withdrawAllFunds"]
    for signal in signals_of(model, model.function("A", "harmless")):  # type: ignore[arg-type]
        assert signal.evidence


def test_program_policy_drives_the_order_and_no_rewards_are_hardcoded() -> None:
    model, suite = _model()
    theft = manifest()
    result = prioritize(model, manifest=theft, candidates=suite.candidates)
    assert result.policy == "program"
    top = result.ranked[0]
    assert top.categories, "the top entry matches a program impact category"
    permissions_only = manifest(
        impact_categories=[
            {"name": "Takeover", "severity": "high", "weight": 95, "tags": ["router_dispatch"]}
        ]
    )
    other = prioritize(model, manifest=permissions_only, candidates=suite.candidates)
    assert [e.identity for e in other.ranked[:3]] != [e.identity for e in result.ranked[:3]]
    assert all(entry.score <= 100 + 30 for entry in result.ranked)


def test_without_a_policy_signals_weigh_equally_and_severity_is_unknown() -> None:
    model, suite = _model()
    result = prioritize(model, manifest=manifest(impact_categories=[]), candidates=suite.candidates)
    assert result.policy == "uniform"
    for entry in result.ranked:
        assert dict(entry.components).get("program_impact") is None
        assert "no impact policy" in entry.explanation or "signals" in entry.explanation
    severity, why = severity_candidate((), suite.candidates[0])
    assert severity == "unknown" and "no impact categories" in why


def test_out_of_scope_assets_are_excluded_from_ranking_but_recorded() -> None:
    model, suite = _model()
    result = prioritize(model, manifest=manifest(), candidates=suite.candidates)
    assert all(e.contract != "BalanceAsDeposit" for e in result.ranked)
    assert any(item.startswith("BalanceAsDeposit.") for item in result.excluded_out_of_scope)


def test_poc_feasible_candidates_rank_higher_when_the_program_requires_a_poc() -> None:
    model, suite = _model()
    required = prioritize(
        model, manifest=manifest(poc_requirement="required"), candidates=suite.candidates
    )
    optional = prioritize(
        model, manifest=manifest(poc_requirement="not_required"), candidates=suite.candidates
    )
    req = {e.identity: e.score for e in required.ranked}
    opt = {e.identity: e.score for e in optional.ranked}
    common = set(req) & set(opt)
    assert common
    assert any(req[i] > opt[i] for i in common)
    assert all(req[i] >= opt[i] for i in common)


def test_ranking_is_deterministic_and_bounded() -> None:
    model, suite = _model()
    a = prioritize(model, manifest=manifest(), candidates=suite.candidates)
    b = prioritize(model, manifest=manifest(), candidates=reversed(suite.candidates))
    assert [e.identity for e in a.ranked] == [e.identity for e in b.ranked]
    assert len(a.ranked) <= 64


# ---- findings -------------------------------------------------------------------------------


def _findings(mani: BountyManifest | None = None):
    mani = mani or manifest()
    model, suite = _model()
    identity = mani.to_campaign_identity(contract="SelfTrustMulticall")
    return build_findings(suite.candidates, manifest=mani, identity=identity), mani


def test_known_issue_is_suppressed_but_not_called_safe() -> None:
    findings, _ = _findings()
    donation = [f for f in findings if f.detector == "accounting.donation_share_price"]
    assert donation and donation[0].known_issue is not None
    assert donation[0].qualification.status == "known_issue"
    assert "not a safety claim" in donation[0].qualification.reasons[0]
    assert donation[0].verified is False


def test_out_of_scope_is_not_safe_and_scope_unknown_is_not_in_scope() -> None:
    findings, _ = _findings()
    balance = [f for f in findings if f.contract == "BalanceAsDeposit"]
    assert balance and all(f.qualification.status == "out_of_scope" for f in balance)
    assert all("not a safety claim" in f.qualification.reasons[0] for f in balance)
    unknown = [f for f in findings if f.scope_status == "unknown"]
    assert unknown and all(f.qualification.status == "needs_scope" for f in unknown)


def test_in_scope_candidate_needs_a_poc_when_the_program_requires_one() -> None:
    findings, _ = _findings()
    elevation = next(f for f in findings if f.detector == "caller_context.self_call_elevation")
    assert elevation.scope_status == "in_scope"
    assert elevation.qualification.status == "needs_poc"
    relaxed, _ = _findings(manifest(poc_requirement="not_required"))
    again = next(f for f in relaxed if f.detector == "caller_context.self_call_elevation")
    assert again.qualification.status == "report_candidate"


def test_severity_is_a_candidate_from_program_policy_only() -> None:
    findings, _ = _findings()
    theft = next(f for f in findings if f.detector.startswith("caller_context.unrestricted"))
    assert theft.severity_candidate in {"critical", "high", "unknown"}
    assert theft.confirmed_severity == "unconfirmed"
    assert "upper bound" in theft.severity_rationale or theft.severity_candidate == "unknown"
    bare, _ = _findings(manifest(impact_categories=[]))
    assert {f.severity_candidate for f in bare} == {"unknown"}


def test_findings_keep_identity_and_never_claim_verification_or_submission() -> None:
    findings, mani = _findings()
    for item in findings:
        ident = dict(item.identity)
        assert ident["program_context"] == mani.identity_digest()
        assert ident["source_snapshot"] == "abc1234"
        assert item.verified is False and item.submitted is False


def test_duplicate_root_cause_is_flagged_not_dropped() -> None:
    mani = manifest(in_scope=[{"kind": "path", "identifier": "caller_context_vulnerable.sol"}])
    model = build_research_model(
        {"caller_context_vulnerable.sol": read("caller_context_vulnerable.sol")}
    )
    suite = run_suite(model)
    doubled = list(suite.candidates) + list(suite.candidates)
    findings = build_findings(doubled, manifest=mani, identity=mani.to_campaign_identity())
    assert len(findings) == len(doubled)
    assert any(f.duplicate_of for f in findings)
    flagged = [f for f in findings if f.duplicate_of and f.qualification.status != "known_issue"]
    assert flagged
    assert any("not safe" in reason for f in flagged for reason in f.qualification.reasons)


def test_known_issue_matching_needs_every_declared_dimension() -> None:
    _model_, suite = _model()
    item = next(c for c in suite.candidates if c.detector == "accounting.donation_share_price")
    mani = manifest()
    assert match_known_issue(mani.known_issues, item) is not None
    other = manifest(
        known_issues=[
            {
                "issue_id": "KI-2",
                "title": "t",
                "source": "audit",
                "contracts": ["DonationVault"],
                "detectors": ["oracle.insufficient_quorum"],
            }
        ]
    )
    assert match_known_issue(other.known_issues, item) is None


def test_unknown_poc_requirement_is_reported_not_assumed() -> None:
    mani = manifest(poc_requirement="unknown")
    findings, _ = _findings(mani)
    scoped = next(f for f in findings if f.scope_status == "in_scope" and f.known_issue is None)
    assert any("proof-of-concept requirement is unknown" in r for r in scoped.qualification.reasons)
    assert PocRequirement.UNKNOWN is mani.poc_requirement
    assert FIXTURES.is_dir()
