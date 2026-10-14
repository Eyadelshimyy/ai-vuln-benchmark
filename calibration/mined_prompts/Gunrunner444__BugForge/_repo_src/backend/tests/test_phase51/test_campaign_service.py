"""Phase 51 service layer: the operational bounty campaign reuses the Phase 49/50 stack."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.discovery.bounty.service import (
    APPROVABLE_CAPABILITIES,
    BountyCampaignService,
    CampaignError,
    CampaignSpec,
    UnknownCampaignError,
)
from tests.test_phase51.phase51_support import (
    CALLER,
    CONTRACT,
    FIXTURES,
    FUNCTION,
    manifest,
)


def _spec(mani=None, **overrides) -> CampaignSpec:
    base = dict(
        manifest=mani or manifest(),
        repo_root=FIXTURES,
        target=f"{CONTRACT}.multicall",
        contract=CONTRACT,
        function=FUNCTION,
        source_file=CALLER,
        files=(CALLER,),
        max_rounds=12,
    )
    base.update(overrides)
    return CampaignSpec(**base)  # type: ignore[arg-type]


def test_create_derives_identity_from_manifest_not_a_new_model() -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec())
    # identity binds to the manifest program context; no second identity system
    assert campaign.identity.program_context == campaign.manifest.identity_digest()
    assert campaign.campaign_id == campaign.identity.campaign_id
    report = svc.report(campaign.campaign_id)
    assert report["program_context"] == campaign.manifest.identity_digest()
    assert report["verified"] is False and report["llm_invoked"] is False


def test_analyze_exercises_the_bounty_capabilities_without_verifying() -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec())
    report = svc.analyze(campaign.campaign_id)
    exercised = set(report["exercised_capabilities"])
    for expected in {
        "bounty_context_analysis",
        "caller_context_analysis",
        "vfcs_generation",
        "compiler_advisory_analysis",
        "static_analysis",
    }:
        assert expected in exercised, exercised
    assert report["verified"] is False and report["submitted"] is False


def test_findings_evidence_and_report_pack_are_never_verified() -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec())
    svc.analyze(campaign.campaign_id)
    findings = svc.findings(campaign.campaign_id)["findings"]
    assert findings
    elevation = next(f for f in findings if f["detector"] == "caller_context.self_call_elevation")
    assert elevation["scope_status"] == "in_scope"
    assert elevation["confirmed_severity"] == "unconfirmed"
    assert elevation["verified"] is False
    evidence = svc.evidence(campaign.campaign_id)
    assert evidence["verified"] is False
    assert all(item["verified"] == "false" for item in evidence["evidence"])
    pack = svc.report_pack(campaign.campaign_id)
    assert pack.verified is False and pack.submitted is False
    assert "Not verified" in pack.markdown and "Not submitted" in pack.markdown


def test_repro_returns_plans_not_executions() -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec())
    svc.analyze(campaign.campaign_id)
    repro = svc.repro(campaign.campaign_id)
    assert repro["verified"] is False
    assert repro["sequences"], "static candidates yield call-sequence plans"
    assert "not executions" in repro["note"]


def test_out_of_scope_target_blocks_before_any_engine_runs() -> None:
    svc = BountyCampaignService()
    campaign = svc.create(
        _spec(
            target="BalanceAsDeposit.creditDeposit",
            contract="BalanceAsDeposit",
            function="creditDeposit()",
            source_file="accounting_vulnerable.sol",
            files=("accounting_vulnerable.sol",),
        )
    )
    report = svc.analyze(campaign.campaign_id)
    assert report["stop_reason"] == "scope_blocked"


def test_run_is_deterministic() -> None:
    # fresh services with independent databases, identical specs -> identical
    # observable outcome.
    import uuid

    from app.services.bounty_campaign_store import runner_for_url

    def _mem_runner():
        name = f"det_{uuid.uuid4().hex}"
        return runner_for_url(f"sqlite:///file:{name}?mode=memory&cache=shared&uri=true")

    a = BountyCampaignService(runner=_mem_runner())
    b = BountyCampaignService(runner=_mem_runner())
    ca = a.create(_spec())
    cb = b.create(_spec())
    ra = a.analyze(ca.campaign_id)
    rb = b.analyze(cb.campaign_id)
    assert ra["exercised_capabilities"] == rb["exercised_capabilities"]
    assert ra["stop_reason"] == rb["stop_reason"]
    fa = [f["finding_id"] for f in a.findings(ca.campaign_id)["findings"]]
    fb = [f["finding_id"] for f in b.findings(cb.campaign_id)["findings"]]
    assert fa == fb


def test_suggestion_cannot_widen_scope_or_budget() -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec())
    before = svc.report(campaign.campaign_id)["budget"]["limits"]
    svc.suggest(campaign.campaign_id, "fork_validation", reason="please")
    svc.analyze(campaign.campaign_id)
    after = svc.report(campaign.campaign_id)["budget"]["limits"]
    assert before == after
    # a fork was never approved, so no fork_validation evidence exists
    evidence = svc.evidence(campaign.campaign_id)["evidence"]
    assert not [e for e in evidence if e["capability"] == "fork_validation"]


def test_grant_approval_only_accepts_approvable_capabilities() -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec())
    assert "fork_validation" in APPROVABLE_CAPABILITIES
    with pytest.raises(CampaignError):
        svc.grant_approval(campaign.campaign_id, "static_analysis")
    approvals = svc.grant_approval(campaign.campaign_id, "fork_validation")
    assert "fork_validation" in approvals


def test_unknown_campaign_and_bad_repo_root() -> None:
    svc = BountyCampaignService()
    with pytest.raises(UnknownCampaignError):
        svc.report("cp_does_not_exist")
    with pytest.raises(CampaignError):
        svc.create(_spec(repo_root=Path("/nonexistent/path/xyz")))
