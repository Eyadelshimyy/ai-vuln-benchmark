"""Property evaluation of the pinned CometWithExtendedAssetList source.

This is not a mock with the same name. The test reads the local checkout named by
BUGFORGE_COMET_ROOT (default /tmp/comet) and requires commit f766f515. CI skips it
when that tree is absent. BUGFORGE_REQUIRE_COMET=1 turns that skip into a failure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.constructor_profile import (
    COMET_CONTRACT,
    COMET_FILE,
    COMET_PIN,
    PAUSE_SIGNATURE,
    comet_pause_profile,
    prepare_profile,
)
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.solc_resolve import resolve_solc
from tests.test_phase51.phase51_support import manifest
from tests.test_phase52.phase52_support import requires_forge
from tests.test_phase58.comet_support import CLOSURE, comet_root, comet_sources


def _sources() -> dict[str, str]:
    return comet_sources(comet_root())


def test_comet_configuration_profile_matches_the_pinned_source() -> None:
    sources = _sources()
    target = sources[COMET_FILE]
    assert "if (msg.sender != governor && msg.sender != pauseGuardian) revert Unauthorized();" in (
        target
    )
    assert f"function {PAUSE_SIGNATURE.split('(')[0]}(" in target
    profile = comet_pause_profile(sources)
    assert profile.contract == COMET_CONTRACT
    assert profile.source_sha256 == hashlib.sha256(target.encode()).hexdigest()
    prepared = prepare_profile(profile, sources)
    again = prepare_profile(comet_pause_profile(sources), sources)
    assert prepared.expression == again.expression
    for name in (
        "governor",
        "pauseGuardian",
        "baseToken",
        "baseTokenPriceFeed",
        "extensionDelegate",
        "baseMinForRewards",
        "storeFrontPriceFactor",
        "assetConfigs",
    ):
        assert f"{name}:" in prepared.expression
    assert "address(0xC0FE)" in prepared.expression
    assert "address(0x6A2D)" in prepared.expression
    assert "address(bf_baseToken)" in prepared.expression
    assert "address(bf_baseFeed)" in prepared.expression
    assert "address(bf_holder)" in prepared.expression
    assert "baseMinForRewards: 1" in prepared.expression
    assert "storeFrontPriceFactor: 0" in prepared.expression
    assert "new CometConfiguration.AssetConfig[](0)" in prepared.expression
    assert "d = 6" in prepared.contracts
    assert "d = 8" in prepared.contracts
    assert profile.public_dict()["local_environment"] is True
    assert profile.public_dict()["establishes_deployed_state"] is False
    changed = dict(sources)
    changed[COMET_FILE] = target + "\n"
    with pytest.raises(Exception, match="profile_source_mismatch"):
        prepare_profile(profile, changed)
    # No profile means the real constructor struct is still not invented.
    from app.discovery.bounty.properties import build_property
    from app.discovery.bounty.stateful import build_harness
    from app.discovery.bounty.vfcs import ATTACKER, SequenceIdentity, Vfcs, VfcsCall
    from app.parsing.solidity_research import build_research_model

    model = build_research_model(sources)
    call = VfcsCall(COMET_CONTRACT, PAUSE_SIGNATURE, "execute", ATTACKER)
    sequence = Vfcs(
        sequence_id="vf_comet_bare",
        template="unauthorized-call",
        origin="test",
        calls=(call,),
        property_under_test=profile.property_statement,
        derived_from=f"profile.authorization@{COMET_CONTRACT}.{PAUSE_SIGNATURE}",
        identity=SequenceIdentity(),
    )
    bare = build_harness(
        sequence, model, oracle=build_property(sequence, model).oracle, sources=sources
    )
    assert bare.reason_code == "unknown_constructor_argument"


@requires_forge
def test_comet_pause_campaign_evaluates_the_authorization_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if resolve_solc("0.8.15") is None:
        if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
            pytest.fail("the Foundry job requires solc 0.8.15")
        pytest.skip("solc 0.8.15 is not installed")
    sources = _sources()
    repo = comet_root()
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{tmp_path / 'campaign.sqlite'}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    profile = comet_pause_profile(sources)
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=manifest(
                compiler={
                    "version": "0.8.15",
                    "optimizer": "enabled",
                    "optimizer_runs": "1",
                    "via_ir": "true",
                    "evm_version": "cancun",
                },
                in_scope=[{"kind": "contract", "identifier": COMET_CONTRACT}],
                out_of_scope=[],
                known_issues=[],
                source_commit=COMET_PIN,
            ),
            repo_root=repo,
            contract=COMET_CONTRACT,
            source_file=COMET_FILE,
            files=CLOSURE,
            constructor_profile=profile.to_dict(),
        )
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    summary = result["campaign_summary"]
    print(
        json.dumps(
            {
                "campaign_id": campaign.campaign_id,
                "sequences": summary.get("sequences"),
                "sequences_submitted": summary.get("sequences_submitted"),
                "sequences_excluded": summary.get("sequences_excluded"),
                "sequences_skipped": summary.get("sequences_skipped"),
                "forge_runs": summary.get("forge_runs"),
                "by_outcome": summary.get("by_outcome"),
                "property_evaluation_demonstrated": summary.get("property_evaluation_demonstrated"),
                "foundry_invoked": summary.get("foundry_invoked"),
                "blocker": summary.get("blocker"),
                "verified": summary.get("verified"),
                "profile": (summary.get("constructor_profile") or {}).get("validation"),
            }
        )
    )
    assert summary["constructor_profile"]["validation"] == "accepted"
    assert summary["constructor_profile"]["contract"] == COMET_CONTRACT
    assert summary["constructor_profile"]["source_sha256"] == profile.source_sha256
    assert summary["constructor_profile"]["local_environment"] is True
    assert summary["constructor_profile"]["establishes_deployed_state"] is False
    assert summary["sequences_submitted"] >= 1
    assert summary["forge_runs"] >= 1
    assert summary["foundry_invoked"] is True
    assert summary["property_evaluation_demonstrated"] is True
    assert summary["by_outcome"].get("property_held", 0) >= 1
    assert summary["verified"] is False
    held = [
        item
        for item in result["observations"]
        if item["outcome"] == "property_held" and item["oracle_kind"] == "call_must_fail"
    ]
    assert held, result["outcomes"]
    assert held[0]["verdict"] == "property_held"
    assert held[0]["strong_candidate"] is False
    assert "probe call reverted" in held[0]["reason"]
    props = [
        spec
        for spec in result["properties"].values()
        if spec["provenance"]["contract"] == COMET_CONTRACT
        and spec["oracle"].get("probe_function") == PAUSE_SIGNATURE
    ]
    assert props
    assert props[0]["oracle"]["getter"] == "isSupplyPaused()"
    assert props[0]["oracle"]["kind"] == "call_must_fail"
    assert props[0]["verified"] is False
    assert "governor" in props[0]["statement"]
    runs = BountyCampaignService().stateful_status(campaign.campaign_id)["runs"]
    tools = runs[-1]["tools"]
    assert str(tools["solc_selected_version"]).startswith("0.8.15")
    assert tools["evm_requested"] == "cancun"
    assert tools["evm_exact"] == "false"
    assert tools["evm_version"] == ""
    assert tools["optimizer"] == "true"
    assert tools["via_ir"] == "true"
    assert tools["optimizer_runs"] == "1"
    skipped = result["skipped"]
    assert all(item.get("forge_invoked") == "false" for item in skipped)
    reloaded = BountyCampaignService().research_ledger(campaign.campaign_id)["campaign_summary"]
    assert reloaded["forge_runs"] == result["forge_runs"]
    assert reloaded["property_evaluation_demonstrated"] is True
    assert reloaded["by_outcome"].get("property_held", 0) >= 1
    assert reloaded["constructor_profile"]["source_sha256"] == profile.source_sha256
    assert reloaded["verified"] is False
    assert reloaded["constructor_profile"]["establishes_deployed_state"] is False
    findings = BountyCampaignService().findings(campaign.campaign_id)["findings"]
    pause = [item for item in findings if str(item.get("function", "")).startswith("pause(")]
    assert pause
    assert pause[0]["execution_status"] == "property_held"
    assert pause[0]["candidate_strength"] == "property_held"
    assert pause[0]["verified"] is False
    assert all(item["verified"] is False for item in findings)
    assert all(item["candidate_strength"] != "strong_candidate" for item in findings)
