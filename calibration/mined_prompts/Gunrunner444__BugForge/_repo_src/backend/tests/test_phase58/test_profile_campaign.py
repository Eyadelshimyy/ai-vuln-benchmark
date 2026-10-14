"""A small local vault proves the campaign path. It is not the Compound target."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.constructor_profile import parse_profile, sha256_text
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.solc_resolve import resolve_solc
from tests.test_phase51.phase51_support import manifest
from tests.test_phase52.phase52_support import requires_forge
from tests.test_phase58.test_constructor_profile import FILE, STATEMENT, VAULT, _profile

pytestmark = requires_forge


def _require_solc() -> None:
    if resolve_solc("0.8.15") is None:
        if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
            pytest.fail("the Foundry job requires solc 0.8.15")
        pytest.skip("solc 0.8.15 is not installed")


def test_pause_vault_campaign_holds_the_authorization_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_solc()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / FILE).write_text(VAULT, encoding="utf-8")
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{tmp_path / 'campaign.sqlite'}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    profile = parse_profile(_profile())
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
                in_scope=[{"kind": "contract", "identifier": "PauseVault"}],
                out_of_scope=[],
                known_issues=[],
                source_commit="phase58-pause-vault",
            ),
            repo_root=repo,
            contract="PauseVault",
            source_file=FILE,
            files=(FILE,),
            constructor_profile=profile.to_dict(),
        )
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    summary = result["campaign_summary"]
    assert summary["constructor_profile"]["validation"] == "accepted"
    assert summary["constructor_profile"]["local_environment"] is True
    assert summary["constructor_profile"]["establishes_deployed_state"] is False
    assert summary["constructor_profile"]["source_sha256"] == sha256_text(VAULT)
    assert summary["forge_runs"] >= 1
    assert summary["foundry_invoked"] is True
    assert summary["property_evaluation_demonstrated"] is True
    assert summary["by_outcome"].get("property_held", 0) >= 1
    assert summary["verified"] is False
    assert summary["sequences_submitted"] >= 1
    held = [
        item
        for item in result["observations"]
        if item["outcome"] == "property_held" and item["oracle_kind"] == "call_must_fail"
    ]
    assert held
    assert held[0]["verdict"] == "property_held"
    assert held[0]["strong_candidate"] is False
    assert held[0]["property_under_test"] == STATEMENT
    assert all(item["strong_candidate"] is False for item in result["observations"])
    props = [
        spec
        for spec in result["properties"].values()
        if spec["provenance"]["contract"] == "PauseVault"
        and spec["oracle"]["getter"] == "isSupplyPaused()"
    ]
    assert props
    assert props[0]["declaration"] == "property_under_test"
    assert props[0]["verified"] is False
    runs = BountyCampaignService().stateful_status(campaign.campaign_id)["runs"]
    tools = runs[-1]["tools"]
    assert tools["solc_selected_version"].startswith("0.8.15")
    assert tools["evm_requested"] == "cancun"
    assert tools["evm_exact"] == "false"
    assert tools["evm_version"] == ""
    assert tools["optimizer"] == "true"
    assert tools["via_ir"] == "true"
    reloaded = BountyCampaignService().research_ledger(campaign.campaign_id)["campaign_summary"]
    assert reloaded["forge_runs"] == summary["forge_runs"]
    assert reloaded["by_outcome"].get("property_held", 0) >= 1
    assert reloaded["constructor_profile"]["profile_id"] == "pause-vault-authorization"
    assert reloaded["verified"] is False
    findings = BountyCampaignService().findings(campaign.campaign_id)["findings"]
    pause = [item for item in findings if str(item.get("function", "")).startswith("pause(")]
    assert pause
    assert pause[0]["execution_status"] == "property_held"
    assert pause[0]["candidate_strength"] == "property_held"
    assert all(item["verified"] is False for item in findings)
    assert all(item["candidate_strength"] != "strong_candidate" for item in findings)
