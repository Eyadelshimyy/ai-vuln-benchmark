"""Sequences refused before Forge stay visible after the campaign is reloaded."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from tests.test_phase51.phase51_support import manifest
from tests.test_phase52.phase52_support import requires_forge

pytestmark = requires_forge

ACCOUNT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract {name} {{
    address public owner;
    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) {{ return 0; }}
    function initialize(address newOwner) external {{ owner = newOwner; }}
}}
"""


def test_skipped_sequences_reload_and_are_not_forge_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "vendor").mkdir()
    (repo / "src" / "Good.sol").write_text(ACCOUNT.format(name="Good"), encoding="utf-8")
    (repo / "vendor" / "Bad.sol").write_text(ACCOUNT.format(name="Bad"), encoding="utf-8")
    (repo / "src" / "Noise.sol").write_text(ACCOUNT.format(name="Noise"), encoding="utf-8")
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{tmp_path / 'campaign.sqlite'}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=manifest(
                in_scope=[{"kind": "contract", "identifier": "Good"}],
                out_of_scope=[{"kind": "path", "identifier": "vendor"}],
                known_issues=[],
                source_commit="phase57-skips",
            ),
            repo_root=repo,
            contract="Good",
            source_file="src/Good.sol",
            files=("src/Good.sol", "vendor/Bad.sol", "src/Noise.sol"),
        )
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    by_code = {item["reason_code"]: item for item in result["skipped"]}
    assert "blocked_by_policy:out_of_scope" in by_code
    assert "blocked_by_policy:unknown_scope" in by_code
    observed = {item.get("sequence_id") for item in result.get("observations", [])}
    skipped_ids = {item["sequence_id"] for item in result["skipped"]}
    assert observed.isdisjoint(skipped_ids)
    summary = result["campaign_summary"]
    assert summary["sequences_excluded"] == len(result["skipped"])
    assert summary["exclusion_reasons"]["blocked_by_policy:out_of_scope"] >= 1
    assert summary["exclusion_reasons"]["blocked_by_policy:unknown_scope"] >= 1
    assert summary["execution_stages"]["tool_unavailable"] == 0
    assert summary["verified"] is False

    reloaded = BountyCampaignService().stateful_status(campaign.campaign_id)
    stored = reloaded["skipped_sequences"]
    assert skipped_ids <= set(stored)
    assert all(item.get("forge_invoked") == "false" for item in stored.values())
    assert all(item.get("outcome") == "skipped" for item in stored.values())
    assert set(reloaded["executions"]) & skipped_ids == set()
    ledger = BountyCampaignService().research_ledger(campaign.campaign_id)["campaign_summary"]
    assert ledger["sequences_excluded"] == summary["sequences_excluded"]
    assert (
        ledger["execution_stages"]["compiler_unavailable"]
        == summary["execution_stages"]["compiler_unavailable"]
    )
    findings = {
        (item["contract"], item["execution_status"]): item
        for item in BountyCampaignService().findings(campaign.campaign_id)["findings"]
    }
    assert any(contract == "Bad" and status == "skipped" for contract, status in findings)
    assert any(contract == "Noise" and status == "skipped" for contract, status in findings)
