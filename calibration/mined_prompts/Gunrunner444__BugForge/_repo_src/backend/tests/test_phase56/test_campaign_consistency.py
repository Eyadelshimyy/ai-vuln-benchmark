"""Unread source, incomplete analysis, identity refusal, and recovery share one pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.source_selection import MAX_FILE_BYTES
from app.discovery.bounty.stateful import IdentityContext, Outcome, StatefulExecutor
from tests.test_phase51.phase51_support import manifest

NOISE = "contract Noise { uint x; }\n"
ACCOUNT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Acct {
    address public owner;

    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) {
        return 0;
    }

    function initialize(address newOwner) external {
        owner = newOwner;
    }
}
"""


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BountyCampaignService:
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{tmp_path / 'campaign.sqlite'}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    return BountyCampaignService()


def test_unreadable_source_blocks_a_complete_claim_and_recovery_reanalyzes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    vault = repo / "Vault.sol"
    vault.write_bytes(b"contract Vault { uint x; }\n" + b" " * (MAX_FILE_BYTES + 4))
    (repo / "Noise.sol").write_text(NOISE, encoding="utf-8")
    svc = _service(tmp_path, monkeypatch)
    campaign = svc.create(
        CampaignSpec(
            manifest=manifest(
                in_scope=[{"kind": "contract", "identifier": "Acct"}],
                out_of_scope=[],
                known_issues=[],
                source_commit="phase56-unreadable",
            ),
            repo_root=repo,
            contract="Acct",
            source_file="Vault.sol",
        )
    )
    assert "Vault.sol" in campaign.selection.selected
    ledger = svc.research_ledger(campaign.campaign_id)
    summary = ledger["campaign_summary"]
    assert summary["analysis_complete"] is False
    assert summary["snapshot_fresh"] is True
    assert summary["dependency_status"] != "traced"
    assert summary["repository_coverage"] != "complete"
    assert summary["files_read"] < summary["files_selected"]
    assert any(
        item["path"] == "Vault.sol" and item["reason"] == "size_limit"
        for item in summary["unreadable_files"]
    )
    assert summary["candidates"] == 0
    assert summary["sequences"] == 0
    assert summary["dynamic_foundry_execution_demonstrated"] is False
    assert summary["property_evaluation_demonstrated"] is False
    assert "size_limit" in summary["blocker"] or "could not be read" in summary["blocker"]
    assert summary["verified"] is False

    attempted = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert attempted["dynamic_foundry_execution_demonstrated"] is False
    if attempted.get("available"):
        assert attempted["forge_runs"] == 0

    vault.write_text(ACCOUNT, encoding="utf-8")
    recovered = svc.reanalyze(campaign.campaign_id)
    assert recovered.model is not None
    assert "Vault.sol" in recovered.sources
    assert any(item.contract == "Acct" for item in recovered.candidates)
    fresh = svc.research_ledger(campaign.campaign_id)["campaign_summary"]
    assert fresh["analysis_stale"] is False
    assert fresh["snapshot_fresh"] is True
    assert fresh["files_read"] == fresh["files_selected"]
    assert fresh["unreadable_files"] == []
    assert fresh["dependency_status"] == "traced"
    assert fresh["analysis_complete"] is True
    assert fresh["dynamic_foundry_execution_demonstrated"] is False
    assert fresh["sequences"] >= 1
    sequence = next(item for item in recovered.sequences if "initialize" in item.template)
    model = recovered.models[sequence.sequence_id]

    vault.write_text(ACCOUNT + "\n", encoding="utf-8")
    stale = svc.research_ledger(campaign.campaign_id)["campaign_summary"]
    assert stale["analysis_stale"] is True
    assert stale["snapshot_fresh"] is False
    assert stale["analysis_complete"] is False
    assert "reanalyze" in stale["blocker"]
    rejected = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert rejected["dynamic_foundry_execution_demonstrated"] is False
    if rejected.get("available"):
        assert rejected["forge_runs"] == 0
        assert {item["outcome"] for item in rejected["observations"]} == {"identity_mismatch"}
        assert "reanalyze" in rejected["blocker"]
    else:
        assert rejected["dynamic_foundry_execution_demonstrated"] is False

    executor = StatefulExecutor()
    observation = executor.execute(
        sequence,
        model,
        {"Vault.sol": ACCOUNT + "\n", "Noise.sol": NOISE},
        context=IdentityContext(
            compiler_configuration=sequence.identity.compiler_configuration,
            implementation=sequence.identity.implementation,
            deployment="source",
        ),
    )
    assert observation.outcome is Outcome.IDENTITY_MISMATCH
    assert executor.runs == 0

    rebound = svc.reanalyze(campaign.campaign_id)
    assert svc.analysis_stale(campaign) is False
    assert rebound.model is not None
    assert (
        dict(rebound.model.source_hashes)["Vault.sol"]
        != dict(recovered.model.source_hashes)["Vault.sol"]
    )
    after = svc.research_ledger(campaign.campaign_id)["campaign_summary"]
    assert after["snapshot_fresh"] is True
    assert after["analysis_complete"] is True
    assert after["dynamic_foundry_execution_demonstrated"] is False
    assert after["verified"] is False
