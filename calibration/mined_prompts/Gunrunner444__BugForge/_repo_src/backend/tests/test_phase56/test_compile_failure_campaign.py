"""A campaign whose harness does not compile must not report a property evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from tests.test_phase51.phase51_support import manifest
from tests.test_phase52.phase52_support import requires_forge

pytestmark = requires_forge

TYPE_ERROR = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Acct {
    address public owner;

    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) {
        return 0;
    }

    function initialize(address newOwner) external {
        owner = true;
    }
}
"""


def test_compile_failure_campaign_does_not_claim_property_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Acct.sol").write_text(TYPE_ERROR, encoding="utf-8")
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
                in_scope=[{"kind": "contract", "identifier": "Acct"}],
                out_of_scope=[],
                known_issues=[],
                source_commit="phase56-compile-failed",
            ),
            repo_root=repo,
            contract="Acct",
            source_file="Acct.sol",
            files=("Acct.sol",),
        )
    )
    assert campaign.selection.dependency_status() == "traced"
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    assert result["forge_runs"] > 0
    assert result["outcomes"].get("compile_failed", 0) >= 1
    assert result["outcomes"].get("property_violated", 0) == 0
    assert result["outcomes"].get("property_held", 0) == 0
    assert result["dynamic_foundry_execution_demonstrated"] is False
    assert result["campaign_summary"]["property_evaluation_demonstrated"] is False
    assert result["campaign_summary"]["foundry_invoked"] is True
    assert result["blocker"] == "compilation failed"
    assert result["campaign_summary"]["execution_stages"]["build_failed"] >= 1
    assert result["campaign_summary"]["execution_stages"]["property_verdicts"] == 0
    assert result["campaign_summary"]["verified"] is False
    stored = svc.stateful_status(campaign.campaign_id)["executions"]
    assert stored
    record = next(iter(stored.values()))
    assert record["outcome"] == "compile_failed"
    assert record["compiler_status"] == "failed"
    assert record["test_status"] == "not_run"
    assert record["property_status"] == "not_run"
