"""A multi-file campaign through analysis, Foundry, persistence, and invalidation.

The positive candidate comes from the real detector suite. The safe twin is not
reported as a discovery. A local property violation is not a production finding.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.stateful import Outcome, tool_status
from app.discovery.process import tool_path
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase51.phase51_support import manifest

GUARD = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Guard {
    address public owner;
    function _check() internal view {
        require(msg.sender == owner, "owner");
    }
    modifier onlyOwner() {
        _check();
        _;
    }
}
"""

INTERFACE = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IAccount {
    function validateUserOp(
        bytes calldata userOp,
        bytes32 userOpHash,
        uint256 missingAccountFunds
    ) external returns (uint256);
}
"""

ACCOUNT = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../access/Guard.sol";
import "../interfaces/IAccount.sol";

contract Account is Guard, IAccount {
    function validateUserOp(bytes calldata, bytes32, uint256) external pure returns (uint256) {
        return 0;
    }

    function initialize(address newOwner) external {
        owner = newOwner;
    }

    function execute(address target, uint256 value, bytes calldata data) external onlyOwner {
        (bool ok,) = target.call{value: value}(data);
        ok;
    }
}
"""

GUARDED = ACCOUNT.replace(
    """    function initialize(address newOwner) external {
        owner = newOwner;
    }""",
    """    bool public initialized;
    function initialize(address newOwner) external {
        require(!initialized, "initialized");
        initialized = true;
        owner = newOwner;
    }""",
)


def _require_toolchain() -> None:
    forge = tool_path("forge")
    solc = tool_path("solc")
    tools = tool_status()
    ready = bool(
        forge and solc and Path(forge).is_file() and Path(solc).is_file() and tools.available
    )
    if ready:
        return
    message = f"forge/solc not usable: forge={forge!r} solc={solc!r}"
    if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
        pytest.fail(message)
    pytest.skip(message)


def test_multifile_campaign_runs_the_real_pipeline_and_invalidates_on_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_toolchain()
    repo = tmp_path / "repo"
    files = {
        "src/access/Guard.sol": GUARD,
        "src/interfaces/IAccount.sol": INTERFACE,
        "src/account/Account.sol": ACCOUNT,
    }
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
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
                in_scope=[{"kind": "contract", "identifier": "Account"}],
                out_of_scope=[{"kind": "contract", "identifier": "Guard"}],
                known_issues=[],
                source_commit="fixture-phase55-multifile",
                testing_mode="local_only",
            ),
            repo_root=repo,
            contract="Account",
            source_file="src/account/Account.sol",
        )
    )
    analysis = svc._analysis(campaign)
    assert "src/account/Account.sol" in analysis.sources
    assert "src/access/Guard.sol" in analysis.sources
    assert "src/interfaces/IAccount.sol" in analysis.sources
    assert analysis.model is not None
    assert dict(analysis.model.source_hashes)["src/account/Account.sol"]
    hits = [c for c in analysis.candidates if c.detector == "aa.unprotected_account_initializer"]
    assert len(hits) == 1 and hits[0].contract == "Account"
    assert not any(
        c.detector == "aa.unauthenticated_account_execution" for c in analysis.candidates
    )
    sequences = [s for s in analysis.sequences if s.template == "initialize→reinitialize"]
    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.derived_from.startswith("aa.unprotected_account_initializer@Account.")
    assert sequence.status == "candidate" and sequence.verified is False

    result = svc.stateful_execute(campaign.campaign_id, rounds=2)
    assert result["available"] is True
    assert result["persistence"]["status"] == "persisted"
    assert result["forge_runs"] >= 1
    assert result["dynamic_foundry_execution_demonstrated"] is True
    assert result["campaign_summary"]["blocker"] == ""
    assert result["campaign_summary"]["verified"] is False
    print(f"multifile forge_runs={result['forge_runs']} outcomes={result['outcomes']}")
    observed = next(
        item for item in result["observations"] if item["sequence_id"] == sequence.sequence_id
    )
    assert observed["outcome"] == Outcome.PROPERTY_VIOLATED.value
    assert observed["verified"] is False
    assert observed["result"]["compiler_status"] == "ok"
    assert observed["result"]["test_status"] == "passed"
    assert observed["result"]["process_status"] == "completed"
    assert observed["identity"]["status"] == "bound"
    assert observed["identity"]["replay_mode"] == "local_source_replay"
    assert observed["identity"]["deployment"] == "source"
    assert any(
        row["name"] == "foundry" and row["invoked"] is True
        for row in result["campaign_summary"]["engines"]
    )

    reloaded = BountyCampaignService()
    status = reloaded.stateful_status(campaign.campaign_id)
    stored = status["executions"][sequence.sequence_id]
    bundle_id = stored["bundle_id"]
    bundle = reloaded.repro_bundle(campaign.campaign_id, bundle_id)["bundle"]
    assert bundle["execution"]["outcome"] == "property_violated"
    assert bundle["execution"]["verified"] is False

    account = repo / "src" / "account" / "Account.sol"
    account.write_text(account.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rejected = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert rejected["forge_runs"] == 0
    assert rejected["dynamic_foundry_execution_demonstrated"] is False
    assert {item["outcome"] for item in rejected["observations"]} == {"identity_mismatch"}
    assert "reanalyze" in rejected["blocker"]

    kept = BountyCampaignService().repro_bundle(campaign.campaign_id, bundle_id)["bundle"]
    assert kept["execution"]["outcome"] == "property_violated"
    latest = BountyCampaignService().stateful_status(campaign.campaign_id)["executions"][
        sequence.sequence_id
    ]
    assert latest["outcome"] == "identity_mismatch"
    assert latest["verified"] is False
    findings = BountyCampaignService().findings(campaign.campaign_id)["findings"]
    item = next(f for f in findings if f["detector"] == "aa.unprotected_account_initializer")
    assert item["verified"] is False
    assert item["execution_status"] == "identity_mismatch"
    assert item["candidate_strength"] == "static_candidate"

    rebound = svc.reanalyze(campaign.campaign_id)
    rebound_sequence = next(s for s in rebound.sequences if s.template == "initialize→reinitialize")
    assert (
        dict(rebound_sequence.identity.source_hashes)["src/account/Account.sol"]
        != dict(sequence.identity.source_hashes)["src/account/Account.sol"]
    )
    assert rebound_sequence.verified is False
    after = BountyCampaignService().findings(campaign.campaign_id)["findings"]
    after_item = next(f for f in after if f["detector"] == "aa.unprotected_account_initializer")
    assert after_item["execution_status"] == "identity_mismatch"
    assert after_item["verified"] is False

    safe = {
        "src/access/Guard.sol": GUARD,
        "src/interfaces/IAccount.sol": INTERFACE,
        "src/account/Account.sol": GUARDED,
    }
    safe_hits = [
        c.detector
        for c in run_suite(build_research_model(safe)).candidates
        if c.detector == "aa.unprotected_account_initializer"
    ]
    assert safe_hits == []
