"""Campaign-path check for the swallowed-call behavioral probe.

The fixture, campaign database, and Forge workspace stay outside the repository.
Forge runs only when ``forge`` and ``solc`` are installed. A missing tool, a
compiler error, or ``UNAVAILABLE`` is not a completed dynamic validation. The
dedicated CI job sets ``BUGFORGE_REQUIRE_FOUNDRY=1`` so that case fails.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.properties import build_property
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.stateful import IdentityContext, Outcome, StatefulExecutor, tool_status
from app.discovery.bounty.vfcs import stamp_source_identity
from app.discovery.process import tool_path
from app.parsing.solidity_call_outcome import DETECTOR
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase51.phase51_support import manifest

SOURCE_REVISION = "fixture-swallowed-call-1"
TEMPLATE = "failing-callee→swallowed-call"

FIXTURE = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Relay {
    function relay(address target, bytes calldata data) external {
        (bool ok, ) = target.call(data);
        ok;
    }
}

contract RelayStrict {
    function relay(address target, bytes calldata data) external {
        (bool ok, ) = target.call(data);
        require(ok, "propagated");
    }
}
"""

PROPAGATED = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Relay {
    function relay(address target, bytes calldata data) external {
        (bool ok, ) = target.call(data);
        require(ok, "propagated");
    }
}
"""


def _require_toolchain() -> None:
    tools = tool_status()
    forge = tool_path("forge")
    solc = tool_path("solc")
    ready = bool(
        forge
        and solc
        and Path(forge).is_file()
        and Path(solc).is_file()
        and tools.available
        and tools.forge_version
        and tools.solc_version
    )
    if ready:
        return
    message = f"forge/solc not usable: forge={forge!r} solc={solc!r} {tools.as_dict()}"
    if os.environ.get("BUGFORGE_REQUIRE_FOUNDRY") == "1":
        pytest.fail(message)
    pytest.skip(message)


def test_campaign_observes_swallowed_call_without_confirming_a_vulnerability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Analysis, VFCS, the campaign service, and Forge, then a persisted read-back."""
    _require_toolchain()
    tools = tool_status()
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "Relay.sol"
    source.write_text(FIXTURE, encoding="utf-8")
    source_hash = hashlib.sha256(FIXTURE.encode("utf-8")).hexdigest()
    negative = tmp_path / "negative" / "Relay.sol"
    negative.parent.mkdir()
    negative.write_text(PROPAGATED, encoding="utf-8")
    database = tmp_path / "campaign.sqlite"

    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{database}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    mani = manifest(
        in_scope=[
            {"kind": "contract", "identifier": "Relay"},
            {"kind": "contract", "identifier": "RelayStrict"},
            {"kind": "path", "identifier": "Relay.sol"},
        ],
        out_of_scope=[],
        known_issues=[],
        source_commit=SOURCE_REVISION,
        testing_mode="local_only",
        impact_categories=[],
    )
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=mani,
            repo_root=repo,
            contract="Relay",
            source_file="Relay.sol",
            files=("Relay.sol",),
        )
    )
    analysis = svc._analysis(campaign)
    swallowed = [c for c in analysis.candidates if c.detector == DETECTOR and c.contract == "Relay"]
    assert len(swallowed) == 1
    candidate = swallowed[0]
    assert candidate.file == "Relay.sol"
    assert candidate.verified is False and candidate.confidence == "low"
    assert dict(candidate.facts)["vulnerability_claim"] == "none"
    assert [c for c in analysis.candidates if c.contract == "RelayStrict"] == []
    assert analysis.source_hashes["Relay.sol"] == source_hash
    sequences = [s for s in analysis.sequences if s.template == TEMPLATE]
    assert len(sequences) == 1, [s.template for s in analysis.sequences]
    sequence = sequences[0]
    assert sequence.derived_from.startswith(f"{DETECTOR}@Relay.")
    assert sequence.identity.source_snapshot == SOURCE_REVISION
    assert dict(sequence.identity.source_hashes)["Relay.sol"] == source_hash
    plans = svc.repro(campaign.campaign_id)["sequences"]
    assert [plan["template"] for plan in plans] == [TEMPLATE]

    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    if not result.get("available"):
        pytest.fail(result.get("reason", "stateful execution did not run"))
    assert result["persistence"]["status"] == "persisted"
    assert result["forge_runs"] == 1
    assert result["outcomes"].get("property_violated", 0) == 0
    assert result["outcomes"].get("mechanism_observed") == 1
    assert result["outcomes"].get("unavailable", 0) == 0
    observed = [
        item for item in result["observations"] if item["sequence_id"] == sequence.sequence_id
    ]
    assert len(observed) == 1
    observation = observed[0]
    assert observation["outcome"] == Outcome.MECHANISM_OBSERVED.value
    assert observation["verified"] is False
    assert observation["strong_candidate"] is False
    assert observation["executed"] is True
    assert "unproven" in observation["reason"]
    assert "security-property violation" in observation["reason"]
    identity = observation["identity"]
    assert identity["status"] == "bound"
    assert identity["contract_file"] == "Relay.sol"
    assert identity["source_snapshot"] == SOURCE_REVISION
    assert identity["source_hashes"]["Relay.sol"] == source_hash
    execution = observation["result"]
    assert execution["process_status"] == "completed"
    assert execution["compiler_status"] == "ok"
    assert execution["test_status"] == "passed"
    assert execution["property_status"] == "mechanism_observed"
    assert execution["call_results"][1] == 1
    assert execution["forge_version"] == tools.forge_version
    assert execution["solc_version"] == tools.solc_version
    assert execution["harness_hash"]
    assert execution["command"][:3] == ["forge", "build", "--offline"]
    assert "forge" in execution["command"] and "test" in execution["command"]
    prop = result["properties"][sequence.sequence_id]
    assert prop["declaration"] == "mechanism_observation"
    assert prop["oracle"]["kind"] == "mechanism_probe"
    assert prop["provenance"]["file"] == "Relay.sol"
    assert prop["verified"] is False

    reloaded = BountyCampaignService()
    status = reloaded.stateful_status(campaign.campaign_id)
    assert database.is_file() and database.stat().st_size > 0
    assert status["runs"][-1]["violations"] == 0
    assert status["runs"][-1]["mechanism_observations"] == 1
    assert status["runs"][-1]["forge_runs"] == 1
    stored = status["executions"][sequence.sequence_id]
    assert stored["outcome"] == "mechanism_observed"
    assert stored["strong_candidate"] is False
    assert stored["verified"] is False
    assert stored["bundle_id"]
    bundle = reloaded.repro_bundle(campaign.campaign_id, stored["bundle_id"])
    assert bundle["self_contained"] is True
    body = bundle["bundle"]
    assert body["execution"]["outcome"] == "mechanism_observed"
    assert body["execution"]["verified"] is False
    assert body["environment"]["offline"] is True
    assert body["environment"]["ffi"] is False
    assert body["environment"]["rpc"] == "none"
    assert body["environment"]["forge_version"] == tools.forge_version
    assert body["environment"]["solc_version"] == tools.solc_version
    assert body["sequence"]["identity"]["source_snapshot"] == SOURCE_REVISION
    assert body["source_hashes"]["Relay.sol"] == source_hash
    assert body["command"][0] == "forge"
    assert bundle["sources"][source_hash] == FIXTURE

    findings = reloaded.findings(campaign.campaign_id)["findings"]
    item = next(f for f in findings if f["detector"] == DETECTOR and f["contract"] == "Relay")
    assert item["file"] == "Relay.sol"
    assert item["verified"] is False
    assert item["candidate_strength"] == "mechanism_observation"
    assert item["poc_status"] == "mechanism_observation_unproven"
    assert item["qualification"] == "mechanism_observation"
    assert item["confirmed_severity"] == "unconfirmed"
    assert item["severity_candidate"] == "unknown"
    assert item["quality"]["verification"] == "unverified"
    assert item["quality"]["corroboration"] == "mechanism_only"
    assert item["poc_status"] != "local_harness_violation_unverified"
    assert all(f["contract"] != "RelayStrict" for f in findings)

    ledger = reloaded.research_ledger(campaign.campaign_id)["RESEARCH_COVERAGE"]
    assert ledger["by_outcome"].get("mechanism_observed") == 1
    assert ledger["by_outcome"].get("property_violated", 0) == 0
    assert ledger["by_family"]["mechanism"]["violated"] == 0

    negative_text = negative.read_text(encoding="utf-8")
    negative_sources = {"Relay.sol": negative_text}
    executor = StatefulExecutor()
    mismatched = executor.execute(
        sequence,
        build_research_model(negative_sources),
        negative_sources,
        spec=build_property(sequence, analysis.models[sequence.sequence_id], candidate),
    )
    assert mismatched.outcome is Outcome.IDENTITY_MISMATCH, mismatched.reason
    assert mismatched.strong_candidate is False and mismatched.verified is False
    assert executor.runs == 0
    # The propagating body is not a detector hit. The control rebinds the campaign
    # plan to that text; it is not a claim that analysis discovered the candidate.
    control_model = build_research_model(negative_sources)
    assert [c for c in run_suite(control_model).candidates if c.detector == DETECTOR] == []
    control = stamp_source_identity(sequence, negative_sources)
    control_spec = build_property(control, control_model)
    # The campaign plan records a compiler configuration. The control has to
    # establish that same configuration; it does not drop the requirement.
    propagated = executor.execute(
        control,
        control_model,
        negative_sources,
        spec=control_spec,
        context=IdentityContext(
            compiler_configuration=control.identity.compiler_configuration,
            implementation=control.identity.implementation,
        ),
    )
    assert executor.runs == 1
    assert propagated.outcome is Outcome.FAILURE_PROPAGATED, propagated.reason
    assert propagated.verified is False and propagated.strong_candidate is False
    assert propagated.result is not None
    assert propagated.result.compiler_status == "ok"
    assert propagated.result.test_status == "passed"
    assert propagated.result.process_status == "completed"
    assert propagated.result.property_status == "failure_propagated"
    assert propagated.result.call_results[1] == 0
    assert "not reproduced" in propagated.reason
