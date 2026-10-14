"""Campaign execution for supported property templates other than the swallowed-call probe.

Each positive path starts from detector output. A safe control that the detector does
not emit is stamped in the test and is not reported as a discovery.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.properties import build_property
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.stateful import Outcome, StatefulExecutor, tool_status
from app.discovery.bounty.vfcs import stamp_source_identity
from app.discovery.process import tool_path
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase51.phase51_support import manifest
from tests.test_phase52.phase52_support import (
    DELTA_VAULT,
    RAW_VAULT,
    deposit_sequence,
    init_sequence,
    model_of,
)

HISTORY = Path(__file__).resolve().parents[1] / "fixtures" / "phase52_history"

STUB = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract EntrypointStub {
    function validateUserOp(
        bytes calldata userOp,
        bytes32 userOpHash,
        uint256 missingAccountFunds
    ) external returns (uint256) {
        return 0;
    }
}
"""

UNGUARDED = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Stub.sol";

contract Acct is EntrypointStub {
    address public owner;

    function initialize(address newOwner) external {
        owner = newOwner;
    }
}
"""

GUARDED = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Stub.sol";

contract Acct is EntrypointStub {
    address public owner;
    bool public initialized;

    function initialize(address newOwner) external {
        require(!initialized, "initialized");
        initialized = true;
        owner = newOwner;
    }
}
"""

BROKEN = UNGUARDED.replace("owner = newOwner;", "owner = true;")


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


def _campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    *,
    contract: str,
    in_scope: list[str],
    out_of_scope: list[str] | None = None,
) -> tuple[BountyCampaignService, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    database = tmp_path / "campaign.sqlite"
    monkeypatch.setattr(
        get_settings(),
        "bounty_campaign_database_url",
        f"sqlite:///{database}",
        raising=False,
    )
    monkeypatch.setattr(service_module, "_SERVICE", None, raising=False)
    mani = manifest(
        in_scope=[{"kind": "contract", "identifier": name} for name in in_scope],
        out_of_scope=[{"kind": "contract", "identifier": name} for name in (out_of_scope or [])],
        known_issues=[],
        source_commit="fixture-phase54",
        testing_mode="local_only",
        impact_categories=[],
    )
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=mani,
            repo_root=repo,
            contract=contract,
            source_file=next(iter(files)),
            files=tuple(files),
        )
    )
    return svc, campaign


def test_initializer_campaign_violates_minimizes_and_rejects_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_toolchain()
    files = {"Stub.sol": STUB, "Acct.sol": UNGUARDED}
    svc, campaign = _campaign(
        tmp_path,
        monkeypatch,
        files,
        contract="Acct",
        in_scope=["Acct"],
        out_of_scope=["EntrypointStub"],
    )
    analysis = svc._analysis(campaign)
    hits = [c for c in analysis.candidates if c.detector == "aa.unprotected_account_initializer"]
    assert len(hits) == 1 and hits[0].contract == "Acct" and hits[0].file == "Acct.sol"
    sequences = [s for s in analysis.sequences if s.template == "initialize→reinitialize"]
    assert len(sequences) == 1
    sequence = sequences[0]
    assert sequence.derived_from.startswith("aa.unprotected_account_initializer@Acct.")
    assert dict(sequence.identity.source_hashes)["Acct.sol"]
    result = svc.stateful_execute(campaign.campaign_id, rounds=2)
    assert result["available"] is True
    assert result["persistence"]["status"] == "persisted"
    assert result["forge_runs"] >= 1
    print(f"initializer forge_runs={result['forge_runs']} outcomes={result['outcomes']}")
    observed = next(
        item for item in result["observations"] if item["sequence_id"] == sequence.sequence_id
    )
    assert observed["outcome"] == Outcome.PROPERTY_VIOLATED.value
    assert observed["verified"] is False and observed["strong_candidate"] is True
    assert observed["result"]["compiler_status"] == "ok"
    assert observed["result"]["test_status"] == "passed"
    assert observed["result"]["process_status"] == "completed"
    assert observed["identity"]["status"] == "bound"
    assert observed["identity"]["source_snapshot"] == "fixture-phase54"
    minimized = result["minimized"][sequence.sequence_id]
    assert minimized["completed"] is True and minimized["minimized"] == 2
    assert minimized["preserved_property"] == sequence.property_under_test
    independent = result["independent_checks"][sequence.sequence_id]
    assert independent["independence"] == "same_toolchain_alternate"
    assert independent["verified"] is False
    assert independent["status"] == "corroborated_candidate"
    assert "same toolchain" in independent["note"]
    assert result["outcomes"].get("mechanism_observed", 0) == 0

    reloaded = BountyCampaignService()
    status = reloaded.stateful_status(campaign.campaign_id)
    stored = status["executions"][sequence.sequence_id]
    assert stored["outcome"] == "property_violated"
    assert stored["independence"] == "same_toolchain_alternate"
    assert stored["verified"] is False
    bundle_id = stored["bundle_id"]
    bundle = reloaded.repro_bundle(campaign.campaign_id, bundle_id)["bundle"]
    assert (
        bundle["sequence"]["identity"]["source_hashes"]["Acct.sol"]
        == dict(sequence.identity.source_hashes)["Acct.sol"]
    )
    assert bundle["property"]["declaration"] == "property_under_test"
    assert bundle["execution"]["verified"] is False
    findings = reloaded.findings(campaign.campaign_id)["findings"]
    item = next(f for f in findings if f["detector"] == "aa.unprotected_account_initializer")
    assert item["verified"] is False
    assert item["poc_status"] == "local_harness_violation_unverified"
    assert item["candidate_strength"] == "corroborated_candidate"
    assert item["qualification"] != "mechanism_observation"
    assert item["confirmed_severity"] == "unconfirmed"
    pack = reloaded.report_pack(campaign.campaign_id)
    assert pack.verified is False and all(report.verified is False for report in pack.reports)
    ledger = reloaded.research_ledger(campaign.campaign_id)["RESEARCH_COVERAGE"]
    assert ledger["by_family"]["initialization"]["violated"] >= 1
    assert ledger["by_outcome"].get("property_violated", 0) >= 1
    assert ledger["by_outcome"].get("mechanism_observed", 0) == 0

    path = tmp_path / "repo" / "Acct.sol"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rejected = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert rejected["forge_runs"] == 0
    assert {item["outcome"] for item in rejected["observations"]} == {"identity_mismatch"}
    assert rejected["outcomes"].get("property_violated", 0) == 0
    fresh = BountyCampaignService()
    kept = fresh.repro_bundle(campaign.campaign_id, bundle_id)["bundle"]
    assert kept["execution"]["outcome"] == "property_violated"
    latest = fresh.stateful_status(campaign.campaign_id)["executions"][sequence.sequence_id]
    assert latest["outcome"] == "identity_mismatch"
    assert latest["verified"] is False
    assert any(item.get("outcome") == "property_violated" for item in latest.get("history", []))
    stale = fresh.findings(campaign.campaign_id)["findings"]
    stale_item = next(f for f in stale if f["detector"] == "aa.unprotected_account_initializer")
    assert stale_item["verified"] is False
    assert stale_item["execution_status"] == "identity_mismatch"
    assert stale_item["candidate_strength"] == "static_candidate"
    assert stale_item["poc_status"] != "local_harness_violation_unverified"

    safe_files = {"Stub.sol": STUB, "Acct.sol": GUARDED}
    safe_model = model_of(safe_files)
    assert [
        c
        for c in run_suite(safe_model).candidates
        if c.detector == "aa.unprotected_account_initializer"
    ] == []
    control = stamp_source_identity(init_sequence(), safe_files)
    held = StatefulExecutor().execute(
        control, safe_model, safe_files, spec=build_property(control, safe_model)
    )
    assert held.outcome is Outcome.PROPERTY_HELD, held.reason
    assert held.verified is False and held.strong_candidate is False


def test_fee_on_transfer_campaign_credits_more_than_it_received(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_toolchain()
    svc, campaign = _campaign(
        tmp_path,
        monkeypatch,
        {"Vault.sol": RAW_VAULT},
        contract="Vault",
        in_scope=["Vault"],
    )
    analysis = svc._analysis(campaign)
    hits = [c for c in analysis.candidates if c.detector == "accounting.fee_on_transfer_mismatch"]
    assert len(hits) == 1 and hits[0].file == "Vault.sol"
    sequence = next(
        s
        for s in analysis.sequences
        if s.derived_from.startswith("accounting.fee_on_transfer_mismatch@")
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert result["forge_runs"] >= 1 and result["persistence"]["status"] == "persisted"
    print(f"fee_on_transfer forge_runs={result['forge_runs']} outcomes={result['outcomes']}")
    observed = next(
        item for item in result["observations"] if item["sequence_id"] == sequence.sequence_id
    )
    assert observed["outcome"] == Outcome.PROPERTY_VIOLATED.value
    assert observed["verified"] is False
    assert observed["result"]["compiler_status"] == "ok"
    assert observed["result"]["test_status"] == "passed"
    assert (
        observed["identity"]["source_hashes"]["Vault.sol"]
        == dict(sequence.identity.source_hashes)["Vault.sol"]
    )
    reloaded = BountyCampaignService()
    stored = reloaded.stateful_status(campaign.campaign_id)["executions"][sequence.sequence_id]
    assert stored["outcome"] == "property_violated" and stored["verified"] is False
    finding = next(
        f
        for f in reloaded.findings(campaign.campaign_id)["findings"]
        if f["detector"] == "accounting.fee_on_transfer_mismatch"
    )
    assert finding["verified"] is False
    assert finding["poc_status"] == "local_harness_violation_unverified"
    assert finding["qualification"] != "mechanism_observation"
    ledger = reloaded.research_ledger(campaign.campaign_id)["RESEARCH_COVERAGE"]
    assert ledger["by_family"]["token_movement"]["violated"] >= 1

    safe_sources = {"Vault.sol": DELTA_VAULT}
    safe_model = model_of(safe_sources)
    assert [
        c
        for c in run_suite(safe_model).candidates
        if c.detector == "accounting.fee_on_transfer_mismatch"
    ] == []
    control = stamp_source_identity(deposit_sequence(), safe_sources)
    held = StatefulExecutor().execute(
        control, safe_model, safe_sources, spec=build_property(control, safe_model)
    )
    assert held.outcome is Outcome.PROPERTY_HELD, held.reason
    assert held.verified is False


def test_bridge_receiver_campaign_rejects_an_unauthenticated_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_toolchain()
    vulnerable = (HISTORY / "bridge_receiver_vulnerable.sol").read_text(encoding="utf-8")
    safe = (HISTORY / "bridge_receiver_safe.sol").read_text(encoding="utf-8")
    svc, campaign = _campaign(
        tmp_path,
        monkeypatch,
        {"OpenReceiver.sol": vulnerable},
        contract="OpenReceiver",
        in_scope=["OpenReceiver"],
    )
    analysis = svc._analysis(campaign)
    hits = [
        c for c in analysis.candidates if c.detector == "bridge.receiver_without_endpoint_check"
    ]
    assert len(hits) == 1 and hits[0].contract == "OpenReceiver"
    sequence = analysis.sequences[0]
    assert sequence.template == "unauthenticated-execution"
    assert sequence.derived_from.startswith("bridge.receiver_without_endpoint_check@")
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert result["forge_runs"] >= 1
    print(f"bridge forge_runs={result['forge_runs']} outcomes={result['outcomes']}")
    observed = result["observations"][0]
    assert observed["sequence_id"] == sequence.sequence_id
    assert observed["outcome"] == Outcome.PROPERTY_VIOLATED.value
    assert observed["verified"] is False and observed["strong_candidate"] is True
    assert observed["result"]["test_status"] == "passed"
    reloaded = BountyCampaignService()
    stored = reloaded.stateful_status(campaign.campaign_id)["executions"][sequence.sequence_id]
    assert stored["outcome"] == "property_violated" and stored["verified"] is False
    bundle = reloaded.repro_bundle(campaign.campaign_id, stored["bundle_id"])["bundle"]
    assert bundle["source_hashes"]["OpenReceiver.sol"]
    finding = next(
        f
        for f in reloaded.findings(campaign.campaign_id)["findings"]
        if f["detector"] == "bridge.receiver_without_endpoint_check"
    )
    assert finding["poc_status"] == "local_harness_violation_unverified"
    assert finding["verified"] is False
    assert finding["qualification"] != "mechanism_observation"

    safe_sources = {"OpenReceiver.sol": safe}
    safe_model = model_of(safe_sources)
    assert [
        c
        for c in run_suite(safe_model).candidates
        if c.detector == "bridge.receiver_without_endpoint_check"
    ] == []
    from app.discovery.bounty.vfcs import SequenceIdentity, Vfcs, VfcsCall

    control = stamp_source_identity(
        Vfcs(
            sequence_id="vf_bridge_control",
            template="unauthenticated-execution",
            origin="test-control:not-a-discovery",
            calls=(
                VfcsCall(
                    "OpenReceiver",
                    "receiveMessage(bytes)",
                    "execute",
                    "attacker",
                    (("message", "unconstrained"),),
                ),
            ),
            property_under_test="execution requires a valid operation",
            derived_from=(
                "bridge.receiver_without_endpoint_check@OpenReceiver.receiveMessage(bytes)"
            ),
            identity=SequenceIdentity(),
        ),
        safe_sources,
    )
    held = StatefulExecutor().execute(
        control, safe_model, safe_sources, spec=build_property(control, safe_model)
    )
    assert held.outcome is Outcome.PROPERTY_HELD, held.reason
    assert held.verified is False and held.strong_candidate is False


def test_compile_failure_stays_a_compile_failure_after_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_toolchain()
    svc, campaign = _campaign(
        tmp_path,
        monkeypatch,
        {"Stub.sol": STUB, "Acct.sol": BROKEN},
        contract="Acct",
        in_scope=["Acct"],
        out_of_scope=["EntrypointStub"],
    )
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert result["forge_runs"] >= 1
    print(f"compile_failure forge_runs={result['forge_runs']} outcomes={result['outcomes']}")
    assert result["observations"][0]["outcome"] == Outcome.COMPILE_FAILED.value
    assert result["outcomes"].get("property_violated", 0) == 0
    reloaded = BountyCampaignService()
    executions = reloaded.stateful_status(campaign.campaign_id)["executions"]
    assert executions
    assert all(item["outcome"] == "compile_failed" for item in executions.values())
    assert all(item["verified"] is False for item in executions.values())
    findings = reloaded.findings(campaign.campaign_id)["findings"]
    init = next(f for f in findings if f["detector"] == "aa.unprotected_account_initializer")
    assert init["verified"] is False
    assert init["poc_status"] != "local_harness_violation_unverified"
    assert init["candidate_strength"] != "corroborated_candidate"
