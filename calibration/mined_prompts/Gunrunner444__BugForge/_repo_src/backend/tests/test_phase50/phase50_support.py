"""Shared helpers for Phase 50 tests. Nothing here runs a tool or touches a network."""

from __future__ import annotations

from pathlib import Path

from app.discovery.bounty.campaign import BountyManifest
from app.discovery.engine import AnalysisRequest
from app.parsing.solidity_research import SemanticCandidate, build_research_model
from app.parsing.solidity_research_suite import SuiteResult, analyze_sources, run_suite

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase50"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def analyze(*names: str, families: tuple[str, ...] | None = None) -> SuiteResult:
    return analyze_sources({name: read(name) for name in names}, families)


def analyze_text(text: str, families: tuple[str, ...] | None = None) -> SuiteResult:
    return run_suite(build_research_model({"Case.sol": text}), families)


def detectors(result: SuiteResult) -> set[str]:
    return {item.detector for item in result.candidates}


def where(result: SuiteResult, detector: str) -> set[tuple[str, str]]:
    return {
        (item.contract, item.function.split("(")[0])
        for item in result.candidates
        if item.detector == detector
    }


def one(result: SuiteResult, detector: str, contract: str) -> SemanticCandidate:
    found = [c for c in result.candidates if c.detector == detector and c.contract == contract]
    assert found, f"{detector} not reported for {contract}: {sorted(detectors(result))}"
    return found[0]


MANIFEST_DATA: dict[str, object] = {
    "platform": "immunefi",
    "program_id": "demo-program",
    "program_name": "Demo Program",
    "program_url": "https://example.test/bounty/demo",
    "rules_version": "2026-01",
    "in_scope": [
        {"kind": "contract", "identifier": "SelfTrustMulticall"},
        {"kind": "contract", "identifier": "DonationVault"},
        {"kind": "path", "identifier": "caller_context_vulnerable.sol"},
        {"kind": "address", "identifier": "0x" + "11" * 20, "chain_id": "1"},
    ],
    "out_of_scope": [{"kind": "contract", "identifier": "BalanceAsDeposit"}],
    "known_issues": [
        {
            "issue_id": "KI-1",
            "title": "audited donation issue",
            "source": "audit",
            "contracts": ["DonationVault"],
            "detectors": ["accounting.donation_share_price"],
        }
    ],
    "poc_requirement": "required",
    "testing_mode": "fork_allowed",
    "repository": "https://example.test/repo.git",
    "source_commit": "abc1234",
    "deployments": [
        {
            "address": "0x" + "11" * 20,
            "chain_id": "1",
            "contract_name": "SelfTrustMulticall",
        }
    ],
    "chain_ids": ["1"],
    "fork": {
        "chain_id": "1",
        "source_ref": "https://rpc.example.test/redacted",
        "block": "19000000",
        "state_snapshot": "snap-19000000",
    },
    "compiler": {
        "version": "0.8.28",
        "optimizer": "enabled",
        "optimizer_runs": "200",
        "via_ir": "true",
        "evm_version": "cancun",
    },
    "impact_categories": [
        {
            "name": "Direct theft of funds",
            "severity": "critical",
            "weight": 90,
            "tags": ["value_transfer", "custody", "token_approvals"],
        },
        {
            "name": "Permission takeover",
            "severity": "high",
            "weight": 70,
            "tags": ["privileged_operations", "permissions_roles", "router_dispatch"],
        },
    ],
}


def manifest(**overrides: object) -> BountyManifest:
    data = {**MANIFEST_DATA, **overrides}
    return BountyManifest.from_dict(data)  # type: ignore[arg-type]


def request_for(
    root: Path, mani: BountyManifest, *, files: tuple[str, ...], contract: str = "", **extra: str
) -> tuple[AnalysisRequest, object]:
    identity = mani.to_campaign_identity(contract=contract, source_file=files[0] if files else "")
    request = AnalysisRequest(
        repo_root=root,
        language="solidity",
        target=contract or "campaign",
        contract=contract,
        source_file=files[0] if files else "",
        files=files,
        campaign_id=identity.campaign_id,
        extra={"project_id": "proj", **mani.request_extra(), **extra},
    )
    return request, identity
