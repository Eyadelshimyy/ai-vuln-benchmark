"""Phase 52 Slice D: evidence-triggered protocol modules (ERC-4337, EIP-7702, v4 hooks,
bridges, governance). A module runs only when the code gives evidence for it, and
every module states whether it ran, has no detector, or was not triggered."""

from __future__ import annotations

from pathlib import Path

from app.discovery.bounty.properties import build_property
from app.discovery.bounty.stateful import Outcome, StatefulExecutor
from app.discovery.bounty.vfcs import generate
from app.parsing.solidity_modules import (
    analyze_protocol_modules,
    module_triggers,
    modules_present,
)
from app.parsing.solidity_research import build_research_model
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase52.phase52_support import VULNERABLE_INIT, requires_forge

HISTORY = Path(__file__).resolve().parents[1] / "fixtures" / "phase52_history"


def _model(name: str):  # type: ignore[no-untyped-def]
    return build_research_model({name: (HISTORY / name).read_text()})


def _status(model) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {s.module: s for s in module_triggers(model)}


def test_modules_are_not_triggered_without_evidence() -> None:
    model = build_research_model({"Acct.sol": VULNERABLE_INIT})
    statuses = _status(model)
    assert set(statuses) == {"erc4337", "eip7702", "uniswap_v4_hooks", "bridge", "governance"}
    assert all(s.status == "not_triggered" and not s.evidence for s in statuses.values())  # type: ignore[attr-defined]
    assert not modules_present(model)
    assert analyze_protocol_modules(model) == []
    suite = run_suite(model)
    assert "protocol_modules" in suite.families_skipped
    assert "protocol_modules" not in suite.families_run


def test_bridge_receiver_triggers_with_evidence_and_detector_fires_only_on_the_open_one() -> None:
    vuln = _model("bridge_receiver_vulnerable.sol")
    bridge = _status(vuln)["bridge"]
    assert bridge.status == "ran" and bridge.evidence  # type: ignore[attr-defined]
    assert any("receiveMessage" in e[2] for e in bridge.evidence)  # type: ignore[attr-defined]
    found = analyze_protocol_modules(vuln)
    assert [c.detector for c in found] == ["bridge.receiver_without_endpoint_check"]
    assert found[0].contract == "OpenReceiver"
    safe = _model("bridge_receiver_safe.sol")
    assert _status(safe)["bridge"].status == "ran"  # type: ignore[attr-defined]
    assert analyze_protocol_modules(safe) == []


def test_v4_hook_callback_detector_and_safe_twin() -> None:
    vuln = _model("v4_hook_vulnerable.sol")
    assert _status(vuln)["uniswap_v4_hooks"].status == "ran"  # type: ignore[attr-defined]
    found = analyze_protocol_modules(vuln)
    assert [c.detector for c in found] == ["v4_hooks.callback_without_pool_manager_check"]
    assert analyze_protocol_modules(_model("v4_hook_safe.sol")) == []


def test_governance_is_reported_as_triggered_without_a_detector() -> None:
    source = """pragma solidity ^0.8.20;
contract Gov {
    function propose(bytes calldata data) external returns (uint256) { return data.length; }
    function castVote(uint256 id, uint8 support) external {}
    function execute(uint256 id) external {}
}
"""
    model = build_research_model({"Gov.sol": source})
    gov = _status(model)["governance"]
    assert gov.triggered and gov.status == "triggered_no_detector"  # type: ignore[attr-defined]
    assert gov.detectors == () and gov.evidence  # type: ignore[attr-defined]
    assert analyze_protocol_modules(model) == []  # no claim without a detector


def test_a_single_governance_name_is_not_evidence() -> None:
    model = build_research_model(
        {"X.sol": "pragma solidity ^0.8.20;\ncontract X { function execute() external {} }"}
    )
    assert _status(model)["governance"].status == "not_triggered"  # type: ignore[attr-defined]


def test_suite_runs_module_family_only_when_triggered() -> None:
    suite = run_suite(_model("bridge_receiver_vulnerable.sol"))
    assert "protocol_modules" in suite.families_run
    assert any(c.family == "protocol_modules" for c in suite.candidates)
    assert all(c.confidence in {"low", "medium", "high"} for c in suite.candidates)


@requires_forge
def test_module_candidates_execute_to_violated_and_safe_twin_holds() -> None:
    executor = StatefulExecutor()
    for stem, detector in (
        ("bridge_receiver", "bridge.receiver_without_endpoint_check"),
        ("v4_hook", "v4_hooks.callback_without_pool_manager_check"),
    ):
        outcomes = {}
        vuln_model = _model(f"{stem}_vulnerable.sol")
        candidate = next(c for c in analyze_protocol_modules(vuln_model) if c.detector == detector)
        for role in ("vulnerable", "safe"):
            name = f"{stem}_{role}.sol"
            model = _model(name)
            from dataclasses import replace

            twin = replace(candidate, file=name)
            text = (HISTORY / name).read_text()
            role_sources = {name: text}
            sequence = generate(model, [twin], sources=role_sources).sequences[0]
            spec = build_property(sequence, model, twin)
            obs = executor.execute(sequence, model, role_sources, spec=spec)
            outcomes[role] = obs.outcome
        assert outcomes == {
            "vulnerable": Outcome.PROPERTY_VIOLATED,
            "safe": Outcome.PROPERTY_HELD,
        }, stem
