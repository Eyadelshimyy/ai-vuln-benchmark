"""Phase 52 Slice C: RESEARCH_COVERAGE / RESEARCH_GAPS ledger, estimates, cost-aware plan."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from app.discovery.bounty.properties import build_property
from app.discovery.bounty.research_ledger import (
    Economic,
    Impact,
    Reachability,
    build_ledger,
    economic_of,
    impact_of,
    plan_actions,
    reachability_of,
)
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.stateful import ReplayMode, replay_bundle, replay_modes
from app.parsing.solidity_research import SemanticCandidate
from tests.test_phase51.phase51_support import FIXTURES, manifest
from tests.test_phase52.phase52_support import init_sequence, model_of

GUARDED = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Acct {
    address public owner;
    uint256 public fee;

    modifier onlyOwner() {
        require(msg.sender == owner, "owner");
        _;
    }

    function initialize(address newOwner) external {
        owner = newOwner;
    }

    function setFee(uint256 f) external onlyOwner {
        fee = f;
    }

    function _bump() internal {
        fee += 1;
    }
}
"""

ENGINES: list[dict[str, Any]] = [
    {"name": "foundry", "role": "stateful_execution", "status": "usable"},
    {"name": "echidna", "role": "property_engine", "status": "usable"},
    {"name": "medusa", "role": "property_engine", "status": "installed"},
    {"name": "fork_replay", "role": "pinned_fork_replay", "status": "blocked_by_policy"},
]


def _candidate(function: str, detector: str = "caller_context.unrestricted_privileged_write"):
    return SemanticCandidate(
        detector=detector,
        family="caller_context",
        title="t",
        summary="s",
        file="Acct.sol",
        line=1,
        contract="Acct",
        function=function,
    )


def _ledger(executions: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    sources = {"Acct.sol": GUARDED}
    model = model_of(sources)
    seq = init_sequence()
    spec = build_property(seq, model)
    base: dict[str, Any] = dict(
        models=(model,),
        candidates=(
            _candidate("setFee(uint256)"),
            _candidate("initialize(address)"),
            _candidate("initialize(address)", "aa.unprotected_account_initializer"),
        ),
        sequences=(seq,),
        specs={seq.sequence_id: spec},
        skipped={
            "caller_context.unrestricted_privileged_write@Acct.setFee(uint256)": "no template",
            "caller_context.unrestricted_privileged_write@Acct.initialize(address)": "no template",
        },
        executions=executions or {},
        engines=ENGINES,
    )
    base.update(kwargs)
    return build_ledger(**base)


def _by_kind(ledger: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [g for g in ledger["RESEARCH_GAPS"] if g["kind"] == kind]


def test_ledger_is_deterministic_and_ordered_by_priority() -> None:
    first, second = _ledger(), _ledger()
    assert first == second
    assert first["schema"] == "bugforge.research_ledger/1" and first["verified"] is False
    priorities = [g["priority"] for g in first["RESEARCH_GAPS"]]
    assert priorities == sorted(priorities, reverse=True)
    for gap in first["RESEARCH_GAPS"]:
        for key in ("importance", "reason", "missing_evidence", "cost", "next_engine"):
            assert gap[key] not in (None, "")
        assert {"reachability", "impact", "economic_feasibility"} <= set(gap["estimate"])


def test_privileged_ranks_below_an_external_caller_for_the_same_gap() -> None:
    ledger = _ledger()
    gaps = {g["subject"]: g for g in _by_kind(ledger, "no_sequence")}
    external = gaps["Acct.initialize(address)"]
    privileged = gaps["Acct.setFee(uint256)"]
    assert external["estimate"]["reachability"] == "external_caller"
    assert privileged["estimate"]["reachability"] == "privileged"
    assert "onlyOwner" in privileged["estimate"]["reachability_basis"]
    assert external["importance"] == privileged["importance"]  # same family and cost...
    assert external["priority"] > privileged["priority"]  # ...but reachability differs


def test_estimates_have_a_basis() -> None:
    model = model_of({"Acct.sol": GUARDED})
    internal = next(f for f in model.functions_of("Acct") if f.name == "_bump")
    assert reachability_of(model, internal)[0] is Reachability.INTERNAL_ONLY
    assert reachability_of(None, None)[0] is Reachability.UNKNOWN
    assert impact_of("initialization", None, None)[0] is Impact.HIGH
    assert impact_of("oracle_safety", None, None)[0] is Impact.MEDIUM
    tagged = SemanticCandidate(
        "x.y", "x", "t", "s", "Acct.sol", 1, "Acct", "f()", impact_tags=("custody",)
    )
    level, basis = impact_of("transient_storage", tagged, manifest())
    assert level is Impact.HIGH and "Direct theft of funds" in basis
    assert economic_of("authorization", None)[0] is Economic.NO_CAPITAL
    assert economic_of("oracle_safety", None)[0] is Economic.MARKET_DEPENDENT
    assert economic_of("solvency", None)[0] is Economic.CAPITAL_REQUIRED


def test_unexecuted_property_is_a_gap_with_a_local_cost() -> None:
    ledger = _ledger()
    (gap,) = _by_kind(ledger, "not_executed")
    assert gap["next_engine"] == "foundry:stateful_execute"
    assert gap["cost"]["local_runs"] == 2  # forge + one usable property engine
    assert ledger["RESEARCH_COVERAGE"]["by_declaration"] == {"property_under_test": 1}
    coverage = ledger["RESEARCH_COVERAGE"]
    assert coverage["functions_state_changing"] == 2  # initialize, setFee (not internal)
    assert coverage["engines"]["medusa"] == "installed"


def _execution(**overrides: Any) -> dict[str, Any]:
    seq = init_sequence()
    base = {
        "sequence_id": seq.sequence_id,
        "outcome": "property_violated",
        "reason": "x",
        "reason_code": "property_violated",
        "corroboration": "single_path",
        "check_paths": [
            {"path": "foundry:primary", "engine": "foundry", "verdict": "property_violated"}
        ],
        "stale": False,
    }
    base.update(overrides)
    return {seq.sequence_id: base}


def test_single_path_violation_needs_corroboration_and_fork_replay_stays_blocked() -> None:
    ledger = _ledger(_execution())
    (need,) = _by_kind(ledger, "needs_corroboration")
    assert need["next_engine"] == "property_engine:medusa"  # echidna usable, medusa not
    (replay,) = _by_kind(ledger, "deployment_replay")
    assert replay["blocked_by"].startswith("blocked_by_policy") and replay["priority"] == 0
    assert ledger["RESEARCH_COVERAGE"]["judged_by_engine"] == {"foundry": 1}
    assert not _by_kind(ledger, "not_executed")


def test_corroborated_disagreement_stale_and_inconclusive_map_to_distinct_gaps() -> None:
    assert not _by_kind(
        _ledger(_execution(corroboration="corroborated_candidate")), "needs_corroboration"
    )
    assert _by_kind(_ledger(_execution(corroboration="disagreement")), "contradiction")
    stale = _ledger(_execution(stale=True))
    assert _by_kind(stale, "stale_execution") and not _by_kind(stale, "deployment_replay")
    inconclusive = _ledger(
        _execution(outcome="inconclusive", reason_code="unknown_constructor_argument")
    )
    (gap,) = _by_kind(inconclusive, "inconclusive:unknown_constructor_argument")
    assert gap["next_engine"].startswith("fixture:constructor_arguments")
    assert gap["cost"]["human"] is True
    no_oracle = _ledger(_execution(outcome="sequence_executed_no_oracle"))
    assert _by_kind(no_oracle, "no_oracle")


def test_out_of_scope_contracts_are_not_ledgered() -> None:
    mani = manifest(
        in_scope=[{"kind": "contract", "identifier": "Other"}],
        out_of_scope=[{"kind": "contract", "identifier": "Acct"}],
    )
    ledger = _ledger(manifest=mani)
    assert ledger["RESEARCH_GAPS"] == []
    assert ledger["RESEARCH_COVERAGE"]["functions_state_changing"] == 0


# ---- planner ----------------------------------------------------------------------------------


BUDGET = {"attempts": 9, "engines": 9, "rounds": 9, "executions": 9}


def test_plan_never_schedules_blocked_or_human_work_and_respects_budgets() -> None:
    ledger = _ledger(_execution())
    plan = plan_actions(ledger, budget_remaining=BUDGET, local_runs_remaining=10)
    kinds = {a["kind"] for a in plan["planned"]}
    assert "deployment_replay" not in kinds and "no_sequence" not in kinds
    assert {a["kind"] for a in plan["blocked"]} == {"deployment_replay"}
    assert any(a["reason"].startswith("needs a human") for a in plan["deferred"])
    assert "needs_corroboration" in kinds
    planned = [a["priority"] for a in plan["planned"]]
    assert planned == sorted(planned, reverse=True) and plan["verified"] is False
    # a Phase 49 dimension that is exhausted defers the action instead of planning it
    starved = plan_actions(
        ledger, budget_remaining={**BUDGET, "engines": 0}, local_runs_remaining=10
    )
    assert "needs_corroboration" not in {a["kind"] for a in starved["planned"]}
    assert any("budget: engines" in a["reason"] for a in starved["deferred"])
    # the local run budget is respected too
    none_left = plan_actions(_ledger(), budget_remaining=BUDGET, local_runs_remaining=0)
    assert none_left["planned"] == []
    assert any(a["reason"] == "local run budget exhausted" for a in none_left["deferred"])


# ---- replay modes -----------------------------------------------------------------------------


def test_replay_modes_are_distinct_and_only_local_runs_here() -> None:
    modes = replay_modes()
    assert set(modes) == {m.value for m in ReplayMode}
    assert modes["deployment_replay"]["status"] == "unavailable"
    assert modes["pinned_fork_replay"]["status"] == "blocked_by_policy"
    assert modes["local_source_replay"]["does_not_establish"]
    refused = replay_bundle({}, {}, mode="pinned_fork_replay")
    assert refused["status"] == "blocked_by_policy" and refused["mode"] == "pinned_fork_replay"
    assert replay_bundle({}, {}, mode="deployment_replay")["status"] == "unavailable"
    assert replay_bundle({}, {}, mode="live")["status"] == "refused"


# ---- campaign wiring --------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name in ("aa_vulnerable.sol", "accounting_vulnerable.sol"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    return tmp_path


def test_campaign_ledger_plan_and_engines_without_running_anything(repo: Path) -> None:
    mani = manifest(
        in_scope=[
            {"kind": "contract", "identifier": "LooseAccount"},
            {"kind": "contract", "identifier": "RawAmountVault"},
        ],
        out_of_scope=[{"kind": "contract", "identifier": "DonationVault"}],
        known_issues=[],
    )
    svc = BountyCampaignService()
    campaign = svc.create(
        CampaignSpec(
            manifest=mani,
            repo_root=repo,
            contract="LooseAccount",
            files=("aa_vulnerable.sol", "accounting_vulnerable.sol"),
        )
    )
    ledger = svc.research_ledger(campaign.campaign_id)
    coverage = ledger["RESEARCH_COVERAGE"]
    assert coverage["sequences"] >= 5 and coverage["candidates"] >= coverage["sequences"]
    assert coverage["sequences_without_template"] >= 1
    assert all("DonationVault" not in g["subject"] for g in ledger["RESEARCH_GAPS"])
    assert _by_kind(ledger, "not_executed") and _by_kind(ledger, "no_sequence")
    assert svc.research_ledger(campaign.campaign_id)["ledger_digest"] == ledger["ledger_digest"]
    plan = svc.research_plan(campaign.campaign_id)
    assert plan["ledger_digest"] == ledger["ledger_digest"]
    assert all(a["next_engine"] == "foundry:stateful_execute" for a in plan["planned"])
    engines = svc.research_engines(campaign.campaign_id)
    current = {e["name"]: e["status"] for e in engines["current"]}
    # a GET runs no engine, so nothing is "usable" except the in-process analysis
    assert {n for n, s in current.items() if s == "usable"} == {"bugforge-static"}
    assert current["fork_replay"] == "blocked_by_policy"
    assert engines["last_run"] == []
