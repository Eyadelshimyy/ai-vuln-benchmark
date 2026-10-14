"""Phase 52 Slice D: the derived evidence graph and the finding quality model."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.discovery.bounty.evidence_graph import build_evidence_graph
from app.discovery.bounty.findings import quality_of
from app.discovery.bounty.properties import build_property
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.parsing.solidity_research_suite import run_suite
from tests.test_phase51.phase51_support import FIXTURES, manifest
from tests.test_phase52.phase52_support import (
    VULNERABLE_INIT,
    init_sequence,
    model_of,
    requires_forge,
)


def _record(*verdicts: tuple[str, str], stale: bool = False) -> dict[str, object]:
    return {
        "outcome": "property_violated",
        "identity_status": "matched",
        "replay_mode": "local_source_replay",
        "stale": stale,
        "bundle_id": "rb_1",
        "check_paths": [
            {"path": path, "engine": path.split(":")[0], "material": True, "verdict": verdict}
            for path, verdict in verdicts
        ],
    }


def _graph(record: dict[str, object] | None, phase49=()):  # type: ignore[no-untyped-def]
    sources = {"Acct.sol": VULNERABLE_INIT}
    model = model_of(sources)
    seq = init_sequence()
    spec = build_property(seq, model)
    return build_evidence_graph(
        candidates=run_suite(model).candidates,
        sequences=[seq],
        specs={seq.sequence_id: spec},
        executions={seq.sequence_id: record} if record else {},
        phase49_contradictions=phase49,
    )


def test_every_edge_has_provenance_and_nothing_is_verified() -> None:
    graph = _graph(
        _record(("foundry", "property_violated"), ("echidna:property", "property_violated"))
    )
    assert graph["edges"] and graph["nodes"]
    kinds = {e["kind"] for e in graph["edges"]}
    assert {"declares", "executed_as", "judged_by", "reproduced_by"} <= kinds
    assert all(e["provenance"] for e in graph["edges"])
    ids = {n["id"] for n in graph["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    assert graph["verified"] is False
    assert all(n.get("verified", False) is False for n in graph["nodes"])
    assert graph["contradictions"] == []
    assert "not a second evidence store" in graph["note"]


def test_a_disagreeing_path_is_a_visible_open_contradiction() -> None:
    graph = _graph(_record(("foundry", "property_violated"), ("medusa:property", "property_held")))
    assert [c["kind"] for c in graph["contradictions"]] == ["independent_paths_disagree"]
    assert graph["contradictions"][0]["status"] == "open"
    assert any(e["kind"] == "contradicts" for e in graph["edges"])


def test_stale_executions_and_phase49_contradictions_are_listed() -> None:
    graph = _graph(
        _record(("foundry", "property_violated"), stale=True),
        phase49=[{"id": "c1", "kind": "verdict_conflict", "status": "open", "identity": "x"}],
    )
    kinds = {c["kind"] for c in graph["contradictions"]}
    assert kinds == {"stale_execution", "phase49:verdict_conflict"}
    assert any(n["kind"] == "phase49_contradiction" for n in graph["nodes"])


def test_graph_is_deterministic() -> None:
    record = _record(("foundry", "property_violated"))
    assert _graph(record)["digest"] == _graph(record)["digest"]
    assert _graph(record)["digest"] != _graph(None)["digest"]


@pytest.mark.parametrize(
    ("strength", "corroboration", "expected"),
    [
        ("static_candidate", "", "none"),
        ("weak_execution_only", "", "execution_only"),
        ("strong_candidate", "single_path", "single_path_execution"),
        ("corroborated_candidate", "corroborated", "corroborated"),
        ("corroborated_candidate", "disagreement", "contradicted"),
    ],
)
def test_quality_axes_are_separate_and_never_verified(
    strength: str, corroboration: str, expected: str
) -> None:
    quality = quality_of(strength, corroboration)
    assert quality["candidate"] == "static_candidate"
    assert quality["corroboration"] == expected
    assert quality["verification"] == "unverified" and quality["verified"] is False


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copy(FIXTURES / "aa_vulnerable.sol", tmp_path / "aa_vulnerable.sol")
    return tmp_path


def _campaign(svc: BountyCampaignService, root: Path):  # type: ignore[no-untyped-def]
    mani = manifest(
        in_scope=[{"kind": "contract", "identifier": "LooseAccount"}],
        out_of_scope=[],
        known_issues=[],
    )
    return svc.create(
        CampaignSpec(
            manifest=mani, repo_root=root, contract="LooseAccount", files=("aa_vulnerable.sol",)
        )
    )


def test_service_graph_before_execution_links_candidates_to_sequences(repo: Path) -> None:
    svc = BountyCampaignService()
    campaign = _campaign(svc, repo)
    graph = svc.evidence_graph(campaign.campaign_id)
    assert graph["campaign_id"] == campaign.campaign_id
    kinds = {n["kind"] for n in graph["nodes"]}
    assert {"static_candidate", "vfcs_sequence", "property"} <= kinds
    assert "execution" not in kinds
    assert any(e["kind"] == "derived_sequence" for e in graph["edges"])
    for finding in svc.findings(campaign.campaign_id)["findings"]:
        assert finding["quality"]["verification"] == "unverified"


@requires_forge
def test_service_graph_after_execution_shows_paths_and_bundles(repo: Path) -> None:
    svc = BountyCampaignService()
    campaign = _campaign(svc, repo)
    svc.stateful_execute(campaign.campaign_id, rounds=1)
    graph = svc.evidence_graph(campaign.campaign_id)
    kinds = {n["kind"] for n in graph["nodes"]}
    assert {"execution", "check_path", "repro_bundle"} <= kinds
    judged = [e for e in graph["edges"] if e["kind"] == "judged_by"]
    assert judged and all(e["provenance"].startswith("engine:") for e in judged)
    corroborated = [
        f
        for f in svc.findings(campaign.campaign_id)["findings"]
        if f["candidate_strength"] == "corroborated_candidate"
    ]
    assert corroborated
    assert all(f["quality"]["corroboration"] == "corroborated" for f in corroborated)
    assert all(f["quality"]["verified"] is False for f in corroborated)
