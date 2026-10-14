"""Phase 52 hardening, Slice A: campaign wiring, fail-closed persistence, durable bundles."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.discovery.bounty.service import (
    BountyCampaignService,
    CampaignControlError,
    CampaignSpec,
    _merge_stateful,
)
from app.services.bounty_campaign_store import CampaignConflictError, CampaignPersistenceError
from tests.test_phase51.phase51_support import FIXTURES, manifest
from tests.test_phase52.phase52_support import requires_forge


def _spec(root: Path, **overrides: object) -> CampaignSpec:
    mani = manifest(
        in_scope=[
            {"kind": "contract", "identifier": "LooseAccount"},
            {"kind": "contract", "identifier": "RawAmountVault"},
        ],
        out_of_scope=[{"kind": "contract", "identifier": "DonationVault"}],
        known_issues=[],
    )
    base: dict[str, object] = dict(
        manifest=mani,
        repo_root=root,
        contract="LooseAccount",
        files=("aa_vulnerable.sol", "accounting_vulnerable.sol"),
    )
    base.update(overrides)
    return CampaignSpec(**base)  # type: ignore[arg-type]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for name in ("aa_vulnerable.sol", "accounting_vulnerable.sol"):
        shutil.copy(FIXTURES / name, tmp_path / name)
    return tmp_path


def _merge(key: str, value: object):  # type: ignore[no-untyped-def]
    def merge(artifacts: dict[str, object]) -> list[str]:
        artifacts[key] = value
        return []

    return merge


def test_persistence_reconciles_a_conflict_without_overwriting_the_other_writer(
    repo: Path,
) -> None:
    first = BountyCampaignService()
    campaign = first.create(_spec(repo))
    second = BountyCampaignService()
    second.pause(campaign.campaign_id, reason="operator pause elsewhere")
    state = first._persist_artifacts(campaign, _merge("probe", {"x": 1}))
    assert state["status"] == "persisted" and state["reconciled"] is True
    fresh = BountyCampaignService().get(campaign.campaign_id)
    assert fresh.record.artifacts["probe"] == {"x": 1}
    assert fresh.record.control == "paused"  # the other writer's change survived


def test_persistence_conflict_and_unavailable_are_reported(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec(repo))

    def always_conflict(record):  # type: ignore[no-untyped-def]
        raise CampaignConflictError("changed")

    monkeypatch.setattr(svc._records, "update", always_conflict)
    state = svc._persist_artifacts(campaign, _merge("a", 1))
    assert state["status"] == "persistence_conflict"
    campaign = svc.get(campaign.campaign_id)

    def down(record):  # type: ignore[no-untyped-def]
        raise CampaignPersistenceError("database is unavailable")

    monkeypatch.setattr(svc._records, "update", down)
    before = dict(campaign.record.artifacts)
    state = svc._persist_artifacts(campaign, _merge("b", 2))
    assert state["status"] == "persistence_unavailable"
    assert campaign.record.artifacts == before  # nothing claimed that was not written


def test_merge_is_idempotent_and_never_overwrites_bundles() -> None:
    artifacts: dict[str, object] = {}
    bundle = {"bundle_id": "rb_1", "artifact_hashes": {"h": "1"}, "source_hashes": {"a": "d1"}}
    kwargs = dict(
        executions={"vf": {"outcome": "property_held"}},
        bundles={"rb_1": bundle},
        blobs={"d1": "text"},
        feedback=({"key": "foundry:near_miss:vf:1"},),
        run={"at": "t"},
    )
    assert _merge_stateful(artifacts, **kwargs) == []  # type: ignore[arg-type]
    assert _merge_stateful(artifacts, **kwargs) == []  # type: ignore[arg-type]
    stateful = artifacts["stateful"]
    assert len(stateful["runs"]) == 1 and len(stateful["feedback"]) == 1  # type: ignore[index]
    tampered = {**bundle, "artifact_hashes": {"h": "2"}}
    refused = _merge_stateful(artifacts, **{**kwargs, "bundles": {"rb_1": tampered}})  # type: ignore[arg-type]
    assert refused == ["rb_1"]
    assert artifacts["repro_bundles"]["rb_1"]["artifact_hashes"] == {"h": "1"}  # type: ignore[index]
    changed = {**kwargs, "executions": {"vf": {"outcome": "property_violated"}}}
    _merge_stateful(artifacts, **changed)  # type: ignore[arg-type]
    record = artifacts["stateful"]["executions"]["vf"]  # type: ignore[index]
    assert record["outcome"] == "property_violated"
    assert record["history"][-1]["outcome"] == "property_held"


def test_stateful_execute_refuses_paused_and_out_of_scope(repo: Path) -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec(repo))
    svc.pause(campaign.campaign_id)
    with pytest.raises(CampaignControlError):
        svc.stateful_execute(campaign.campaign_id)
    blocked = svc.create(_spec(repo, contract="DonationVault"))
    with pytest.raises(CampaignControlError):
        svc.stateful_execute(blocked.campaign_id)


@requires_forge
def test_end_to_end_campaign_execution_is_durable_and_drives_findings(repo: Path) -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec(repo))
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    assert result["available"] is True and result["persistence"]["status"] == "persisted"
    assert result["outcomes"].get("property_violated", 0) >= 3
    skipped = {item["sequence_id"] for item in result["skipped"]}
    assert skipped  # the out-of-scope DonationVault sequence never ran
    assert all("bundle_dir" not in key for key in result)  # no temp-dir bundles

    # durable: a fresh service (as after a restart) sees executions and bundles
    status = BountyCampaignService().stateful_status(campaign.campaign_id)
    assert status["executions"] and status["bundles"]
    violated = [b for b, v in status["bundles"].items() if v["outcome"] == "property_violated"]
    full = BountyCampaignService().repro_bundle(campaign.campaign_id, violated[0])
    assert full["self_contained"] is True

    findings = {
        (f["detector"], f["contract"]): f for f in svc.findings(campaign.campaign_id)["findings"]
    }
    init = findings[("aa.unprotected_account_initializer", "LooseAccount")]
    assert init["candidate_strength"] == "corroborated_candidate"
    assert init["poc_status"] == "local_harness_violation_unverified"
    assert init["verified"] is False and init["bundle_ids"]
    never = findings[("aa.signature_result_ignored", "LooseAccount")]
    assert never["candidate_strength"] == "static_candidate"

    # the engine registry the run used is persisted with honest statuses
    engines = {e["name"]: e for e in status["engines"]}
    assert engines["fork_replay"]["status"] == "blocked_by_policy"
    assert engines["foundry"]["status"] == "usable"
    for name in ("echidna", "medusa"):
        assert engines[name]["status"] in {"usable", "installed", "unavailable"}
    # every executed violation lists its check paths; a usable engine was really run
    executed = [e for e in status["executions"].values() if e["outcome"] == "property_violated"]
    assert executed and all(e["check_paths"] for e in executed)
    for name, entry in engines.items():
        if entry.get("role") == "property_engine" and entry["status"] == "usable":
            ran = [p for e in executed for p in e["check_paths"] if p["path"] == f"{name}:property"]
            assert ran and all(p["verdict"] != "not_evaluated" for p in ran)


@requires_forge
def test_a_source_changed_after_analysis_is_an_identity_mismatch(repo: Path) -> None:
    svc = BountyCampaignService()
    campaign = svc.create(_spec(repo, files=("aa_vulnerable.sol",)))
    svc.findings(campaign.campaign_id)  # analysis snapshot is taken here
    path = repo / "aa_vulnerable.sol"
    path.write_text(path.read_text() + "\n// edited after analysis\n")
    result = svc.stateful_execute(campaign.campaign_id, rounds=1)
    outcomes = {o["outcome"] for o in result["observations"]}
    assert outcomes == {"identity_mismatch"}
    assert result["forge_runs"] == 0
    assert result["outcomes"].get("property_violated", 0) == 0
    assert all(
        item["verified"] is False and item["strong_candidate"] is False
        for item in result["observations"]
    )
