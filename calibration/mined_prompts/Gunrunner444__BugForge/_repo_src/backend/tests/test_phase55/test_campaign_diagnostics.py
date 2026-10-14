"""Coverage summary, cache identity, minimization, and the registry matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.discovery.bounty.service as service_module
from app.core.config import get_settings
from app.discovery.bounty.benchmark import pipeline_coverage_matrix
from app.discovery.bounty.research_ledger import campaign_coverage
from app.discovery.bounty.service import BountyCampaignService, CampaignSpec
from app.discovery.bounty.vfcs import VfcsCall, minimize
from tests.test_phase51.phase51_support import manifest
from tests.test_phase52.phase52_support import VULNERABLE_INIT, init_sequence

HISTORY = Path(__file__).resolve().parents[1] / "fixtures" / "phase52_history"


def test_zero_run_blockers_name_the_stage_that_stopped() -> None:
    assert campaign_coverage(files_selected=0).get("blocker") == "no files selected"
    assert (
        campaign_coverage(files_selected=2, files_read=0)["blocker"]
        == "no relevant source was successfully read"
    )
    no_candidate = campaign_coverage(files_selected=1, files_read=1, candidates=0, sequences=0)
    assert no_candidate["blocker"] == "no detector emitted a candidate"
    assert no_candidate["dynamic_foundry_execution_demonstrated"] is False
    skipped = {
        "bridge.receiver_without_endpoint_check@Bridge.lzReceive": "no sequence template applies"
    }
    assert (
        campaign_coverage(
            files_selected=1, files_read=1, candidates=1, sequences=0, skipped=skipped
        )["blocker"]
        == "candidates had no supported sequence template"
    )
    identity = {"aa.x@Acct.initialize": "model source attestation does not match"}
    assert (
        "identity"
        in campaign_coverage(
            files_selected=1, files_read=1, candidates=1, sequences=0, skipped=identity
        )["blocker"]
    )
    waiting = campaign_coverage(files_selected=1, files_read=1, candidates=1, sequences=1)
    assert waiting["blocker"] == "no dynamic Foundry execution was demonstrated"
    missing_tool = campaign_coverage(
        files_selected=1,
        files_read=1,
        candidates=1,
        sequences=1,
        executions={"vf": {"outcome": "unavailable"}},
        forge_runs=0,
    )
    assert missing_tool["blocker"] == "Foundry or solc is unavailable"
    mismatch = campaign_coverage(
        files_selected=1,
        files_read=1,
        candidates=1,
        sequences=1,
        executions={"vf": {"outcome": "identity_mismatch"}},
        forge_runs=0,
    )
    assert "identity" in mismatch["blocker"]
    inconclusive = campaign_coverage(
        files_selected=1,
        files_read=1,
        candidates=1,
        sequences=1,
        executions={"vf": {"outcome": "inconclusive"}},
        forge_runs=0,
    )
    assert "inconclusive" in inconclusive["blocker"]
    ran = campaign_coverage(
        files_selected=1,
        files_read=1,
        candidates=1,
        sequences=1,
        executions={"vf": {"outcome": "property_violated"}},
        forge_runs=2,
    )
    assert ran["blocker"] == ""
    assert ran["dynamic_foundry_execution_demonstrated"] is True
    assert ran["local_results_are_not_production_verification"] is True
    assert ran["verified"] is False
    assert ran["forge_runs"] == 2


def test_minimization_rejects_a_reduction_that_loses_the_property() -> None:
    sequence = init_sequence()
    kept = {call.identity for call in sequence.calls}

    def evaluator(calls: tuple[VfcsCall, ...]) -> bool:
        return [call.role for call in calls] == [call.role for call in sequence.calls] and {
            call.identity for call in calls
        } == kept

    result = minimize(sequence, evaluator, evaluator_name="test")
    assert result.completed is True
    assert result.minimized == sequence.calls
    assert result.preserved_property == sequence.property_under_test
    assert result.removed == ()


def test_registry_matrix_marks_missing_fixtures_untested() -> None:
    matrix = pipeline_coverage_matrix(HISTORY)
    assert matrix["verified"] is False
    assert matrix["rows"]
    assert all(row["foundry_execution"] == "untested" for row in matrix["rows"])
    assert all(row["persistence"] == "untested" for row in matrix["rows"])
    present = [row for row in matrix["rows"] if row["positive_fixture"] == "present"]
    missing = [row for row in matrix["rows"] if row["positive_fixture"] == "untested"]
    assert present and missing
    assert "untested" not in {row["positive_fixture"] for row in present}
    assert all(row["positive_fixture"] != "pass" for row in matrix["rows"])


def test_unchanged_analysis_is_reused_and_a_changed_snapshot_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "Acct.sol"
    source.write_text(VULNERABLE_INIT, encoding="utf-8")
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
                source_commit="fixture-phase55",
            ),
            repo_root=repo,
            contract="Acct",
            source_file="Acct.sol",
            files=("Acct.sol",),
        )
    )
    first = svc._analysis(campaign)
    second = svc._analysis(campaign)
    assert second is first
    assert first.model is not None
    assert svc.analysis_stale(campaign) is False
    original = dict(first.model.source_hashes)["Acct.sol"]
    source.write_text(VULNERABLE_INIT + "\n", encoding="utf-8")
    cached = svc._analysis(campaign)
    assert cached is first
    assert svc.analysis_stale(campaign) is True
    fresh = svc.reanalyze(campaign.campaign_id)
    assert fresh is not first and fresh.model is not None
    assert svc.analysis_stale(campaign) is False
    updated = dict(fresh.model.source_hashes)["Acct.sol"]
    assert updated != original
    if first.sequences and fresh.sequences:
        assert dict(fresh.sequences[0].identity.source_hashes)["Acct.sol"] == updated
        assert dict(first.sequences[0].identity.source_hashes)["Acct.sol"] == original
    summary = svc.research_ledger(campaign.campaign_id)["campaign_summary"]
    assert summary["analysis_stale"] is False
    assert summary["verified"] is False
    assert summary["dynamic_foundry_execution_demonstrated"] is False
