from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import pytest

import procs.harness.record as record
from procs.harness.record import check_immutability, normalized
from procs.harness.record import write_transcripts
from procs.harness.replay import normalize


def test_partial_record_preserves_other_index_entries(tmp_path) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps(
            [
                {"module": "plans", "scenario": "PLANS-001", "rules": ["PLANS-001"]},
                {"module": "rating", "scenario": "RATING-001", "rules": ["RATING-001"]},
            ]
        )
    )
    write_transcripts(
        [
            {
                "module": "plans",
                "scenario": "PLANS-001",
                "rules": ["PLANS-001"],
                "business_fields": {},
                "probes": {},
            }
        ],
        "current",
        tmp_path,
    )
    index = json.loads((tmp_path / "index.json").read_text())
    assert {(item["module"], item["scenario"]) for item in index} == {
        ("plans", "PLANS-001"),
        ("rating", "RATING-001"),
    }


def test_typed_capture_round_trips_with_replay_normalization() -> None:
    assert normalized("1", "integer") == normalize("1", "integer") == 1
    assert (
        normalized("2026-02-28", "date")
        == normalize("2026-02-28", "date")
        == "2026-02-28"
    )
    assert normalized(Decimal("2.2506")) == normalize("2.2506") == "2.2506"


def test_unchanged_source_rerecord_requires_audited_reason(tmp_path) -> None:
    scenarios = [{"module": "plans", "id": "PLANS-001"}]
    path = tmp_path / "plans" / "PLANS-001.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"source_sha": "same"}))
    with pytest.raises(RuntimeError, match="harness-change"):
        check_immutability(scenarios, "same", True, None, tmp_path)
    check_immutability(scenarios, "same", True, "harness-change", tmp_path)
    check_immutability(scenarios, "same", True, "scenario-redesign", tmp_path)


def test_fixture_change_permits_rerecord_without_audited_reason(tmp_path) -> None:
    scenarios = [{"module": "plans", "id": "PLANS-001"}]
    path = tmp_path / "plans" / "PLANS-001.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"source_sha": "same", "fixture_sha": "old"}))
    check_immutability(scenarios, "same", True, None, tmp_path, "new")


def test_partial_rerecord_then_all_module_rerecord_is_allowed(tmp_path) -> None:
    scenarios = [
        {"module": "plans", "id": "PLANS-001"},
        {"module": "rating", "id": "RATING-001"},
    ]
    for scenario in scenarios:
        path = tmp_path / scenario["module"] / f"{scenario['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source_sha": "old", "fixture_sha": "fixture"}))

    check_immutability([scenarios[0]], "new", True, None, tmp_path, "fixture")
    (tmp_path / "plans" / "PLANS-001.json").write_text(
        json.dumps({"source_sha": "new", "fixture_sha": "fixture"})
    )
    check_immutability(scenarios, "new", True, None, tmp_path, "fixture")


def test_empty_recording_does_not_rewrite_index_or_fingerprint(tmp_path) -> None:
    index = tmp_path / "index.json"
    source = tmp_path / "SOURCE_SHA"
    index.write_text("original index\n")
    source.write_text("original source\n")
    record.write_transcripts([], "new source", tmp_path)
    assert index.read_text() == "original index\n"
    assert source.read_text() == "original source\n"


def test_unknown_module_fails_before_recording(monkeypatch) -> None:
    monkeypatch.setattr(record, "load_scenarios", lambda _module: [])
    monkeypatch.setattr(sys, "argv", ["record.py", "--module", "typo"])
    assert record.main() == record.SCENARIO_FAILED
