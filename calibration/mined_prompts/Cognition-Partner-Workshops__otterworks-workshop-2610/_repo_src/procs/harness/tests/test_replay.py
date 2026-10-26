from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from procs.harness.replay import (
    classify_transcript,
    compare,
    grade_response,
    normalize,
    source_matches,
    stale_fixture_transcripts,
    stale_transcripts,
    write_report,
)
import procs.harness.replay as replay


def test_pending_module_is_skip() -> None:
    transcript = {"module": "rating", "scenario": "RATING-001"}
    routes = {"rating": {"status": "pending"}}
    assert classify_transcript(transcript, routes) == "SKIP"


def test_unrecognized_module_status_is_contract_error() -> None:
    transcript = {"module": "rating", "scenario": "RATING-001"}
    routes = {"rating": {"status": "draft"}}
    assert classify_transcript(transcript, routes) == "CONTRACT_ERROR"


def test_source_sha_mismatch_is_detected(monkeypatch) -> None:
    monkeypatch.setattr("procs.harness.replay.source_sha", lambda: "actual")
    assert not source_matches("recorded")


def test_stale_transcript_source_is_detected() -> None:
    errors = stale_transcripts(
        [{"scenario": "RATING-001", "source_sha": "old"}],
        "current",
    )
    assert errors and "RATING-001" in errors[0]


def test_stale_fixture_digest_is_detected() -> None:
    errors = stale_fixture_transcripts(
        [{"scenario": "RATING-001", "fixture_sha": "old"}],
        "current",
    )
    assert errors and "RATING-001" in errors[0]


def test_business_field_mismatch_is_diagnostic() -> None:
    transcript = {"scenario": "PLANS-001", "business_fields": {"code": "STARTER"}, "probes": {}}
    contract = {"response": {"business_fields": {"code": {"json_path": "$.code"}}, "probes": {}}}
    failures, errors = compare(transcript, contract, {"code": "GROWTH"})
    assert not errors
    assert failures == [{"kind": "field", "name": "code", "expected": "STARTER", "actual": "GROWTH"}]


def test_probe_mismatch_is_diagnostic() -> None:
    transcript = {"scenario": "PLANS-004", "business_fields": {}, "probes": {"rows": [{"id": 1}]}}
    contract = {"response": {"business_fields": {}, "probes": {"rows": {"json_path": "$.rows"}}}}
    failures, errors = compare(transcript, contract, {"rows": [{"id": 2}]})
    assert not errors
    assert failures[0]["kind"] == "probe"


def test_unmapped_business_field_and_probe_are_contract_errors() -> None:
    transcript = {
        "scenario": "PLANS-001",
        "business_fields": {"unmapped": "value"},
        "probes": {"missing_probe": "value"},
    }
    failures, errors = compare(transcript, {"response": {"business_fields": {}, "probes": {}}}, {})
    assert not failures
    assert "business field unmapped" in errors[0]
    assert "probe missing_probe" in errors[1]


def test_structured_rows_preserve_recorded_order_and_normalize() -> None:
    contract = {
        "response": {
            "business_fields": {
                "rows": {"json_path": "$.rows", "type": "rows"}
            },
            "probes": {},
        }
    }
    transcript = {"scenario": "x", "business_fields": {"rows": [{"id": "1"}]}, "probes": {}}
    failures, errors = compare(transcript, contract, {"rows": [{"id": "1"}]})
    assert not failures
    assert not errors
    assert normalize({"b": 2, "a": 1}) == {"a": 1, "b": 2}


def test_structured_rows_wrong_order_is_a_failure() -> None:
    contract = {
        "response": {
            "business_fields": {"rows": {"json_path": "$.rows", "type": "rows"}},
            "probes": {},
        }
    }
    transcript = {
        "scenario": "x",
        "business_fields": {"rows": [{"id": "1"}, {"id": "2"}]},
        "probes": {},
    }
    failures, errors = compare(
        transcript, contract, {"rows": [{"id": "2"}, {"id": "1"}]}
    )
    assert not errors
    assert failures[0]["kind"] == "field"


def test_structured_rows_missing_column_is_a_reported_failure() -> None:
    contract = {
        "response": {
            "business_fields": {"rows": {"json_path": "$.rows", "type": "rows"}},
            "probes": {},
        }
    }
    transcript = {
        "scenario": "x",
        "business_fields": {"rows": [{"id": "1", "code": "STARTER"}]},
        "probes": {},
    }
    failures, errors = compare(transcript, contract, {"rows": [{"id": "1"}]})
    assert not errors
    assert failures[0]["kind"] == "field"


def test_contract_errors_are_written_to_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("procs.harness.replay.REPORT_DIR", tmp_path)
    write_report("sha", [], ["PLANS-001: business field missing"])
    assert "Contract error" in (tmp_path / "parity.md").read_text()


def test_non_200_response_is_reported_without_comparing_error_body(tmp_path, monkeypatch) -> None:
    transcript = {
        "scenario": "PLANS-001",
        "business_fields": {"code": "STARTER"},
        "probes": {},
    }
    contract = {
        "response": {
            "business_fields": {"code": {"json_path": "$.code"}},
            "probes": {},
        }
    }
    failures, errors = grade_response(transcript, contract, 503, {"detail": "unavailable"})
    assert failures == [{"kind": "status", "name": "status", "expected": 200, "actual": 503}]
    assert not errors
    monkeypatch.setattr("procs.harness.replay.REPORT_DIR", tmp_path)
    write_report(
        "sha",
        [{"module": "plans", "scenario": "PLANS-001", "status": "FAIL", "failures": failures}],
        [],
    )
    assert "actual `503`" in (tmp_path / "parity.md").read_text()


def test_missing_mapped_field_is_a_reported_failure(tmp_path, monkeypatch) -> None:
    transcript = {"scenario": "PLANS-001", "business_fields": {"code": "STARTER"}, "probes": {}}
    contract = {
        "response": {"business_fields": {"code": {"json_path": "$.code"}}, "probes": {}}
    }
    failures, errors = compare(transcript, contract, {"name": "missing"})
    assert not errors
    assert failures[0]["actual"].startswith("<unresolvable $.code:")
    monkeypatch.setattr("procs.harness.replay.REPORT_DIR", tmp_path)
    write_report("sha", [{"module": "plans", "scenario": "PLANS-001", "status": "FAIL", "failures": failures}], [])
    assert "unresolvable" in (tmp_path / "parity.md").read_text()


def test_scalar_payload_under_mapped_path_is_a_reported_failure(tmp_path, monkeypatch) -> None:
    transcript = {"scenario": "PLANS-001", "business_fields": {"code": "STARTER"}, "probes": {}}
    contract = {
        "response": {"business_fields": {"code": {"json_path": "$.code.value"}}, "probes": {}}
    }
    failures, errors = compare(transcript, contract, {"code": None})
    assert not errors
    assert failures[0]["actual"].startswith("<unresolvable $.code.value:")
    monkeypatch.setattr("procs.harness.replay.REPORT_DIR", tmp_path)
    write_report("sha", [{"module": "plans", "scenario": "PLANS-001", "status": "FAIL", "failures": failures}], [])
    assert (tmp_path / "parity.md").exists()


def test_collect_on_non_list_is_a_reported_failure(tmp_path, monkeypatch) -> None:
    transcript = {"scenario": "PLANS-001", "business_fields": {"codes": ["STARTER"]}, "probes": {}}
    contract = {
        "response": {
            "business_fields": {"codes": {"json_path": "$.codes", "collect": True}},
            "probes": {},
        }
    }
    failures, errors = compare(transcript, contract, {"codes": "STARTER"})
    assert not errors
    assert failures[0]["actual"].startswith("<unresolvable $.codes:")
    monkeypatch.setattr("procs.harness.replay.REPORT_DIR", tmp_path)
    write_report("sha", [{"module": "plans", "scenario": "PLANS-001", "status": "FAIL", "failures": failures}], [])
    assert (tmp_path / "parity.md").exists()


def test_rows_with_scalar_items_are_reported_and_written(tmp_path, monkeypatch) -> None:
    transcript = {
        "scenario": "PLANS-001",
        "business_fields": {"rows": [{"id": "1"}]},
        "probes": {},
    }
    contract = {
        "response": {
            "business_fields": {
                "rows": {"json_path": "$.rows", "type": "rows"}
            },
            "probes": {},
        }
    }
    failures, errors = compare(transcript, contract, {"rows": ["scalar"]})
    assert not errors
    assert failures[0]["actual"].startswith("<unresolvable $.rows: expected object rows")
    monkeypatch.setattr("procs.harness.replay.REPORT_DIR", tmp_path)
    write_report(
        "sha",
        [{"module": "plans", "scenario": "PLANS-001", "status": "FAIL", "failures": failures}],
        [],
    )
    assert (tmp_path / "parity.md").exists()
    assert (tmp_path / "parity.json").exists()


def test_rows_with_missing_column_are_reported_and_written(tmp_path, monkeypatch) -> None:
    transcript = {
        "scenario": "PLANS-001",
        "business_fields": {"rows": [{"id": "1", "code": "STARTER"}]},
        "probes": {},
    }
    contract = {
        "response": {
            "business_fields": {
                "rows": {"json_path": "$.rows", "type": "rows"}
            },
            "probes": {},
        }
    }
    failures, errors = compare(transcript, contract, {"rows": [{"id": "1"}]})
    assert not errors
    assert failures[0]["kind"] == "field"
    monkeypatch.setattr("procs.harness.replay.REPORT_DIR", tmp_path)
    write_report(
        "sha",
        [{"module": "plans", "scenario": "PLANS-001", "status": "FAIL", "failures": failures}],
        [],
    )
    assert (tmp_path / "parity.md").exists()
    assert (tmp_path / "parity.json").exists()


def test_empty_selection_fails_without_grading(monkeypatch, tmp_path) -> None:
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "SOURCE_SHA").write_text("sha\n")
    (transcripts / "FIXTURE_SHA").write_text("fixture\n")
    monkeypatch.setattr(replay, "TRANSCRIPTS", transcripts)
    monkeypatch.setattr(replay, "load_transcripts", lambda _module: [])
    monkeypatch.setattr(replay, "source_matches", lambda _expected: True)
    monkeypatch.setattr(replay, "source_sha", lambda: "sha")
    monkeypatch.setattr(replay, "fixture_sha", lambda: "fixture")
    monkeypatch.setattr(sys, "argv", ["replay.py", "--module", "typo"])
    assert replay.main() == replay.SELECTION_EMPTY


def test_mid_loop_target_failure_writes_partial_report(monkeypatch, tmp_path) -> None:
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    (transcripts / "SOURCE_SHA").write_text("sha\n")
    (transcripts / "FIXTURE_SHA").write_text("fixture\n")
    routes = tmp_path / "routes.yaml"
    routes.write_text(
        "modules:\n"
        "  plans:\n"
        "    status: extracted\n"
        "    entrypoints:\n"
        "      billing.fn_list_plans:\n"
        "        response:\n"
        "          business_fields: {}\n"
        "          probes: {}\n"
    )
    selected = [
        {
            "module": "plans",
            "scenario": "PLANS-001",
            "entrypoint": "billing.fn_list_plans",
            "inputs": {},
            "business_fields": {},
            "probes": {},
            "source_sha": "sha",
            "fixture_sha": "fixture",
        },
        {
            "module": "plans",
            "scenario": "PLANS-002",
            "entrypoint": "billing.fn_list_plans",
            "inputs": {},
            "business_fields": {},
            "probes": {},
            "source_sha": "sha",
            "fixture_sha": "fixture",
        },
    ]
    monkeypatch.setattr(replay, "TRANSCRIPTS", transcripts)
    monkeypatch.setattr(replay, "ROUTES", routes)
    monkeypatch.setattr(replay, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(replay, "load_transcripts", lambda _module: selected)
    monkeypatch.setattr(replay, "source_matches", lambda _expected: True)
    monkeypatch.setattr(replay, "source_sha", lambda: "sha")
    monkeypatch.setattr(replay, "fixture_sha", lambda: "fixture")
    monkeypatch.setattr(
        replay.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(replay, "reset_target", lambda _url: (True, ""))
    calls = 0

    def target_request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("connection reset")
        return 200, {}

    monkeypatch.setattr(replay, "target_request", target_request)
    monkeypatch.setattr(sys, "argv", ["replay.py"])

    assert replay.main() == replay.TARGET_UNREACHABLE
    report = (tmp_path / "reports" / "parity.json").read_text()
    assert "PLANS-001" in report
    assert "PLANS-002" not in report
    assert (tmp_path / "reports" / "parity.md").exists()
