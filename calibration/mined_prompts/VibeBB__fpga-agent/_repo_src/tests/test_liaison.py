"""SLP v2 liaison: inbox states and respond refusal rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from fpga import liaison, records, service


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))


def _sha(path: Path) -> str:
    return records.sha256_file(path)


def _request(workspace: Path, request_id: str, **overrides: Any) -> Path:
    body: dict[str, Any] = {
        "schema_version": 2,
        "system": "ux-creator",
        "id": request_id,
        "target_agent": "fpga",
        "stage": "design",
        "risk": "low",
        "purpose": "Route the blinky LED bank onto the shared carrier connector",
        "rationale": "The carrier board only exposes bank 0.",
        "requested_changes": ["move led[7:0] to bank 0"],
        "inputs": [],
        "expected_deliverables": ["fpga-reports/blinky.fpga-pinmap.json"],
        "acceptance": ["fpga gates passes"],
        "depends_on": [],
        "created_at": "2026-02-20T00:00:00Z",
    }
    body.update(overrides)
    directory = workspace / "liaison"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{request_id}{liaison.REQUEST_SUFFIX}"
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def _decision(workspace: Path) -> str:
    workspace.joinpath("out").mkdir(exist_ok=True)
    workspace.joinpath("out/design.v").write_text("module m; endmodule\n", encoding="utf-8")
    result = records.record_decision(
        {
            "id": "fpga-test-clock",
            "stage": "design",
            "question": "Which clock source feeds the counter?",
            "principles": ["a synchronous design needs one stable clock domain"],
            "options": [
                {"name": "onboard oscillator", "pros": ["stable"], "cons": ["fixed"]},
                {"name": "PLL from 12 MHz", "pros": ["flexible"], "cons": ["more area"]},
            ],
            "chosen": "onboard oscillator",
            "rationale": (
                "The onboard oscillator already meets the 25 MHz requirement without"
                " spending a PLL or adding jitter; keeping the clock domain simple"
                " makes timing analysis trivial, keeps the constraint file honest, and"
                " leaves the design portable to every other board the family supports."
            ),
            "risks": ["frequency is fixed"],
            "revisit_when": "a second clock domain is needed",
            "evidence": [{"path": "out/design.v"}],
        },
        root=workspace,
    )
    return cast(dict[str, Any], result["record"])["event_id"]


def _impression(workspace: Path) -> str:
    result = records.record_impression(
        {
            "stage": "design",
            "artifacts": ["out"],
            "impression": (
                "The design came through cleanly: the counter maps onto four carry"
                " chains and the LEDs sit on dedicated user I/O, which is exactly"
                " what the contract promised. A maker reading the pin map would find"
                " the routing obvious and the report honest, with no surprises left"
                " for the circuit side. What worries me is nothing structural; only"
                " that reuse elsewhere may want a second clock, which we should note"
                " for the next revision before it becomes a rework. The next step is"
                " to hand the pin map to the circuit plugin and record the decision."
                " The module m declaration closes with endmodule as expected."
            ),
            "facets": {
                "observed": ["the design module is declared", "the module is closed"],
                "works": ["the counter maps onto carry chains"],
                "concerns": [
                    {
                        "id": "c1",
                        "text": "reuse may need a second clock domain later",
                        "severity": "info",
                        "about": "self",
                        "anchor": "module m",
                    }
                ],
                "feelings": {
                    "maker": "the pin map reads without surprises",
                    "user": "the LEDs behave as the contract promised",
                },
                "next_actions": ["hand the pin map to the circuit plugin"],
            },
            "claims": [
                {
                    "text": "the design declares its top module",
                    "anchor": "module m",
                    "artifact": "out/design.v",
                },
                {
                    "text": "the top module is closed",
                    "anchor": "endmodule",
                    "artifact": "out/design.v",
                },
            ],
            "confidence": "medium",
            "unknowns": [],
        },
        root=workspace,
    )
    return cast(dict[str, Any], result["record"])["event_id"]


def test_inbox_empty(tmp_path: Path) -> None:
    result = liaison.inbox(tmp_path)
    assert result["requests"] == [] and result["malformed"] == []
    assert result["counts"] == {"new": 0, "answered": 0, "stale": 0, "blocked": 0}


def test_inbox_new_and_other_target(tmp_path: Path) -> None:
    _request(tmp_path, "job-leds")
    _request(tmp_path, "job-other", target_agent="circuit")
    result = liaison.inbox(tmp_path)
    assert [r["id"] for r in result["requests"]] == ["job-leds"]
    assert result["requests"][0]["state"] == "new"
    assert result["counts"]["new"] == 1


def test_inbox_malformed(tmp_path: Path) -> None:
    liaison_dir = tmp_path / "liaison"
    liaison_dir.mkdir()
    (liaison_dir / "bad.ux-request.json").write_text("{not json", encoding="utf-8")
    _request(tmp_path, "for-wire", target_agent="wire")
    (liaison_dir / "wire-bad.ux-request.json").write_text(
        json.dumps({"target_agent": "wire"}), encoding="utf-8"
    )
    result = liaison.inbox(tmp_path)
    assert len(result["malformed"]) == 1
    assert result["malformed"][0]["path"].endswith("bad.ux-request.json")


def test_inbox_blocked_and_stale(tmp_path: Path) -> None:
    _request(tmp_path, "job-blocked", depends_on=["job-missing"])
    input_file = tmp_path / "input.txt"
    input_file.write_text("v1", encoding="utf-8")
    _request(
        tmp_path,
        "job-stale",
        inputs=[{"path": "input.txt", "sha256": _sha(input_file)}],
    )
    input_file.write_text("v2", encoding="utf-8")
    result = liaison.inbox(tmp_path)
    states = {r["id"]: r["state"] for r in result["requests"]}
    assert states["job-blocked"] == "blocked"
    assert states["job-stale"] == "stale"
    assert result["requests"][1]["stale_inputs"] == ["input.txt"]


def test_respond_writes_answered(tmp_path: Path) -> None:
    _request(tmp_path, "job-answer")
    decision = _decision(tmp_path)
    impression = _impression(tmp_path)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    result = liaison.respond(
        tmp_path,
        "job-answer",
        "done",
        reason="the bank-0 LED reroute is implemented and gates pass",
        artifacts=["artifact.json"],
        gate_verdicts=[{"gate": "fpga.pins", "verdict": "pass"}],
        decision_refs=[decision],
        impression_refs=[impression],
    )
    written = Path(str(result["written"]))
    assert written.name == "job-answer.ux-response.json"
    response = liaison.UxResponse.model_validate(json.loads(written.read_text(encoding="utf-8")))
    assert response.status == "done" and response.responder == "fpga"
    assert liaison.inbox(tmp_path)["requests"][0]["state"] == "answered"


def test_respond_refuses_missing_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no liaison request"):
        liaison.respond(tmp_path, "ghost", "accepted")


def test_respond_refuses_wrong_target(tmp_path: Path) -> None:
    _request(tmp_path, "job-wire", target_agent="wire")
    with pytest.raises(ValueError, match="targets wire"):
        liaison.respond(tmp_path, "job-wire", "accepted")


def test_respond_refuses_unknown_refs(tmp_path: Path) -> None:
    _request(tmp_path, "job-refs")
    with pytest.raises(ValueError, match="decision_ref"):
        liaison.respond(
            tmp_path,
            "job-refs",
            "deferred",
            reason="waiting on a sibling response for now",
            decision_refs=["0" * 64],
        )
    with pytest.raises(ValueError, match="impression_ref"):
        liaison.respond(
            tmp_path,
            "job-refs",
            "deferred",
            reason="waiting on a sibling response for now",
            impression_refs=["1" * 64],
        )


def test_respond_refuses_refs_into_broken_chain(tmp_path: Path) -> None:
    _request(tmp_path, "job-chain")
    decision = _decision(tmp_path)
    log = records.records_dir(tmp_path) / "decisions.jsonl"
    intact = log.read_text(encoding="utf-8")
    forged = json.loads(intact.splitlines()[-1]) | {"event_id": "a" * 64}
    log.write_text(intact + json.dumps(forged) + "\n", encoding="utf-8")
    for ref in ("a" * 64, decision):
        with pytest.raises(ValueError, match="fails integrity"):
            liaison.respond(
                tmp_path,
                "job-chain",
                "deferred",
                reason="waiting on a sibling response for now",
                decision_refs=[ref],
            )
    log.write_text(intact, encoding="utf-8")
    with pytest.raises(ValueError, match="decision_ref"):
        liaison.respond(
            tmp_path,
            "job-chain",
            "deferred",
            reason="waiting on a sibling response for now",
            decision_refs=["a" * 64],
        )


def test_respond_refuses_stale_inputs(tmp_path: Path) -> None:
    input_file = tmp_path / "input.txt"
    input_file.write_text("v1", encoding="utf-8")
    _request(
        tmp_path,
        "job-stale2",
        inputs=[{"path": "input.txt", "sha256": _sha(input_file)}],
    )
    input_file.write_text("v2", encoding="utf-8")
    with pytest.raises(ValueError, match="reissue"):
        liaison.respond(tmp_path, "job-stale2", "accepted")


def test_respond_done_requires_evidence(tmp_path: Path) -> None:
    _request(tmp_path, "job-done")
    with pytest.raises(ValueError, match="needs at least one artifact"):
        liaison.respond(tmp_path, "job-done", "done")


def test_respond_done_refuses_failed_verdict(tmp_path: Path) -> None:
    _request(tmp_path, "job-fail")
    decision = _decision(tmp_path)
    impression = _impression(tmp_path)
    artifact = tmp_path / "a.txt"
    artifact.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="needs_info or rejected"):
        liaison.respond(
            tmp_path,
            "job-fail",
            "done",
            reason="attempted a done verdict that should be refused",
            artifacts=["a.txt"],
            gate_verdicts=[{"gate": "fpga.timing", "verdict": "fail"}],
            decision_refs=[decision],
            impression_refs=[impression],
        )


def test_respond_reason_and_needs_info_rules(tmp_path: Path) -> None:
    _request(tmp_path, "job-rules")
    with pytest.raises(ValueError, match="at least 20 characters"):
        liaison.respond(tmp_path, "job-rules", "rejected", reason="no")
    with pytest.raises(ValueError, match="needs_info needs"):
        liaison.respond(
            tmp_path,
            "job-rules",
            "needs_info",
            reason="cannot proceed without a defined clock pin",
        )
    result = liaison.respond(
        tmp_path,
        "job-rules",
        "needs_info",
        reason="cannot proceed without a defined clock pin",
        questions_for_user=["which pin is the 25 MHz clock on?"],
    )
    assert result["verdict"] == "pass"


def test_respond_artifact_rules(tmp_path: Path) -> None:
    _request(tmp_path, "job-artifacts")
    with pytest.raises(ValueError, match="does not exist"):
        liaison.respond(tmp_path, "job-artifacts", "accepted", artifacts=["nope.bin"])
    with pytest.raises(ValueError, match="outside the workspace"):
        liaison.respond(tmp_path, "job-artifacts", "accepted", artifacts=["../x.bin"])
    link = tmp_path / "link.txt"
    real = tmp_path / "real.txt"
    real.write_text("data", encoding="utf-8")
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        liaison.respond(tmp_path, "job-artifacts", "accepted", artifacts=["link.txt"])


def test_respond_gate_report(tmp_path: Path) -> None:
    _request(tmp_path, "job-gates")
    report = tmp_path / "dut.fpga-report.json"
    report.write_text(
        json.dumps(
            {
                "checks": [
                    {"id": "fpga.pins", "status": "pass"},
                    {"id": "fpga.formal", "status": "not_applicable"},
                    {"id": "fpga.timing", "status": "unknown"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = liaison.respond(
        tmp_path,
        "job-gates",
        "in_progress",
        gate_report="dut.fpga-report.json",
    )
    response = cast(dict[str, Any], result["response"])
    verdicts = {v["gate"]: v["verdict"] for v in cast(list[Any], response["gate_verdicts"])}
    assert verdicts == {"fpga.pins": "pass", "fpga.timing": "unknown"}
    paths = [a["path"] for a in cast(list[Any], response["artifacts"])]
    assert "dut.fpga-report.json" in paths


def test_mcp_and_cli_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fpga import cli, mcp_server

    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    _request(tmp_path, "job-cli")
    payload = mcp_server.dispatch("fpga_ux_inbox", {"workspace": str(tmp_path)})
    assert payload["counts"] == {"new": 1, "answered": 0, "stale": 0, "blocked": 0}
    answer = tmp_path / "answer.json"
    answer.write_text(
        json.dumps(
            {
                "request": "job-cli",
                "status": "deferred",
                "reason": "waiting on the circuit plugin for now",
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["ux", "respond", "--workspace", str(tmp_path), "--json", str(answer)]) == 0
    capsys.readouterr()
    payload = mcp_server.dispatch("fpga_ux_inbox", {"workspace": str(tmp_path)})
    requests = cast("list[dict[str, Any]]", payload["requests"])
    assert requests[0]["state"] == "answered"


def test_service_ux_respond_fail_closed(tmp_path: Path) -> None:
    payload = service.ux_respond_payload(tmp_path, {"request": "ghost", "status": "done"})
    assert payload["verdict"] == "fail"


def test_inbox_lists_malformed_responses(tmp_path: Path) -> None:
    directory = tmp_path / "liaison"
    directory.mkdir()
    (directory / "bad-json.ux-response.json").write_text("{not json", encoding="utf-8")
    (directory / "bad-stem.ux-response.json").write_text(
        json.dumps(
            {
                "request": "other-id",
                "responder": "fpga",
                "status": "accepted",
                "responded_at": "2026-02-20T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    result = liaison.inbox(tmp_path)
    malformed = cast(list[dict[str, Any]], result["malformed"])
    names = {Path(str(m["path"])).name for m in malformed}
    assert {"bad-json.ux-response.json", "bad-stem.ux-response.json"} <= names


def test_workspace_outside_root_is_error(tmp_path: Path) -> None:
    import tempfile

    outside = Path(tempfile.mkdtemp())
    with pytest.raises(ValueError, match="outside"):
        liaison.inbox(outside)
    with pytest.raises(ValueError, match="outside"):
        liaison.respond(outside, "job", "accepted")
    # inside paths still work
    inside = tmp_path / "project"
    inside.mkdir()
    result = liaison.inbox(inside)
    assert result["counts"]["new"] == 0


def test_changed_artifacts_cover_directories(tmp_path: Path) -> None:
    _request(tmp_path, "job-dir")
    out_dir = tmp_path / "out-dir"
    out_dir.mkdir()
    (out_dir / "a.txt").write_text("v1", encoding="utf-8")
    liaison.respond(
        tmp_path,
        "job-dir",
        "accepted",
        artifacts=["out-dir"],
    )
    assert liaison.inbox(tmp_path)["requests"][0]["state"] == "answered"
    (out_dir / "a.txt").write_text("v2 changed", encoding="utf-8")
    request = liaison.inbox(tmp_path)["requests"][0]
    assert request["state"] == "stale"
    assert "out-dir" in cast(list[str], request["changed_artifacts"])


def test_family_request_id_pattern_is_pinned(tmp_path: Path) -> None:
    _request(tmp_path, "kettle.leds_v2")
    _request(tmp_path, "Kettle-Upper")
    result = liaison.inbox(tmp_path)
    assert [r["id"] for r in result["requests"]] == ["kettle.leds_v2"]
    assert len(result["malformed"]) == 1
