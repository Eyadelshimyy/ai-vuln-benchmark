from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fpga import cli, mcp_server
from fpga._envelope import sidecar_path, write_envelope
from fpga.contract import FpgaContract, load_contract
from fpga.gates import run_gates
from fpga.sim_thermal import (
    expected_request,
    resolve_response,
    thermal_brief,
    thermal_check,
    thermal_findings,
    write_sim_request,
)

THERMAL: dict[str, Any] = {
    "ambient_c": 50,
    "power_w": 1.2,
    "tj_max_c": 100,
    "derating_margin_c": 10,
    "theta_ja_c_per_w": 25,
    "source": "Lattice Power Calculator 2026.1, 25 MHz blinky, typical process",
    "response_path": "fpga-reports/blinky.thermal.sim-response.json",
}
PASSING: list[dict[str, Any]] = [
    {"id": "thermal.U1.tj", "verdict": "pass", "measured": 80.0, "limit": "<= 90 C"},
]


def _with_thermal(contract_path: Path, **changes: Any) -> FpgaContract:
    data = json.loads(contract_path.read_text(encoding="utf-8"))
    data["thermal"] = {**THERMAL, **changes}
    contract_path.write_text(json.dumps(data), encoding="utf-8")
    return load_contract(contract_path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _answer(
    root: Path,
    contract: FpgaContract,
    contract_path: Path,
    checks: list[dict[str, Any]],
    *,
    status: str | None = None,
    verdict: str | None = None,
    report_verdict: str | None = None,
    **overrides: Any,
) -> Path:
    """Play simulation-agent: write request (fpga), report and response (sim)."""
    out = write_sim_request(contract, contract_path.parent / "fpga-reports", root=root)
    request_path = Path(out["request"])
    rows: list[dict[str, Any]] = [
        {"analysis": "thermal", "detail": "", "evidence": [], **c} for c in checks
    ]
    statuses = {c["verdict"] for c in rows}
    aggregate = "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"
    verdict = verdict or aggregate
    report = root / "out" / contract.name / "sim-report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"schema_version": 1, "verdict": report_verdict or verdict, "checks": rows}),
        encoding="utf-8",
    )
    response: dict[str, Any] = {
        "schema_version": 2,
        "request_id": out["request_id"],
        "request_sha256": _sha(request_path),
        "brief_sha256": out["sim_brief_sha256"],
        "status": status
        or {"pass": "accepted", "fail": "rejected", "unknown": "needs_info"}[verdict],
        "verdict": verdict,
        "report_path": report.relative_to(root).as_posix(),
        "sha256": _sha(report),
        "decision_refs": [],
        "reasons": [],
        **overrides,
    }
    path = resolve_response(contract, contract_path)
    assert path is not None
    path.write_text(json.dumps(response), encoding="utf-8")
    write_envelope(path, "sim", [], root)
    return path


def _root(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = ulx3s.parents[2]
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(root))
    return root


def _status(contract: FpgaContract, path: Path | None, root: Path) -> dict[str, str]:
    return {subject: status for subject, status, _, _ in thermal_findings(contract, path, root)}


def test_brief_carries_the_package_as_one_component(ulx3s: Path) -> None:
    contract = _with_thermal(ulx3s)
    payload = thermal_brief(contract)
    assert payload["thermal"] == {
        "ambient_c": 50,
        "components": [
            {
                "ref": "U1",
                "power_w": 1.2,
                "tj_max_c": 100,
                "derating_margin_c": 10,
                "path": {"theta_ja_c_per_w": 25},
            }
        ],
    }
    assert "Lattice Power Calculator" in payload["description"]
    assert "response_path" not in json.dumps(payload)


def test_package_without_theta_is_left_for_simulation(ulx3s: Path) -> None:
    contract = _with_thermal(ulx3s, theta_ja_c_per_w=None)
    assert "path" not in thermal_brief(contract)["thermal"]["components"][0]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source": ""}, "source"),
        ({"power_w": -1}, "power_w"),
        ({"tj_max_c": 60}, "must exceed ambient_c"),
        ({"response_path": "/tmp/x.sim-response.json"}, "relative"),
    ],
)
def test_thermal_rejects_bad_facts(ulx3s: Path, changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _with_thermal(ulx3s, **changes)


def test_request_identity_ignores_response_path(ulx3s: Path) -> None:
    first = expected_request(_with_thermal(ulx3s))
    moved = expected_request(_with_thermal(ulx3s, response_path="sim/b.sim-response.json"))
    assert first == moved
    assert first[0].startswith("blinky-thermal-")
    assert expected_request(_with_thermal(ulx3s, power_w=1.3)) != first


def test_round_trip_pass_and_gate(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    path = _answer(root, contract, ulx3s, PASSING)
    request = json.loads(path.with_name("blinky.thermal.sim-request.json").read_text("utf-8"))
    assert (request["from_system"], request["kind"]) == ("fpga", "thermal")
    assert request["brief_path"] == "examples/blinky-ulx3s/fpga-reports/blinky.thermal.sim.json"
    assert _status(contract, path, root) == {
        "response": "pass",
        "U1.tj": "pass",
        "provenance": "pass",
    }
    checks = [c for c in run_gates(ulx3s, root / "gates", full=False).checks]
    sim = [c for c in checks if c.id == "fpga.sim_thermal"]
    assert [c.status for c in sim] == ["pass", "pass", "pass"]
    assert sim[1].evidence == ["measured=80"]


def test_simulation_fail_stays_fail_with_guidance(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    failing = [
        {
            **PASSING[0],
            "verdict": "fail",
            "measured": 95.0,
            "margin": -5.0,
            "guidance": ["reduce power_w by 0.2 W"],
        }
    ]
    path = _answer(root, contract, ulx3s, failing)
    result = thermal_check(contract, ulx3s, root)
    assert result["verdict"] == "fail"
    detail = result["checks"][1]["detail"]
    assert "margin -5" in detail
    assert "reduce power_w by 0.2 W" in detail
    assert path.is_file()
    report = run_gates(ulx3s, root / "gates", full=False)
    assert report.verdict == "fail"


@pytest.mark.parametrize("status", ["needs_info", "deferred"])
def test_unanswered_is_unknown_and_fails_the_gate(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    _answer(
        root,
        contract,
        ulx3s,
        PASSING,
        status=status,
        verdict="unknown",
        report_path=None,
        sha256=None,
        reasons=["no theta"],
    )
    assert thermal_check(contract, ulx3s, root)["verdict"] == "unknown"
    gate = [
        c for c in run_gates(ulx3s, root / "g", full=False).checks if c.id == "fpga.sim_thermal"
    ]
    assert [c.status for c in gate] == ["fail"]
    assert gate[0].detail.startswith("unknown: simulation")


def test_missing_and_unset_response_are_unknown(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(ulx3s, monkeypatch)
    assert _status(_with_thermal(ulx3s), None, root) == {"response": "unknown"}
    contract = _with_thermal(ulx3s)
    assert _status(contract, resolve_response(contract, ulx3s), root) == {"response": "unknown"}
    assert thermal_check(_with_thermal(ulx3s, response_path=None), ulx3s, root)["verdict"] == (
        "unknown"
    )


def test_stale_contract_fails(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(ulx3s, monkeypatch)
    path = _answer(root, _with_thermal(ulx3s), ulx3s, PASSING)
    assert _status(_with_thermal(ulx3s, power_w=2.0), path, root) == {"response": "fail"}


def test_tampered_request_and_report_fail(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    path = _answer(root, contract, ulx3s, PASSING)
    report = root / "out" / "blinky" / "sim-report.json"
    report.write_text(report.read_text("utf-8").replace("80.0", "70.0"), encoding="utf-8")
    assert _status(contract, path, root) == {"report": "fail"}
    path = _answer(root, contract, ulx3s, PASSING)
    request = path.with_name("blinky.thermal.sim-request.json")
    request.write_text(request.read_text("utf-8") + " ", encoding="utf-8")
    assert _status(contract, path, root) == {"response": "fail"}


def test_foreign_requester_and_verdict_mismatch_fail(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    path = _answer(root, contract, ulx3s, PASSING, report_verdict="fail")
    assert _status(contract, path, root) == {"report": "fail"}
    path = _answer(root, contract, ulx3s, PASSING)
    request = path.with_name("blinky.thermal.sim-request.json")
    data = json.loads(request.read_text("utf-8"))
    request.write_text(json.dumps({**data, "from_system": "circuit"}), encoding="utf-8")
    response = json.loads(path.read_text("utf-8"))
    path.write_text(json.dumps({**response, "request_sha256": _sha(request)}), encoding="utf-8")
    assert _status(contract, path, root) == {"response": "fail"}


@pytest.mark.parametrize(
    "payload",
    ["not json", "[]", json.dumps({"schema_version": 2}), json.dumps({"status": "accepted"})],
)
def test_malformed_response_is_unknown(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    path = _answer(root, contract, ulx3s, PASSING)
    path.write_text(payload, encoding="utf-8")
    assert _status(contract, path, root) == {"response": "unknown"}


@pytest.mark.parametrize(
    "checks",
    [[], [{"id": "pdn.N1.drop", "verdict": "pass"}], [{"id": "thermal.U1.tj", "verdict": "ok"}]],
)
def test_report_without_usable_thermal_checks_is_unknown(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch, checks: list[dict[str, Any]]
) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    path = _answer(root, contract, ulx3s, checks, verdict="pass")
    assert _status(contract, path, root) == {"report": "unknown"}


def test_report_outside_workspace_is_unknown(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    path = _answer(root, contract, ulx3s, PASSING, report_path="../escape.json")
    assert _status(contract, path, root) == {"report": "unknown"}


def test_cli_and_mcp_round_trip(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    assert cli.main(["sim-request", str(ulx3s)]) == 0
    written = json.loads(capsys.readouterr().out)
    assert written["request_id"] == expected_request(contract)[0]
    assert cli.main(["sim-check", str(ulx3s)]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "unknown"
    _answer(root, contract, ulx3s, PASSING)
    payload = mcp_server.dispatch("fpga_sim_thermal_check", {"contract_path": str(ulx3s)})
    assert payload["verdict"] == "pass"
    with pytest.raises(ValueError, match="outside the workspace"):
        mcp_server.dispatch(
            "fpga_sim_thermal_request", {"contract_path": str(ulx3s), "out_dir": "/tmp"}
        )


def test_request_without_thermal_section_fails(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _root(ulx3s, monkeypatch)
    assert cli.main(["sim-request", str(ulx3s)]) == 1
    assert "no thermal section" in json.loads(capsys.readouterr().out)["detail"]
    assert all(c.id != "fpga.sim_thermal" for c in run_gates(ulx3s, ulx3s.parent / "g").checks)


def test_response_provenance_missing_or_foreign_is_not_a_pass(
    ulx3s: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(ulx3s, monkeypatch)
    contract = _with_thermal(ulx3s)
    path = _answer(root, contract, ulx3s, PASSING)
    sidecar = sidecar_path(path)
    body = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar.write_text(json.dumps(body | {"producer": "circuit"}), encoding="utf-8")
    assert _status(contract, path, root)["provenance"] == "fail"
    sidecar.unlink()
    assert _status(contract, path, root)["provenance"] == "unknown"
