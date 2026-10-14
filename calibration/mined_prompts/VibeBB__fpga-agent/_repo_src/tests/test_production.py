from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fpga import mcp_server, service
from fpga.contract import load_contract, resolve
from fpga.flow import sha256
from fpga.gates import GateReport
from fpga.interchange import FpgaProduction, sha256_file

from .conftest import read_json, write_json


def _gated(contract_path: Path, *, scope: str = "full", verdict: str = "pass") -> str:
    """Stand in for a passing `fpga gates` run: a bitstream plus its report."""
    contract = load_contract(contract_path)
    bitstream = resolve(contract_path, contract.build.bitstream)
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(b"\xff\x00bitstream" * 64)
    report = GateReport.model_validate(
        {
            "design": contract.name,
            "scope": scope,
            "contract_sha256": sha256_file(contract_path),
            "circuit_sha256": None,
            "profile": contract.device.profile,
            "bitstream_sha256": sha256(bitstream),
            "verdict": verdict,
            "checks": [],
        }
    )
    out = contract_path.parent / "fpga-reports"
    out.mkdir(exist_ok=True)
    (out / f"{contract.name}.fpga-report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return sha256(bitstream)


def _artifact(contract_path: Path) -> tuple[Path, FpgaProduction]:
    path = contract_path.parent / "fpga-reports" / "blinky.fpga-production.json"
    return path, FpgaProduction.model_validate_json(path.read_text(encoding="utf-8"))


def test_export_binds_the_gated_bitstream(ulx3s: Path) -> None:
    digest = _gated(ulx3s)
    payload = service.production_payload(ulx3s, None)
    assert payload["verdict"] == "pass", payload
    path, artifact = _artifact(ulx3s)
    assert artifact.artifact_kind == "fpga_production" and artifact.system == "fpga"
    assert artifact.bitstream_sha256 == digest
    assert artifact.contract_sha256 == sha256_file(ulx3s)
    assert artifact.gate_report_sha256 == sha256(path.parent / "blinky.fpga-report.json")
    bitstream = path.parent / artifact.bitstream
    assert sha256(bitstream) == digest and artifact.bitstream_bytes == bitstream.stat().st_size
    assert artifact.device_ref == "U1" and artifact.part and artifact.package
    assert artifact.target == "sram"
    assert artifact.argv == ["openFPGALoader", "-b", "ulx3s", artifact.bitstream]
    first = path.read_bytes()
    assert service.production_payload(ulx3s, None)["verdict"] == "pass"
    assert path.read_bytes() == first


def test_flash_target_follows_write_flash(ulx3s: Path) -> None:
    value = read_json(ulx3s)
    value["programmer"] = {"board": "ulx3s", "cable": "ft2232", "write_flash": True}
    write_json(ulx3s, value)
    _gated(ulx3s)
    assert service.production_payload(ulx3s, None)["verdict"] == "pass"
    _, artifact = _artifact(ulx3s)
    assert artifact.target == "flash"
    assert artifact.argv[:6] == ["openFPGALoader", "-b", "ulx3s", "-c", "ft2232", "-f"]


def test_no_gate_report_fails(ulx3s: Path) -> None:
    payload = service.production_payload(ulx3s, None)
    assert payload["verdict"] == "fail" and "gate report" in str(payload["detail"])


@pytest.mark.parametrize(("scope", "verdict"), [("static", "pass"), ("full", "fail")])
def test_not_a_passing_full_run_fails(ulx3s: Path, scope: str, verdict: str) -> None:
    _gated(ulx3s, scope=scope, verdict=verdict)
    payload = service.production_payload(ulx3s, None)
    assert payload["verdict"] == "fail" and "did not pass" in str(payload["detail"])


def test_contract_changed_after_gates_fails(ulx3s: Path) -> None:
    _gated(ulx3s)
    value = read_json(ulx3s)
    value["programmer"] = {"board": "ulx3s", "write_flash": True}
    write_json(ulx3s, value)
    payload = service.production_payload(ulx3s, None)
    assert payload["verdict"] == "fail" and "contract changed" in str(payload["detail"])


def test_tampered_or_missing_bitstream_fails(ulx3s: Path) -> None:
    _gated(ulx3s)
    bitstream = ulx3s.parent / "build" / "blinky.bit"
    bitstream.write_bytes(b"tampered")
    payload = service.production_payload(ulx3s, None)
    assert payload["verdict"] == "fail" and "bitstream changed" in str(payload["detail"])
    bitstream.unlink()
    payload = service.production_payload(ulx3s, None)
    assert payload["verdict"] == "fail" and "missing" in str(payload["detail"])


def test_no_programmer_fails(ulx3s: Path) -> None:
    value = read_json(ulx3s)
    del value["programmer"]
    write_json(ulx3s, value)
    _gated(ulx3s)
    payload = service.production_payload(ulx3s, None)
    assert payload["verdict"] == "fail" and "programmer" in str(payload["detail"])


def test_artifact_rejects_extra_keys_and_wrong_kind(ulx3s: Path) -> None:
    _gated(ulx3s)
    service.production_payload(ulx3s, None)
    path, _ = _artifact(ulx3s)
    data = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        FpgaProduction.model_validate({**data, "extra": 1})
    with pytest.raises(ValidationError):
        FpgaProduction.model_validate({**data, "artifact_kind": "fpga_pinmap"})


def test_mcp_dispatch(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(ulx3s.parents[2]))
    _gated(ulx3s)
    payload = mcp_server.dispatch("fpga_production_export", {"contract_path": str(ulx3s)})
    assert payload["verdict"] == "pass", payload
