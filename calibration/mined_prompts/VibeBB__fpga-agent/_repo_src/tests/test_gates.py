from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fpga import flow, service
from fpga.contract import FpgaContract, load_contract
from fpga.devices import load_profile
from fpga.gates import Check, GateReport, check_timing, check_utilization, run_gates
from fpga.projections import constraints_text

from .conftest import read_json, write_json


def _static(contract: Path) -> dict[str, Check]:
    report = run_gates(contract, contract.parent / "fpga-reports", full=False)
    return {c.id: c for c in report.checks}


def _edit(contract: Path, **changes: Any) -> None:
    value = read_json(contract)
    value.update(changes)
    write_json(contract, value)


def _regen(contract: Path) -> None:
    assert service.constraints_payload(contract)["verdict"] == "pass"


def test_static_gates_pass(
    ulx3s: Path, tangnano: Path, tangnano20k: Path, tangprimer20k: Path, uart_echo: Path
) -> None:
    for contract in (ulx3s, tangnano, tangnano20k, tangprimer20k, uart_echo):
        report = run_gates(contract, contract.parent / "fpga-reports", full=False)
        assert report.verdict == "pass", [c for c in report.checks if c.status == "fail"]
        assert report.scope == "static"


def test_uart_echo_records_colibri_provenance(uart_echo: Path) -> None:
    checks = _static(uart_echo)
    assert "CERN-OHL-W-2.0" in checks["fpga.provenance"].evidence[0]
    assert checks["fpga.netlist_match"].status == "pass"


@pytest.mark.parametrize(
    ("family_fixture", "needles"),
    [
        ("ulx3s", ['LOCATE COMP "led[0]" SITE "B2";', 'IOBUF PORT "clk_25mhz" IO_TYPE=LVCMOS33;']),
        ("tangnano", ['IO_LOC "clk" 52;', 'IO_PORT "led_n[0]" IO_TYPE=LVCMOS18;']),
        ("tangprimer20k", ['IO_LOC "clk" H11;', 'IO_LOC "led[3]" N16;']),
    ],
)
def test_constraint_projection(
    family_fixture: str, needles: list[str], request: pytest.FixtureRequest
) -> None:
    contract: Path = request.getfixturevalue(family_fixture)
    loaded = load_contract(contract)
    text = constraints_text(loaded, load_profile(loaded.device.profile))
    assert text.startswith(("# Generated", "// Generated"))
    for needle in needles:
        assert needle in text


def test_stale_constraints_fail(ulx3s: Path) -> None:
    value = read_json(ulx3s)
    value["pins"][1]["package_pin"] = "C2"
    value["pins"][2]["package_pin"] = "B2"
    write_json(ulx3s, value)
    assert _static(ulx3s)["fpga.constraints"].status == "fail"
    _regen(ulx3s)
    assert _static(ulx3s)["fpga.constraints"].status == "pass"


def test_unknown_package_pin_fails(ulx3s: Path) -> None:
    value = read_json(ulx3s)
    value["pins"][1]["package_pin"] = "Z99"
    write_json(ulx3s, value)
    _regen(ulx3s)
    check = _static(ulx3s)["fpga.pins"]
    assert check.status == "fail" and "Z99" in check.detail


def test_config_pin_requires_acknowledgement(uart_echo: Path) -> None:
    value = read_json(uart_echo)
    value["pins"][3]["package_pin"] = "14"
    value["pins"][3]["net"] = "FLASH_MISO"
    write_json(uart_echo, value)
    _regen(uart_echo)
    assert "config pin" in _static(uart_echo)["fpga.pins"].detail
    value["pins"][3]["acknowledge"] = ["config"]
    value["pins"][3]["rationale"] = "LED shares the flash MISO line after configuration"
    write_json(uart_echo, value)
    _regen(uart_echo)
    assert _static(uart_echo)["fpga.pins"].status == "pass"


def test_ice40_rejects_io_standard_and_pulldown(uart_echo: Path) -> None:
    value = read_json(uart_echo)
    value["pins"][1]["io_standard"] = "LVCMOS33"
    value["pins"][2]["pull"] = "down"
    write_json(uart_echo, value)
    _regen(uart_echo)
    detail = _static(uart_echo)["fpga.pins"].detail
    assert "cannot set an I/O standard" in detail and "no pull-down" in detail


def test_netlist_mismatch_fails(uart_echo: Path) -> None:
    value = read_json(uart_echo)
    value["pins"][2]["net"] = "WRONG_NET"
    write_json(uart_echo, value)
    detail = _static(uart_echo)["fpga.netlist_match"].detail
    assert "contract says WRONG_NET" in detail


def test_unconstrained_circuit_pin_fails(uart_echo: Path) -> None:
    circuit = uart_echo.parent / "circuit" / "icebreaker.firmware.json"
    value = read_json(circuit)
    value["mcus"][0]["pins"].append(
        {"pin": "2", "net": "BTN_N", "signal_class": "signal", "voltage_v": 3.3}
    )
    write_json(circuit, value)
    assert "BTN_N" in _static(uart_echo)["fpga.netlist_match"].detail
    _edit(
        uart_echo,
        circuit={"connectivity": "circuit/icebreaker.firmware.json", "unused_pins": ["2"]},
    )
    assert _static(uart_echo)["fpga.netlist_match"].status == "pass"


def test_missing_circuit_artifact_fails(uart_echo: Path) -> None:
    (uart_echo.parent / "circuit" / "icebreaker.firmware.json").unlink()
    assert _static(uart_echo)["fpga.netlist_match"].status == "fail"


def test_provenance_rejects_license_mismatch(uart_echo: Path) -> None:
    value = read_json(uart_echo)
    value["libraries"][0]["license"] = "MIT"
    write_json(uart_echo, value)
    check = _static(uart_echo)["fpga.provenance"]
    assert check.status == "fail" and "CERN-OHL-W-2.0" in check.detail


def test_provenance_rejects_missing_spdx(tmp_path: Path, uart_echo: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.vhdl").write_text("entity a is end;\n", encoding="utf-8")
    value = read_json(uart_echo)
    value["libraries"].append(
        {
            "name": "extra",
            "root": "../../lib",
            "files": ["a.vhdl"],
            "license": "MIT",
            "source_url": "https://example.com/extra",
            "revision": "0123456789abcdef",
        }
    )
    write_json(uart_echo, value)
    assert "SPDX" in _static(uart_echo)["fpga.provenance"].detail


def test_invalid_contract_fails_closed(ulx3s: Path) -> None:
    ulx3s.write_text("{not json", encoding="utf-8")
    report = run_gates(ulx3s, ulx3s.parent / "fpga-reports", full=True)
    assert report.verdict == "fail"
    assert [c.id for c in report.checks] == ["fpga.contract"]


def test_missing_tools_fail_closed(ulx3s: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    report = run_gates(ulx3s, ulx3s.parent / "fpga-reports", full=True)
    assert report.verdict == "fail"
    statuses = {c.id: c.status for c in report.checks}
    for gate in ("fpga.lint", "fpga.sim.count", "fpga.formal.increment", "fpga.synth"):
        assert statuses[gate] == "fail", gate
    assert statuses["fpga.bitstream"] == "fail"
    assert report.bitstream_sha256 is None


def _report(fmax: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {"fmax": fmax}


def test_timing_gate(ulx3s: Path) -> None:
    contract = load_contract(ulx3s)
    ok = check_timing(
        contract, _report({"$glbnet$clk_25mhz": {"achieved": 120.0, "constraint": 25.0}})
    )
    assert ok.status == "pass"
    slow = check_timing(
        contract, _report({"$glbnet$clk_25mhz": {"achieved": 20.0, "constraint": 25.0}})
    )
    assert slow.status == "fail" and "20.00 MHz" in slow.detail
    assert check_timing(contract, _report({})).status == "fail"
    other = check_timing(
        contract,
        _report(
            {
                "$glbnet$clk_25mhz": {"achieved": 120.0, "constraint": 25.0},
                "pll_out": {"achieved": 50.0, "constraint": 100.0},
            }
        ),
    )
    assert other.status == "fail" and "pll_out" in other.detail


def test_utilization_gate(ulx3s: Path) -> None:
    contract = load_contract(ulx3s)
    profile = load_profile(contract.device.profile)
    within = {"utilization": {"TRELLIS_COMB": {"used": 100, "available": 1000}}}
    assert check_utilization(contract, profile, within).status == "pass"
    over = {"utilization": {"TRELLIS_COMB": {"used": 900, "available": 1000}}}
    assert "exceeds the logic budget" in check_utilization(contract, profile, over).detail
    unknown = {"utilization": {"MYSTERY_CELL": {"used": 1, "available": 1}}}
    assert "unclassified" in check_utilization(contract, profile, unknown).detail
    assert check_utilization(contract, profile, {}).status == "fail"


@pytest.mark.parametrize(
    ("family", "data", "ok"),
    [
        ("ice40", b"\xff\x00\x00\xff\x7e\xaa\x99\x7e" + bytes(2048), True),
        ("ice40", bytes(4096), False),
        ("ecp5", b"\xff\x00" + b"\xff\xff\xbd\xb3" + bytes(2048), True),
        ("ecp5", bytes(4096), False),
        ("gowin", b"\n".join([b"1" * 16, b"1010010111000011", b"0" * 4096]), True),
        ("gowin", b"hello\n" * 400, False),
        ("ice40", b"\x7e\xaa\x99\x7e", False),
    ],
)
def test_bitstream_structure(family: str, data: bytes, ok: bool) -> None:
    assert (flow.bitstream_problems(family, data) == []) is ok


def test_report_schema_round_trip(ulx3s: Path) -> None:
    payload = service.gates_payload(ulx3s, None, full=False)
    report = GateReport.model_validate_json(
        (ulx3s.parent / "fpga-reports" / "blinky.fpga-report.json").read_text(encoding="utf-8")
    )
    assert report.verdict == payload["verdict"] == "pass"
    assert (ulx3s.parent / "fpga-reports" / "blinky.fpga-pinmap.json").is_file()


def test_vision_points_name_checklist_and_record_tool() -> None:
    images = [
        "fpga-reports/blinky.fpga-pinmap.png",
        "fpga-reports/blinky.fpga-wave-uart.png",
        "fpga-reports/blinky.fpga-timing.png",
    ]
    points = service.vision_points(images)

    assert [p["image_path"] for p in points] == images
    assert [p["checklist"] for p in points] == ["pinmap", "waveform", "timing"]
    assert all(p["record_with"] == "fpga_record_vision_review" for p in points)


def test_pinmap_payload_carriesvision_points(ulx3s: Path) -> None:
    payload = service.pinmap_payload(ulx3s, None)

    images = [str(p) for p in payload["images"]]  # type: ignore[index]
    assert images and images[0].endswith(".fpga-pinmap.png")
    points = payload["vision_points"]
    assert [p["image_path"] for p in points] == images  # type: ignore[union-attr]
    assert points[0]["checklist"] == "pinmap"  # type: ignore[index]
    assert points[0]["record_with"] == "fpga_record_vision_review"  # type: ignore[index]


@pytest.mark.parametrize(("achieved", "status"), [(27.5, "pass"), (27.49, "fail")])
def test_timing_corner_margin_boundary(ulx3s: Path, achieved: float, status: str) -> None:
    contract = load_contract(ulx3s)
    clocks = [clock.model_copy(update={"margin_pct": 10}) for clock in contract.clocks]
    banded = contract.model_copy(update={"clocks": clocks})
    report = _report({"$glbnet$clk_25mhz": {"achieved": achieved, "constraint": 25.0}})
    check = check_timing(banded, report)
    assert check.status == status
    text = check.detail + " ".join(check.evidence)
    assert "27.5 MHz" in text and "+10% corner margin" in text


def test_timing_corner_margin_rejects_out_of_range(ulx3s: Path) -> None:
    data = json.loads(ulx3s.read_text(encoding="utf-8"))
    for bad in (-1, 100):
        data["clocks"][0]["margin_pct"] = bad
        with pytest.raises(ValidationError):
            FpgaContract.model_validate(data)
