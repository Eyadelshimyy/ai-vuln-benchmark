from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fpga import mcp_server, service
from fpga._envelope import check_envelope, sidecar_path
from fpga.contract import FpgaContract, load_contract
from fpga.gates import run_gates
from fpga.interchange import sha256_file

from .conftest import read_json, write_json

REGISTERS: dict[str, Any] = {
    "bus": "spi",
    "data_width": 8,
    "address_width": 4,
    "registers": [
        {
            "name": "status",
            "offset": 0,
            "access": "ro",
            "reset": 1,
            "fields": [
                {"name": "ready", "lsb": 0, "width": 1},
                {"name": "fault", "lsb": 1, "width": 2, "access": "w1c"},
            ],
        },
        {"name": "led_level", "offset": 3, "access": "rw", "reset": 128},
    ],
}


def _check(contract: Path) -> tuple[str, str]:
    report = run_gates(contract, contract.parent / "fpga-reports", full=False)
    check = next(c for c in report.checks if c.id == "fpga.regmap")
    return check.status, check.detail


@pytest.fixture
def verilog(ulx3s: Path) -> Path:
    data = read_json(ulx3s)
    data["registers"] = {**copy.deepcopy(REGISTERS), "hdl_package": "rtl/blinky_regs.vh"}
    write_json(ulx3s, data)
    rtl = ulx3s.parent / "rtl" / "blinky.v"
    rtl.write_text(
        '`include "blinky_regs.vh"\n' + rtl.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return ulx3s


@pytest.fixture
def vhdl(tangnano: Path) -> Path:
    data = read_json(tangnano)
    package = "rtl/blinky_regs_pkg.vhd"
    data["registers"] = {
        **copy.deepcopy(REGISTERS),
        "bus": "i2c",
        "i2c_address": 0x42,
        "hdl_package": package,
    }
    data["sources"].insert(0, {"path": package, "language": "vhdl"})
    write_json(tangnano, data)
    rtl = tangnano.parent / "rtl" / "blinky.vhdl"
    text = rtl.read_text(encoding="utf-8")
    rtl.write_text(
        text.replace(
            "use ieee.numeric_std.all;",
            "use ieee.numeric_std.all;\n  use work.blinky_regs_pkg.all;",
            1,
        ),
        encoding="utf-8",
    )
    return tangnano


def test_regmap_export_and_gate_pass(verilog: Path) -> None:
    payload = service.regmap_payload(verilog, None)
    assert payload["verdict"] == "pass" and payload["registers"] == 2
    export = read_json(verilog.parent / "fpga-reports" / "blinky.fpga-regmap.json")
    assert export["artifact_kind"] == "fpga_regmap" and export["system"] == "fpga"
    assert export["contract_sha256"] == sha256_file(verilog)
    assert export["bus"] == "spi" and export["i2c_address"] is None
    status = export["registers"][0]
    assert [f["mask"] for f in status["fields"]] == [0x1, 0x6]
    assert [f["access"] for f in status["fields"]] == ["ro", "w1c"]
    assert [r["offset"] for r in export["registers"]] == [0, 3]
    assert _check(verilog) == ("pass", "")


def test_export_is_byte_deterministic(verilog: Path) -> None:
    out = verilog.parent / "fpga-reports"
    service.regmap_payload(verilog, None)
    first = (out / "blinky.fpga-regmap.json").read_bytes()
    package = (verilog.parent / "rtl" / "blinky_regs.vh").read_bytes()
    service.regmap_payload(verilog, None)
    assert (out / "blinky.fpga-regmap.json").read_bytes() == first
    assert (verilog.parent / "rtl" / "blinky_regs.vh").read_bytes() == package


def test_missing_package_fails(verilog: Path) -> None:
    status, detail = _check(verilog)
    assert status == "fail" and "missing" in detail


def test_changed_map_without_regenerating_fails(verilog: Path) -> None:
    service.regmap_payload(verilog, None)
    data = read_json(verilog)
    data["registers"]["registers"][1]["offset"] = 4
    write_json(verilog, data)
    status, detail = _check(verilog)
    assert status == "fail" and "stale" in detail


def test_hand_edited_package_fails(verilog: Path) -> None:
    service.regmap_payload(verilog, None)
    package = verilog.parent / "rtl" / "blinky_regs.vh"
    package.write_text(
        package.read_text(encoding="utf-8").replace("4'h3", "4'h4"), encoding="utf-8"
    )
    assert _check(verilog)[0] == "fail"


def test_rtl_not_using_package_fails(verilog: Path) -> None:
    service.regmap_payload(verilog, None)
    rtl = verilog.parent / "rtl" / "blinky.v"
    rtl.write_text(rtl.read_text(encoding="utf-8").split("\n", 1)[1], encoding="utf-8")
    status, detail = _check(verilog)
    assert status == "fail" and "not bound" in detail


def test_vhdl_package_and_gate(vhdl: Path) -> None:
    assert service.regmap_payload(vhdl, None)["verdict"] == "pass"
    text = (vhdl.parent / "rtl" / "blinky_regs_pkg.vhd").read_text(encoding="utf-8")
    assert "package blinky_regs_pkg is" in text
    assert "REGMAP_I2C_ADDRESS : natural := 16#42#;" in text
    assert 'REG_LED_LEVEL_RESET : std_logic_vector(7 downto 0) := x"80";' in text
    assert _check(vhdl) == ("pass", "")
    rtl = vhdl.parent / "rtl" / "blinky.vhdl"
    rtl.write_text(
        rtl.read_text(encoding="utf-8").replace("use work.blinky_regs_pkg.all;", ""),
        encoding="utf-8",
    )
    assert _check(vhdl)[0] == "fail"


def test_contract_without_registers(ulx3s: Path) -> None:
    report = run_gates(ulx3s, ulx3s.parent / "fpga-reports", full=False)
    assert "fpga.regmap" not in {c.id for c in report.checks}
    payload = service.regmap_payload(ulx3s, None)
    assert payload["verdict"] == "fail" and "no registers" in str(payload["detail"])


def _target(regs: dict[str, Any], where: str) -> dict[str, Any]:
    if where == "map":
        return regs
    if where == "field1":
        target: dict[str, Any] = regs["registers"][0]["fields"][1]
        return target
    reg: dict[str, Any] = regs["registers"][int(where.removeprefix("reg"))]
    return reg


@pytest.mark.parametrize(
    ("where", "key", "value", "message"),
    [
        ("reg1", "offset", 0, "duplicate register offset"),
        ("reg1", "name", "status", "duplicate register name"),
        ("reg1", "offset", 16, "address bits"),
        ("reg1", "reset", 256, "does not fit"),
        ("field1", "lsb", 0, "overlaps"),
        ("field1", "lsb", 7, "exceed"),
        ("field1", "name", "ready", "duplicate field"),
        ("reg1", "name", "led__level", "pattern"),
        ("map", "i2c_address", 0x42, "i2c_address"),
        ("map", "bus", "i2c", "i2c_address"),
        ("map", "hdl_package", "rtl/blinky.vh", "_regs.vh"),
        ("map", "hdl_package", "/abs/blinky_regs.vh", "relative"),
        ("map", "extra", 1, "Extra inputs"),
    ],
)
def test_invalid_register_maps_rejected(
    verilog: Path, where: str, key: str, value: object, message: str
) -> None:
    data = read_json(verilog)
    _target(data["registers"], where)[key] = value
    with pytest.raises(ValidationError) as exc:
        FpgaContract.model_validate(data)
    assert message in str(exc.value)


def test_offset_at_address_limit_accepted(verilog: Path) -> None:
    data = read_json(verilog)
    data["registers"]["registers"][1]["offset"] = 15
    assert FpgaContract.model_validate(data).registers is not None


def test_verilog_include_must_not_be_a_source(verilog: Path) -> None:
    data = read_json(verilog)
    data["sources"].append({"path": "rtl/blinky_regs.vh", "language": "verilog"})
    with pytest.raises(ValidationError, match="not a source"):
        FpgaContract.model_validate(data)


def test_vhdl_package_must_be_a_source(vhdl: Path) -> None:
    data = read_json(vhdl)
    data["sources"].pop(0)
    with pytest.raises(ValidationError, match="listed in sources"):
        FpgaContract.model_validate(data)


def test_mcp_regmap_export(verilog: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(verilog.parents[2]))
    payload = mcp_server.dispatch("fpga_regmap_export", {"contract_path": str(verilog)})
    assert payload["verdict"] == "pass"
    assert load_contract(verilog).registers is not None


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="needs iverilog")
def test_verilog_include_values(verilog: Path, tmp_path: Path) -> None:
    service.regmap_payload(verilog, None)
    tb = tmp_path / "tb.v"
    tb.write_text(
        '`include "blinky_regs.vh"\n'
        "module tb;\n"
        "  initial begin\n"
        '    $display("%0d %0d %0d %0d %0d", `BLINKY_REGS_LED_LEVEL_ADDR, '
        "`BLINKY_REGS_LED_LEVEL_RESET, `BLINKY_REGS_STATUS_FAULT_LSB, "
        "`BLINKY_REGS_STATUS_FAULT_WIDTH, `BLINKY_REGS_DATA_WIDTH);\n"
        "  end\n"
        "endmodule\n",
        encoding="utf-8",
    )
    binary = tmp_path / "tb.vvp"
    subprocess.run(
        ["iverilog", "-I", str(verilog.parent / "rtl"), "-o", str(binary), str(tb)],
        check=True,
    )
    result = subprocess.run(["vvp", str(binary)], check=True, capture_output=True, text=True)
    assert result.stdout.split()[:5] == ["3", "128", "1", "2", "8"]


@pytest.mark.skipif(shutil.which("nvc") is None, reason="needs nvc")
def test_vhdl_package_values(vhdl: Path, tmp_path: Path) -> None:
    service.regmap_payload(vhdl, None)
    tb = tmp_path / "tb.vhd"
    tb.write_text(
        "library ieee;\nuse ieee.std_logic_1164.all;\nuse work.blinky_regs_pkg.all;\n"
        "entity tb is end entity;\narchitecture sim of tb is begin\n"
        "  process begin\n"
        "    assert REG_LED_LEVEL_ADDR = 3 severity failure;\n"
        '    assert REG_LED_LEVEL_RESET = x"80" severity failure;\n'
        "    assert REG_STATUS_FAULT_LSB = 1 and REG_STATUS_FAULT_WIDTH = 2 severity failure;\n"
        "    assert REGMAP_I2C_ADDRESS = 66 severity failure;\n"
        '    report "REGMAP OK";\n    wait;\n  end process;\nend architecture;\n',
        encoding="utf-8",
    )
    work = tmp_path / "work"
    package = vhdl.parent / "rtl" / "blinky_regs_pkg.vhd"
    result = subprocess.run(
        ["nvc", f"--work={work}", "-a", str(package), str(tb), "-e", "tb", "-r"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "REGMAP OK" in result.stdout + result.stderr


def test_export_round_trips_as_json(verilog: Path) -> None:
    service.regmap_payload(verilog, None)
    raw = (verilog.parent / "fpga-reports" / "blinky.fpga-regmap.json").read_text(encoding="utf-8")
    assert json.loads(raw)["design"] == "blinky"


@pytest.mark.parametrize(
    "registers",
    [
        [
            {"name": "led", "offset": 0, "access": "rw"},
            {"name": "led_reset", "offset": 1, "access": "rw"},
        ],
        [
            {
                "name": "a_b",
                "offset": 0,
                "access": "rw",
                "fields": [{"name": "lsb", "lsb": 0, "width": 1}],
            },
            {
                "name": "a",
                "offset": 1,
                "access": "rw",
                "fields": [{"name": "b_lsb", "lsb": 0, "width": 1}],
            },
        ],
    ],
)
def test_colliding_generated_names_rejected(verilog: Path, registers: list[dict[str, Any]]) -> None:
    data = read_json(verilog)
    data["registers"]["registers"] = registers
    with pytest.raises(ValidationError, match="collides"):
        FpgaContract.model_validate(data)


def test_include_dir_reaches_every_verilog_front_end(verilog: Path) -> None:
    from fpga.devices import load_profile
    from fpga.flow import synth_script
    from fpga.hdl import include_dirs

    contract = load_contract(verilog)
    rtl = (verilog.parent / "rtl").resolve()
    assert include_dirs(contract, verilog) == [rtl]
    profile = load_profile(contract.device.profile, [verilog.parent])
    script = synth_script(contract, verilog, profile)
    assert f"-I{rtl.as_posix()}" in script


def test_vhdl_needs_no_include_dir(vhdl: Path) -> None:
    from fpga.hdl import include_dirs

    assert include_dirs(load_contract(vhdl), vhdl) == []


@pytest.mark.skipif(shutil.which("verilator") is None, reason="needs verilator")
def test_verilator_lint_resolves_generated_include(verilog: Path, tmp_path: Path) -> None:
    from fpga.flow import lint

    assert service.regmap_payload(verilog, None)["verdict"] == "pass"
    run = lint(load_contract(verilog), verilog, tmp_path)
    assert run.ok, run.detail


def test_regmap_sidecar_checks_and_refuses_a_broken_log(
    verilog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENHANDS_PROJECT_DIR", str(tmp_path))
    assert service.regmap_payload(verilog, None)["verdict"] == "pass"
    regmap = verilog.parent / "fpga-reports" / "blinky.fpga-regmap.json"
    assert check_envelope(regmap, "fpga", tmp_path)[0] == "pass"
    log = tmp_path / "observations" / "fpga" / "decisions.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("{not json\n", encoding="utf-8")
    refused = service.regmap_payload(verilog, None)
    assert refused["verdict"] == "fail"
    assert str(refused["detail"]).startswith("provenance refused")
    assert not sidecar_path(regmap).exists()
