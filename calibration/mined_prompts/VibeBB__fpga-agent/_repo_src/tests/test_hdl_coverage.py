"""Simulation coverage gate (fpga.coverage.<id>): real Verilator/NVC runs plus parser edges."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fpga.contract import FpgaContract, load_contract
from fpga.coverage import cobertura_counts, run_coverage, verilator_counts

SAT_V = """module sat(input wire [3:0] a, input wire [3:0] hi, output reg [3:0] y);
  always @* begin
    if (a > hi) y = hi;
    else y = a;
  end
endmodule
"""
TB_V = """`timescale 1ns/1ps
module sat_tb;
  reg [3:0] a = 0; wire [3:0] y;
  sat dut(.a(a), .hi(4'd9), .y(y));
  initial begin
    a = 3; #1; if (y !== 3) $fatal(1, "FAIL low");
{extra}    $display("PASS: sat"); $finish;
  end
endmodule
"""
SAT_VHDL = """library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;
entity sat is port (a, hi : in unsigned(3 downto 0); y : out unsigned(3 downto 0)); end;
architecture rtl of sat is begin
  process (a, hi) begin
    if a > hi then y <= hi; else y <= a; end if;
  end process;
end;
"""
TB_VHDL = """library ieee; use ieee.std_logic_1164.all; use ieee.numeric_std.all;
entity sat_tb is end;
architecture sim of sat_tb is signal a, y : unsigned(3 downto 0) := (others => '0'); begin
  dut: entity work.sat port map (a => a, hi => to_unsigned(9, 4), y => y);
  process begin
    a <= to_unsigned(3, 4); wait for 1 ns; assert y = 3 severity failure;
{extra}    report "PASS: sat"; wait;
  end process;
end;
"""
HIGH_V = '    a = 12; #1; if (y !== 9) $fatal(1, "FAIL high");\n'
HIGH_VHDL = "    a <= to_unsigned(12, 4); wait for 1 ns; assert y = 9 severity failure;\n"


def _contract(
    tmp_path: Path, language: str, *, extra: str = "", floors: tuple[float, float] = (100, 100)
) -> Path:
    vhdl = language == "vhdl"
    ext = "vhdl" if vhdl else "v"
    (tmp_path / "rtl").mkdir(exist_ok=True)
    (tmp_path / "tb").mkdir(exist_ok=True)
    (tmp_path / f"rtl/sat.{ext}").write_text(SAT_VHDL if vhdl else SAT_V, encoding="utf-8")
    tb = (TB_VHDL if vhdl else TB_V).replace("{extra}", extra)
    (tmp_path / f"tb/sat_tb.{ext}").write_text(tb, encoding="utf-8")
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "sat",
        "device": {"profile": "ice40-up5k-sg48"},
        "top": "sat",
        "sources": [{"path": f"rtl/sat.{ext}", "language": language}],
        "pins": [{"port": "a[0]", "package_pin": "2"}],
        "build": {"constraints": "sat.fpga.pcf", "bitstream": "build/sat.bin"},
        "simulations": [
            {
                "id": "sat",
                "runner": "nvc" if vhdl else "iverilog",
                "top": "sat_tb",
                "sources": [{"path": f"tb/sat_tb.{ext}", "language": language}],
                "expect": ["PASS: sat"],
                "coverage": {
                    "measure": [f"rtl/sat.{ext}"],
                    "min_line_pct": floors[0],
                    "min_branch_pct": floors[1],
                    "rationale": "both clamp arms are product behaviour",
                },
            }
        ],
    }
    path = tmp_path / "sat.fpga.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _run(path: Path) -> tuple[bool, str, list[str]]:
    contract = load_contract(path)
    sim = contract.simulations[0]
    result = run_coverage(contract, path, sim, path.parent / "fpga-reports")
    return result.ok, result.detail, result.evidence


_TOOLS = {"verilog": "verilator", "vhdl": "nvc"}


def _need(language: str) -> None:
    if shutil.which(_TOOLS[language]) is None or shutil.which("g++") is None:
        pytest.skip(f"{_TOOLS[language]} or g++ not on PATH")


@pytest.mark.tools
@pytest.mark.parametrize("language", ["verilog", "vhdl"])
def test_both_arms_meet_full_floors(tmp_path: Path, language: str) -> None:
    _need(language)
    ok, detail, evidence = _run(
        _contract(tmp_path, language, extra=HIGH_V if language == "verilog" else HIGH_VHDL)
    )
    assert ok, detail
    assert any(e.startswith("branches=") and "(100.0%)" in e for e in evidence)


@pytest.mark.tools
@pytest.mark.parametrize("language", ["verilog", "vhdl"])
def test_one_arm_misses_the_branch_floor(tmp_path: Path, language: str) -> None:
    _need(language)
    ok, detail, _ = _run(_contract(tmp_path, language))
    assert not ok and "branch coverage" in detail


@pytest.mark.tools
def test_floor_at_the_measured_value_passes(tmp_path: Path) -> None:
    _need("verilog")
    ok, detail, _ = _run(_contract(tmp_path, "verilog", floors=(0, 50)))
    assert ok, detail


@pytest.mark.tools
def test_failing_testbench_fails(tmp_path: Path) -> None:
    _need("verilog")
    bad = '    a = 12; #1; if (y !== 4) $fatal(1, "FAIL high");\n'
    ok, detail, _ = _run(_contract(tmp_path, "verilog", extra=bad, floors=(0, 0)))
    assert not ok and detail.startswith("run:")


def test_missing_tool_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def which(_name: str) -> None:
        return None

    monkeypatch.setattr("fpga.toolrun.shutil.which", which)
    ok, detail, _ = _run(_contract(tmp_path, "verilog", floors=(0, 0)))
    assert not ok and "not found on PATH" in detail


def test_missing_measured_source_fails(tmp_path: Path) -> None:
    path = _contract(tmp_path, "verilog")
    (tmp_path / "rtl/sat.v").unlink()
    ok, detail, _ = _run(path)
    assert not ok and "missing" in detail


def test_measure_must_be_a_design_source(tmp_path: Path) -> None:
    path = _contract(tmp_path, "verilog")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["simulations"][0]["coverage"]["measure"] = ["tb/sat_tb.v"]
    with pytest.raises(ValidationError, match="non-design"):
        FpgaContract.model_validate(data)


@pytest.mark.parametrize("floor", [-0.1, 100.1])
def test_floor_outside_percent_range_is_rejected(tmp_path: Path, floor: float) -> None:
    data = json.loads(_contract(tmp_path, "verilog").read_text(encoding="utf-8"))
    data["simulations"][0]["coverage"]["min_line_pct"] = floor
    with pytest.raises(ValidationError):
        FpgaContract.model_validate(data)


def _point(file: Path, kind: str, count: int) -> str:
    return f"C '\x01f\x02{file}\x01l\x023\x01t\x02{kind}\x01page\x02v_{kind}/sat' {count}"


def test_verilator_counts_only_measured_files(tmp_path: Path) -> None:
    rtl, tb = tmp_path / "sat.v", tmp_path / "tb.v"
    text = "\n".join(
        [
            "# SystemC::Coverage-3",
            _point(rtl, "line", 3),
            _point(rtl, "branch", 0),
            _point(rtl, "branch", 2),
            _point(rtl, "toggle", 0),
            _point(tb, "line", 1),
        ]
    )
    assert verilator_counts(text, {rtl.resolve()}) == {"line": [1, 1], "branch": [2, 1]}


@pytest.mark.parametrize("text", ["", "C 'x' 1", "# SystemC::Coverage-3\nC 'broken"])
def test_verilator_counts_reject_malformed_data(text: str) -> None:
    with pytest.raises(ValueError):
        verilator_counts(text, set())


def test_cobertura_counts_decisions_and_rejects_malformed(tmp_path: Path) -> None:
    xml = (
        '<coverage><packages><package><classes><class filename="sat.vhdl"><lines>'
        '<line number="5" hits="4" branch="true"><conditions>'
        '<condition number="0" type="jump" coverage="50 %"/></conditions></line>'
        '<line number="6" hits="0"/></lines></class>'
        '<class filename="tb.vhdl"><lines><line number="1" hits="1"/></lines></class>'
        "</classes></package></packages></coverage>"
    )
    measured = {(tmp_path / "sat.vhdl").resolve()}
    assert cobertura_counts(xml, tmp_path, measured) == {"line": [2, 1], "branch": [1, 0]}
    with pytest.raises(ValueError):
        cobertura_counts("<coverage><class", tmp_path, measured)
    with pytest.raises(ValueError):
        cobertura_counts("<report/>", tmp_path, measured)
    with pytest.raises(ValueError):
        cobertura_counts(xml.replace('hits="0"', 'hits="x"'), tmp_path, measured)
