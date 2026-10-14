"""Implementation flow on the open-source toolchain.

lint (NVC or Verilator) -> synthesis (Yosys, ghdl-yosys-plugin for VHDL)
-> place and route (nextpnr) -> bitstream (icepack / ecppack / gowin_pack).
Every step is a subprocess; nothing passes unless the tool exits 0 and the
expected artefact exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .contract import FpgaContract, resolve
from .devices import DeviceProfile
from .hdl import (
    Unit,
    all_files,
    ghdl_args,
    include_dirs,
    is_systemverilog,
    param_text,
    units,
    verilog_read,
)
from .toolrun import ToolRun, run_tool

NEXTPNR = {"ice40": "nextpnr-ice40", "ecp5": "nextpnr-ecp5", "gowin": "nextpnr-himbaechel"}
PACKER = {"ice40": "icepack", "ecp5": "ecppack", "gowin": "gowin_pack"}
SYNTH = {"ice40": "synth_ice40", "ecp5": "synth_ecp5", "gowin": "synth_gowin"}
LAYOUT_SUFFIX = {"ice40": ".asc", "ecp5": ".config", "gowin": ".pnr.json"}
TOOL_TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class Paths:
    build: Path
    netlist: Path
    layout: Path
    report: Path
    placed: Path
    constraints: Path
    bitstream: Path


def paths(contract: FpgaContract, contract_path: Path, profile: DeviceProfile) -> Paths:
    build = resolve(contract_path, contract.build.dir)
    return Paths(
        build=build,
        netlist=build / f"{contract.name}.json",
        layout=build / f"{contract.name}{LAYOUT_SUFFIX[profile.family]}",
        report=build / f"{contract.name}.nextpnr-report.json",
        placed=build / f"{contract.name}.placed.json",
        constraints=resolve(contract_path, contract.build.constraints),
        bitstream=resolve(contract_path, contract.build.bitstream),
    )


def nvc_std(contract: FpgaContract) -> str:
    return "--std=2008" if contract.build.vhdl_standard == "08" else "--std=1993"


def nvc_analyse(
    contract: FpgaContract, unit_list: list[Unit], lib_dir: Path, cwd: Path, log: Path
) -> ToolRun:
    lib_dir.mkdir(parents=True, exist_ok=True)
    last = ToolRun(True, [])
    for unit in unit_list:
        argv = [
            "nvc",
            nvc_std(contract),
            "-L",
            lib_dir.as_posix(),
            f"--work={unit.library}:{(lib_dir / unit.library).as_posix()}",
            "-a",
            *[p.as_posix() for p in unit.files],
        ]
        last = run_tool(argv, cwd, log, TOOL_TIMEOUT_S)
        if not last.ok:
            return last
    return last


def nvc_elaborate_argv(contract: FpgaContract, lib_dir: Path, top: str) -> list[str]:
    generics = (
        [f"-g{k}={v}" for k, v in contract.build.parameters.items()] if top == contract.top else []
    )
    return [
        "nvc",
        nvc_std(contract),
        "-L",
        lib_dir.as_posix(),
        f"--work=work:{(lib_dir / 'work').as_posix()}",
        "-e",
        *generics,
        top,
    ]


def lint(contract: FpgaContract, contract_path: Path, out_dir: Path) -> ToolRun:
    """Analyse and elaborate the design top with a simulator front end."""
    unit_list = units(contract, contract_path)
    log = out_dir / "lint.log"
    log.unlink(missing_ok=True)
    root = contract_path.parent
    if contract.design_language == "vhdl":
        lib_dir = resolve(contract_path, contract.build.dir) / "lint-nvc"
        analysed = nvc_analyse(contract, unit_list, lib_dir, root, log)
        if not analysed.ok:
            return analysed
        return run_tool(
            nvc_elaborate_argv(contract, lib_dir, contract.top), root, log, TOOL_TIMEOUT_S
        )
    argv = ["verilator", "--lint-only", "-Wall", "-Wno-fatal", "--top-module", contract.top]
    if is_systemverilog(unit_list):
        argv.append("-sv")
    argv += [f"-I{p.as_posix()}" for p in include_dirs(contract, contract_path)]
    argv += [f"-D{k}={v}" for k, v in contract.build.defines.items()]
    argv += [f"-G{k}={param_text(v)}" for k, v in contract.build.parameters.items()]
    argv += [p.as_posix() for p in all_files(unit_list)]
    run = run_tool(argv, root, log, TOOL_TIMEOUT_S)
    warnings = sum(1 for line in run.output.splitlines() if line.startswith("%Warning"))
    run.evidence.append(f"verilator warnings={warnings}")
    return run


def synth_script(contract: FpgaContract, contract_path: Path, profile: DeviceProfile) -> str:
    unit_list = units(contract, contract_path)
    out = paths(contract, contract_path, profile)
    if contract.design_language == "vhdl":
        read = [f"ghdl {ghdl_args(contract, unit_list, contract.top, contract.build.parameters)}"]
    else:
        read = verilog_read(
            contract, unit_list, formal=False, includes=include_dirs(contract, contract_path)
        )
        chparams = " ".join(
            f"-chparam {k} {param_text(v)}" for k, v in contract.build.parameters.items()
        )
        read.append(f"hierarchy -check -top {contract.top} {chparams}".rstrip())
    return (
        "\n".join(
            [
                *read,
                " ".join(
                    [
                        SYNTH[profile.family],
                        *profile.synth_args,
                        f"-top {contract.top} -json {out.netlist.as_posix()}",
                    ]
                ),
                "stat",
            ]
        )
        + "\n"
    )


def synthesize(
    contract: FpgaContract, contract_path: Path, profile: DeviceProfile, out_dir: Path
) -> ToolRun:
    out = paths(contract, contract_path, profile)
    out.build.mkdir(parents=True, exist_ok=True)
    out.netlist.unlink(missing_ok=True)
    script = out.build / "synth.ys"
    script.write_text(synth_script(contract, contract_path, profile), encoding="utf-8")
    argv = ["yosys"]
    if contract.design_language == "vhdl":
        argv += ["-m", "ghdl"]
    argv += ["-s", script.as_posix()]
    log = out_dir / "synth.log"
    log.unlink(missing_ok=True)
    run = run_tool(argv, contract_path.parent, log, TOOL_TIMEOUT_S)
    if run.ok and not out.netlist.is_file():
        run.ok, run.detail = False, f"yosys wrote no netlist {out.netlist.name}"
    warnings = len(re.findall(r"^Warning:", run.output, flags=re.MULTILINE))
    run.evidence.append(f"yosys warnings={warnings}")
    return run


def netlist_ports(netlist: Path, top: str) -> dict[str, str]:
    """Bit-level top ports (``name`` or ``name[i]``) -> direction."""
    data = cast(dict[str, Any], json.loads(netlist.read_text(encoding="utf-8")))
    modules = cast(dict[str, Any], data.get("modules", {}))
    module = cast(dict[str, Any] | None, modules.get(top))
    if module is None:
        raise ValueError(f"netlist has no module {top}")
    result: dict[str, str] = {}
    for name, port in cast(dict[str, dict[str, Any]], module.get("ports", {})).items():
        width = len(cast(list[object], port.get("bits", [])))
        offset = int(port.get("offset", 0))
        direction = str(port.get("direction", ""))
        if width == 1 and offset == 0:
            result[name] = direction
        else:
            for index in range(offset, offset + width):
                result[f"{name}[{index}]"] = direction
    return result


def pnr_argv(contract: FpgaContract, contract_path: Path, profile: DeviceProfile) -> list[str]:
    out = paths(contract, contract_path, profile)
    argv = [
        NEXTPNR[profile.family],
        *profile.nextpnr_args,
        "--json",
        out.netlist.as_posix(),
        "--report",
        out.report.as_posix(),
        "--seed",
        str(contract.build.seed),
        "--timing-allow-fail",
    ]
    if contract.clocks:
        argv += ["--freq", f"{max(c.frequency_mhz for c in contract.clocks):g}"]
    constraints = out.constraints.as_posix()
    layout = out.layout.as_posix()
    placed = out.placed.as_posix()
    if profile.family == "ice40":
        argv += ["--pcf", constraints, "--asc", layout, "--write", placed]
    elif profile.family == "ecp5":
        argv += ["--lpf", constraints, "--textcfg", layout, "--write", placed]
    else:
        argv += ["--vopt", f"cst={constraints}", "--write", layout]
    return argv


def placed_json(out: Paths, family: str) -> Path:
    """Post-P&R design JSON with NEXTPNR_BEL attributes (gowin reuses layout)."""
    return out.layout if family == "gowin" else out.placed


def place_and_route(
    contract: FpgaContract, contract_path: Path, profile: DeviceProfile, out_dir: Path
) -> ToolRun:
    out = paths(contract, contract_path, profile)
    for stale in (out.layout, out.report, out.placed):
        stale.unlink(missing_ok=True)
    log = out_dir / "pnr.log"
    log.unlink(missing_ok=True)
    run = run_tool(
        pnr_argv(contract, contract_path, profile), contract_path.parent, log, TOOL_TIMEOUT_S
    )
    if run.ok and not (out.layout.is_file() and out.report.is_file()):
        run.ok, run.detail = False, "nextpnr wrote no layout or report"
    return run


def pack(
    contract: FpgaContract, contract_path: Path, profile: DeviceProfile, out_dir: Path
) -> ToolRun:
    out = paths(contract, contract_path, profile)
    out.bitstream.parent.mkdir(parents=True, exist_ok=True)
    out.bitstream.unlink(missing_ok=True)
    tool = PACKER[profile.family]
    if profile.family == "gowin":
        argv = [tool, *profile.pack_args, "-o", out.bitstream.as_posix(), out.layout.as_posix()]
    else:
        argv = [tool, *profile.pack_args, out.layout.as_posix(), out.bitstream.as_posix()]
    log = out_dir / "pack.log"
    log.unlink(missing_ok=True)
    return run_tool(argv, contract_path.parent, log, TOOL_TIMEOUT_S)


def load_report(report: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(report.read_text(encoding="utf-8")))


def bitstream_problems(family: str, data: bytes) -> list[str]:
    """Structural check of a packed bitstream: size and family preamble."""
    if len(data) < 1024:
        return [f"bitstream is only {len(data)} bytes"]
    if family == "ice40" and b"\x7e\xaa\x99\x7e" not in data[:256]:
        return ["iCE40 preamble 7EAA997E not found"]
    if family == "ecp5" and b"\xff\xff\xbd\xb3" not in data[:2048]:
        return ["ECP5 preamble FFFFBDB3 not found"]
    if family == "gowin":
        head = data[:4096].decode("ascii", "replace").splitlines()
        if any(set(line) - {"0", "1"} for line in head[:-1]):
            return ["Gowin .fs is not a 0/1 text bitstream"]
        if "1010010111000011" not in head[:8]:
            return ["Gowin preamble A5C3 not found"]
    return []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
