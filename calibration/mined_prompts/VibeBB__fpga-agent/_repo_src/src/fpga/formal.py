"""Formal verification with SymbiYosys.

The ``.sby`` script is generated from the contract so the proof always
reads the same sources as synthesis. VHDL designs go through the
ghdl-yosys-plugin (PSL ``assert``/``assume``/``cover``); Verilog designs
are read with ``read_verilog -formal`` and ``FORMAL`` defined (immediate
and concurrent SVA subset supported by Yosys).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contract import Formal, FpgaContract, resolve
from .hdl import ghdl_args, include_dirs, param_text, units, verilog_read
from .toolrun import run_tool


@dataclass
class FormalResult:
    ok: bool
    detail: str
    argv: list[str]
    status: str
    seconds: float


def sby_text(contract: FpgaContract, contract_path: Path, run: Formal) -> str:
    unit_list = units(contract, contract_path, run.sources)
    if contract.design_language == "vhdl":
        read = [f"ghdl {ghdl_args(contract, unit_list, run.top, run.parameters)}"]
    else:
        read = verilog_read(
            contract, unit_list, formal=True, includes=include_dirs(contract, contract_path)
        )
        chparams = " ".join(f"-chparam {k} {param_text(v)}" for k, v in run.parameters.items())
        read.append(f"hierarchy -check -top {run.top} {chparams}".rstrip())
    return "\n".join(
        [
            "[options]",
            f"mode {run.mode}",
            f"depth {run.depth}",
            "",
            "[engines]",
            run.engine,
            "",
            "[script]",
            *read,
            f"prep -top {run.top}",
            "",
        ]
    )


def _status(output: str) -> str:
    for line in reversed(output.splitlines()):
        if "DONE (" in line:
            return line.split("DONE (", 1)[1].split(",", 1)[0].split(")", 1)[0].strip()
    return "UNKNOWN"


def run_formal(
    contract: FpgaContract, contract_path: Path, run: Formal, out_dir: Path
) -> FormalResult:
    work = resolve(contract_path, contract.build.dir) / f"formal-{run.id}"
    work.mkdir(parents=True, exist_ok=True)
    script = work / f"{run.id}.sby"
    script.write_text(sby_text(contract, contract_path, run), encoding="utf-8")
    argv = ["sby", "-f", "-d", (work / "run").as_posix()]
    if contract.design_language == "vhdl":
        argv += ["--yosys", "yosys -m ghdl"]
    argv.append(script.as_posix())
    log = out_dir / f"formal-{run.id}.log"
    log.unlink(missing_ok=True)
    result = run_tool(argv, contract_path.parent, log, run.timeout_s)
    status = _status(result.output)
    ok = result.ok and status == "PASS"
    detail = "" if ok else (result.detail or f"sby status {status}")
    if not ok and status not in ("UNKNOWN", ""):
        detail = f"sby status {status}" + (f" ({result.detail})" if result.detail else "")
    return FormalResult(ok, detail, argv, status, result.seconds)
