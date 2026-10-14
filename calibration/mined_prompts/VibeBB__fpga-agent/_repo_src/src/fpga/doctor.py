"""Probe the FPGA toolchain (JSON verdict).

Required tools are the open-source flow the gates run. Vendor tools are
probed only to report their presence: they are proprietary, never bundled,
and no gate depends on them.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict

Kind = Literal["required", "optional", "vendor"]

REQUIRED: tuple[tuple[str, list[str]], ...] = (
    ("yosys", ["yosys", "-V"]),
    ("ghdl-yosys-plugin", ["yosys", "-m", "ghdl", "-p", "help ghdl", "-q"]),
    ("nvc", ["nvc", "--version"]),
    ("iverilog", ["iverilog", "-V"]),
    ("vvp", ["vvp", "-V"]),
    ("verilator", ["verilator", "--version"]),
    ("sby", ["sby", "--help"]),
    ("yosys-smtbmc", ["yosys-smtbmc", "--help"]),
    ("yices-smt2", ["yices-smt2", "--version"]),
    ("nextpnr-ice40", ["nextpnr-ice40", "--version"]),
    ("nextpnr-ecp5", ["nextpnr-ecp5", "--version"]),
    ("nextpnr-himbaechel", ["nextpnr-himbaechel", "--version"]),
    ("icepack", ["icepack", "-h"]),
    ("ecppack", ["ecppack", "--help"]),
    ("gowin_pack", ["gowin_pack", "-h"]),
)
OPTIONAL: tuple[tuple[str, list[str]], ...] = (
    ("ghdl", ["ghdl", "--version"]),
    ("z3", ["z3", "--version"]),
    ("boolector", ["boolector", "--version"]),
    ("openFPGALoader", ["openFPGALoader", "-V"]),
)
VENDOR: tuple[tuple[str, list[str]], ...] = (
    ("vivado", ["vivado", "-version"]),
    ("quartus_sh", ["quartus_sh", "--version"]),
    ("radiantc", ["radiantc", "-version"]),
    ("gw_sh", ["gw_sh", "-h"]),
)


class ToolCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    status: Literal["ok", "warn", "fail", "absent"]
    kind: Kind
    version: str = ""


def _probe(argv: list[str]) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode not in (0, 1) or (argv[1:2] == ["-m"] and proc.returncode != 0):
        return None
    text = [line for line in (proc.stdout or proc.stderr).strip().splitlines() if line.strip()]
    return text[0].strip()[:200] if text else ""


def checks() -> list[ToolCheck]:
    results: list[ToolCheck] = []
    tables: tuple[tuple[Kind, tuple[tuple[str, list[str]], ...]], ...] = (
        ("required", REQUIRED),
        ("optional", OPTIONAL),
        ("vendor", VENDOR),
    )
    for kind, table in tables:
        for name, argv in table:
            version = _probe(argv)
            if version is not None:
                results.append(ToolCheck(name=name, status="ok", kind=kind, version=version))
            elif kind == "required":
                results.append(ToolCheck(name=name, status="fail", kind=kind))
            else:
                results.append(
                    ToolCheck(
                        name=name, status="warn" if kind == "optional" else "absent", kind=kind
                    )
                )
    return results
