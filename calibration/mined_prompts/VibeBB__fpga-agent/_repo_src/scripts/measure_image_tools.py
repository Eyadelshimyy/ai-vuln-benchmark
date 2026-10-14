#!/usr/bin/env python3
"""Measure version-reporting commands exposed by the FPGA tools image."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

_IMAGE_REF = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")
_LOCAL_IMAGE_REF = re.compile(r"[a-z0-9][a-z0-9_.-]*:[a-z0-9][a-z0-9_.-]*\Z")

_PROBES = (
    ("python", "python", "--version"),
    ("uv", "uv", "--version"),
    (
        "fpga-agent",
        "python",
        "-c",
        "import importlib.metadata as m; print(m.version('fpga-agent'))",
    ),
    ("ghdl-yosys-plugin", "yosys", "-m", "ghdl", "-p", "help ghdl", "-q"),
    ("yosys", "yosys", "--version"),
    ("nvc", "nvc", "--version"),
    ("iverilog", "iverilog", "-V"),
    ("vvp", "vvp", "-V"),
    ("verilator", "verilator", "--version"),
    ("sby", "sby", "--help"),
    ("yosys-smtbmc", "yosys-smtbmc", "--help"),
    ("yices-smt2", "yices-smt2", "--version"),
    ("nextpnr-ice40", "nextpnr-ice40", "--version"),
    ("nextpnr-ecp5", "nextpnr-ecp5", "--version"),
    ("nextpnr-himbaechel", "nextpnr-himbaechel", "--version"),
    ("icepack", "icepack", "-h"),
    ("ecppack", "ecppack", "--help"),
    ("gowin_pack", "gowin_pack", "-h"),
    ("ghdl", "ghdl", "--version"),
    ("z3", "z3", "--version"),
    ("boolector", "boolector", "--version"),
    ("openFPGALoader", "openFPGALoader", "-V"),
)

_REQUIRED_TOOLS = {
    "ghdl-yosys-plugin",
    "yosys",
    "nvc",
    "iverilog",
    "vvp",
    "verilator",
    "sby",
    "yosys-smtbmc",
    "yices-smt2",
    "nextpnr-ice40",
    "nextpnr-ecp5",
    "nextpnr-himbaechel",
    "icepack",
    "ecppack",
    "gowin_pack",
}


def _probe_script() -> str:
    lines = [
        "set -u",
        "probe() {",
        "  name=$1",
        "  shift",
        "  executable=$1",
        '  if ! command -v "$executable" >/dev/null 2>&1; then return 0; fi',
        '  if output=$("$@" 2>&1); then status=0; else status=$?; fi',
        "  version=$(printf '%s\\n' \"$output\" | sed -n '1p')",
        '  if [ -z "$version" ]; then version=$(command -v "$executable"); fi',
        '  printf \'%s\\t%s\\t%s\\n\' "$name" "$*" "$version"',
        "}",
    ]
    lines.extend(
        "probe " + " ".join("'" + part.replace("'", "'\\''") + "'" for part in probe)
        for probe in _PROBES
    )
    return "\n".join(lines)


def measure(image_ref: str) -> dict[str, str]:
    if _IMAGE_REF.fullmatch(image_ref) is None and _LOCAL_IMAGE_REF.fullmatch(image_ref) is None:
        raise ValueError("image ref must be digest-pinned or a local tag")
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "",
            image_ref,
            "sh",
            "-c",
            _probe_script(),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "tool metadata probe failed")
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, command, version = line.split("\t", 2)
        if name and command and version:
            values[name] = f"{command}: {version}"
    required = _REQUIRED_TOOLS | {"python", "uv", "fpga-agent"}
    if required - values.keys():
        raise ValueError(f"metadata probe omitted: {sorted(required - values.keys())}")
    return dict(sorted(values.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = measure(args.image_ref)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
