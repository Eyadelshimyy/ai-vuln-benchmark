"""Regenerate the bundled device profiles from the OSS CAD Suite databases.

Package pin lists come from the open device databases shipped with the
pinned OSS CAD Suite release: Project IceStorm ``chipdb-*.txt`` (ISC),
Project Trellis ``iodb.json`` (ISC) and Project Apicula chip databases
(MIT). Only the facts needed by the pin gate (pin names, banks, dedicated
functions, global-clock capability) are extracted; the databases are not
copied.

Usage: python scripts/extract_device_profiles.py [--suite /opt/oss-cad-suite]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "fpga" / "devices"
DEFAULT_SUITE = Path(os.environ.get("OSS_CAD_SUITE", "/opt/oss-cad-suite"))
SUITE_RELEASE = "2026-09-30"

VCCIO_MAX_V = 3.465
LVCMOS = ["LVCMOS33", "LVCMOS25", "LVCMOS18", "LVCMOS15", "LVCMOS12"]

ICE40_RESOURCES = {
    "ICESTORM_LC": "logic",
    "ICESTORM_RAM": "ram",
    "ICESTORM_SPRAM": "ram",
    "ICESTORM_DSP": "dsp",
    "SB_IO": "io",
    "SB_GB": "clock",
    "ICESTORM_PLL": "pll",
    "ICESTORM_HFOSC": "other",
    "ICESTORM_LFOSC": "other",
    "SB_I2C": "other",
    "SB_SPI": "other",
    "SB_LEDDA_IP": "other",
    "SB_RGBA_DRV": "other",
    "SB_WARMBOOT": "other",
    "IO_I3C": "other",
}
ECP5_RESOURCES = {
    "TRELLIS_COMB": "logic",
    "TRELLIS_FF": "logic",
    "TRELLIS_RAMW": "ram",
    "DP16KD": "ram",
    "MULT18X18D": "dsp",
    "ALU54B": "dsp",
    "TRELLIS_IO": "io",
    "IOLOGIC": "io",
    "SIOLOGIC": "io",
    "DCCA": "clock",
    "DCSC": "clock",
    "TRELLIS_ECLKBUF": "clock",
    "ECLKSYNCB": "clock",
    "ECLKBRIDGECS": "clock",
    "CLKDIVF": "clock",
    "EHXPLLL": "pll",
    "DLLDELD": "other",
    "DDRDLL": "other",
    "DQSBUFM": "other",
    "DCUA": "other",
    "EXTREFB": "other",
    "PCSCLKDIV": "other",
    "DTR": "other",
    "GSR": "other",
    "JTAGG": "other",
    "OSCG": "other",
    "SEDGA": "other",
    "USRMCLK": "other",
}
GOWIN_RESOURCES = {
    "LUT4": "logic",
    "DFF": "logic",
    "ALU": "logic",
    "MUX2_LUT5": "logic",
    "MUX2_LUT6": "logic",
    "MUX2_LUT7": "logic",
    "MUX2_LUT8": "logic",
    "BSRAM": "ram",
    "RAM16SDP4": "ram",
    "MULT9X9": "dsp",
    "MULT18X18": "dsp",
    "MULT36X36": "dsp",
    "MULTALU18X18": "dsp",
    "MULTALU36X18": "dsp",
    "MULTADDALU18X18": "dsp",
    "ALU54D": "dsp",
    "PADD9": "dsp",
    "PADD18": "dsp",
    "IOB": "io",
    "IOLOGICI": "io",
    "IOLOGICO": "io",
    "IDES16": "io",
    "OSER16": "io",
    "MIPI_IBUF": "io",
    "MIPI_OBUF": "io",
    "BUFG": "clock",
    "DCS": "clock",
    "DQCE": "clock",
    "DHCEN": "clock",
    "CLKDIV": "clock",
    "CLKDIV2": "clock",
    "rPLL": "pll",
    "OSC": "other",
    "FLASH608K": "other",
    "GSR": "other",
    "GND": "other",
    "VCC": "other",
}

# iCE40 UltraPlus SG48 SPI configuration pins, from the Lattice iCE40
# UltraPlus Family Data Sheet (FPGA-DS-02008) pinout: IOB_32a_SPI_SO (14),
# IOB_34a_SPI_SCK (15), IOB_35b_SPI_SS (16), IOB_33b_SPI_SI (17).
ICE40_UP5K_SG48_CONFIG = {"14": "SPI_SO", "15": "SPI_SCK", "16": "SPI_SS", "17": "SPI_SI"}
ECP5_CONFIG_FUNCTIONS = ("MOSI", "MISO", "CSN", "CS1N", "HOLDN", "DOUT", "WRITEN", "D0/", "D1/")
GOWIN_JTAG = {"TMS", "TCK", "TDI", "TDO", "JTAGSEL_N"}
GOWIN_CONFIG_PREFIXES = ("MSPI", "SSPI", "RECONFIG", "READY", "DONE", "MODE", "CPU", "SCLK")

_APICULA_PINOUT = """
import json, os, sys, apycula
from apycula import chipdb
root = os.path.dirname(apycula.__file__)
db = chipdb.load_chipdb(os.path.join(root, sys.argv[1] + ".msgpack.xz"))
print(json.dumps(db.pinout[sys.argv[2]][sys.argv[3]]))
"""


def _ice40_pins(suite: Path, chip: str, package: str) -> list[dict[str, Any]]:
    lines = (suite / "share" / "icebox" / f"chipdb-{chip}.txt").read_text().splitlines()
    gbuf: set[tuple[int, int, int]] = set()
    pins: list[tuple[str, int, int, int]] = []
    section = ""
    for line in lines:
        if line.startswith("."):
            section = line.strip()
            continue
        fields = line.split()
        if section == f".pins {package}" and len(fields) == 4:
            pins.append((fields[0], int(fields[1]), int(fields[2]), int(fields[3])))
        elif section == ".gbufpin" and len(fields) == 4:
            gbuf.add((int(fields[0]), int(fields[1]), int(fields[2])))
    result: list[dict[str, Any]] = []
    for name, x, y, z in pins:
        functions: list[str] = []
        caution = None
        if chip == "5k" and package == "sg48" and name in ICE40_UP5K_SG48_CONFIG:
            functions.append(ICE40_UP5K_SG48_CONFIG[name])
            caution = "config"
        clock = (x, y, z) in gbuf
        if clock:
            functions.append("GBIN")
        result.append(
            {"name": name, "bank": None, "functions": functions, "caution": caution, "clock": clock}
        )
    return sorted(result, key=lambda p: int(p["name"]) if p["name"].isdigit() else 0)


def _ecp5_pins(suite: Path, part: str, package: str) -> list[dict[str, Any]]:
    iodb = json.loads(
        (suite / "share" / "trellis" / "database" / "ECP5" / part / "iodb.json").read_text()
    )
    meta = {(m["row"], m["col"], m["pio"]): m for m in iodb["pio_metadata"]}
    result: list[dict[str, Any]] = []
    for name, loc in sorted(iodb["packages"][package].items()):
        info = meta.get((loc["row"], loc["col"], loc["pio"]), {})
        function = info.get("function")
        functions = [function] if function else []
        caution = None
        if function and info.get("bank") == 8 and any(k in function for k in ECP5_CONFIG_FUNCTIONS):
            caution = "config"
        clock = bool(function) and ("PCLK" in function and "GR_" not in function)
        result.append(
            {
                "name": name,
                "bank": info.get("bank"),
                "functions": functions,
                "caution": caution,
                "clock": clock,
            }
        )
    return result


def _pin_key(name: str) -> tuple[str, int]:
    """Sort QFN pins numerically and BGA balls by row letters, then column."""
    row = name.rstrip("0123456789")
    return (row, int(name[len(row) :]))


def _gowin_pins(suite: Path, chip: str, device: str, package: str) -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            str(suite / "py3bin" / "python3"),
            "-W",
            "ignore",
            "-c",
            _APICULA_PINOUT,
            chip,
            device,
            package,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pinout: dict[str, list[Any]] = json.loads(proc.stdout)
    result: list[dict[str, Any]] = []
    for name, (_, functions) in sorted(pinout.items(), key=lambda kv: _pin_key(kv[0])):
        caution = None
        if any(f in GOWIN_JTAG for f in functions):
            caution = "jtag"
        elif any(f.startswith(GOWIN_CONFIG_PREFIXES) for f in functions):
            caution = "config"
        clock = any(f.startswith("GCLK") for f in functions)
        result.append(
            {"name": name, "bank": None, "functions": functions, "caution": caution, "clock": clock}
        )
    return result


def profiles(suite: Path) -> list[dict[str, Any]]:
    common = {"schema_version": 1, "system": "fpga", "artifact_kind": "fpga_device"}
    return [
        {
            **common,
            "id": "ice40-up5k-sg48",
            "vendor": "Lattice",
            "family": "ice40",
            "part": "iCE40UP5K",
            "package": "SG48",
            "boards": ["iCEBreaker", "UPduino v3"],
            "source": f"Project IceStorm chipdb-5k.txt (OSS CAD Suite {SUITE_RELEASE})",
            "nextpnr_args": ["--up5k", "--package", "sg48"],
            "pack_args": [],
            "io_voltage_max_v": VCCIO_MAX_V,
            "io_standards": [],
            "resources": ICE40_RESOURCES,
            "pins": _ice40_pins(suite, "5k", "sg48"),
        },
        *(
            {
                **common,
                "id": f"ecp5-lfe5u-{size.lower()}-cabga381",
                "vendor": "Lattice",
                "family": "ecp5",
                "part": f"LFE5U-{size}",
                "package": "CABGA381",
                "speed": "6",
                "boards": boards,
                "source": f"Project Trellis LFE5U-{size}/iodb.json (OSS CAD Suite {SUITE_RELEASE})",
                "nextpnr_args": [flag, "--package", "CABGA381", "--speed", "6"],
                "pack_args": [],
                "io_voltage_max_v": VCCIO_MAX_V,
                "io_standards": [*LVCMOS, "LVDS", "SSTL135_I", "SSTL15_I"],
                "resources": ECP5_RESOURCES,
                "pins": _ecp5_pins(suite, f"LFE5U-{size}", "CABGA381"),
            }
            for size, flag, boards in (
                ("25F", "--25k", ["ULX3S (25F)"]),
                ("85F", "--85k", ["ULX3S (85F)", "OrangeCrab 85F"]),
            )
        ),
        {
            **common,
            "id": "gowin-gw1nr9c-qn88p",
            "vendor": "Gowin",
            "family": "gowin",
            "part": "GW1NR-LV9QN88PC6/I5",
            "package": "QN88P",
            "speed": "C6/I5",
            "boards": ["Sipeed Tang Nano 9K"],
            "source": (
                f"Project Apicula GW1N-9C chipdb, GW1NR-9C QFN88P (OSS CAD Suite {SUITE_RELEASE})"
            ),
            "synth_args": ["-family", "gw1n"],
            "nextpnr_args": ["--device", "GW1NR-LV9QN88PC6/I5", "--vopt", "family=GW1N-9C"],
            "pack_args": ["-d", "GW1N-9C"],
            "io_voltage_max_v": VCCIO_MAX_V,
            "io_standards": [*LVCMOS, "LVDS25"],
            "resources": GOWIN_RESOURCES,
            "pins": _gowin_pins(suite, "GW1N-9C", "GW1NR-9C", "QFN88P"),
        },
        *(
            {
                **common,
                "id": profile_id,
                "vendor": "Gowin",
                "family": "gowin",
                "part": part,
                "package": package,
                "speed": "C8/I7",
                "boards": boards,
                "source": (
                    f"Project Apicula GW2A-18C chipdb, {device} {apicula_package} "
                    f"(OSS CAD Suite {SUITE_RELEASE})"
                ),
                "synth_args": ["-family", "gw2a"],
                "nextpnr_args": ["--device", part, "--vopt", "family=GW2A-18C"],
                "pack_args": ["-d", "GW2A-18C"],
                "io_voltage_max_v": VCCIO_MAX_V,
                "io_standards": [*LVCMOS, "LVDS25"],
                "resources": GOWIN_RESOURCES,
                "pins": _gowin_pins(suite, "GW2A-18C", device, apicula_package),
            }
            for profile_id, part, package, device, apicula_package, boards in (
                (
                    "gowin-gw2ar18c-qn88p",
                    "GW2AR-LV18QN88C8/I7",
                    "QN88P",
                    "GW2AR-18C",
                    "QFN88P",
                    ["Sipeed Tang Nano 20K"],
                ),
                (
                    "gowin-gw2a18c-pbga256",
                    "GW2A-LV18PG256C8/I7",
                    "PG256",
                    "GW2A-18C",
                    "PBGA256",
                    ["Sipeed Tang Primer 20K"],
                ),
            )
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for profile in profiles(args.suite):
        path = OUT / f"{profile['id']}.json"
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {len(profile['pins'])} pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
