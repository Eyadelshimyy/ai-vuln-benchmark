"""Host-only board programming with openFPGALoader.

Programming changes real hardware, so it is never a gate and never an MCP
tool. It runs only when the last full gate report for the current
contract passed, the bitstream on disk still has the reported SHA-256, and
the caller repeats that SHA-256 as confirmation.

``production_export`` hands the same gated bitstream and loader options to
production-engineering-agent; it writes JSON only and touches no hardware.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .contract import FpgaContract, Programmer, resolve
from .devices import DeviceProfile
from .flow import sha256
from .gates import GateReport
from .interchange import FpgaProduction, sha256_file


@dataclass
class ProgramPlan:
    argv: list[str]
    bitstream: Path
    sha256: str


@dataclass
class GatedBitstream:
    path: Path
    sha256: str
    report_sha256: str


def gated_bitstream(contract: FpgaContract, contract_path: Path, out_dir: Path) -> GatedBitstream:
    """The bitstream the last passing full gate run for this contract built."""
    report_path = out_dir / f"{contract.name}.fpga-report.json"
    try:
        report = GateReport.model_validate(json.loads(report_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValueError(
            f"no readable gate report ({report_path}); run `fpga gates` first"
        ) from exc
    if report.scope != "full" or report.verdict != "pass" or report.bitstream_sha256 is None:
        raise ValueError("the last full gate run did not pass; run `fpga gates` first")
    if report.contract_sha256 != sha256_file(contract_path):
        raise ValueError("the contract changed since the last gate run; rerun `fpga gates`")
    bitstream = resolve(contract_path, contract.build.bitstream)
    if not bitstream.is_file():
        raise ValueError(f"bitstream {contract.build.bitstream} is missing")
    digest = sha256(bitstream)
    if digest != report.bitstream_sha256:
        raise ValueError("the bitstream changed since the last gate run; rerun `fpga gates`")
    return GatedBitstream(bitstream, digest, sha256(report_path))


def _programmer(contract: FpgaContract) -> Programmer:
    if contract.programmer is None or not (contract.programmer.board or contract.programmer.cable):
        raise ValueError("contract declares no programmer board or cable")
    return contract.programmer


def _loader_argv(programmer: Programmer, bitstream: str) -> list[str]:
    argv = ["openFPGALoader"]
    if programmer.board:
        argv += ["-b", programmer.board]
    if programmer.cable:
        argv += ["-c", programmer.cable]
    if programmer.write_flash:
        argv.append("-f")
    argv.append(bitstream)
    return argv


def plan(
    contract: FpgaContract, contract_path: Path, out_dir: Path, confirm_sha256: str
) -> ProgramPlan:
    programmer = _programmer(contract)
    gated = gated_bitstream(contract, contract_path, out_dir)
    if confirm_sha256 != gated.sha256:
        raise ValueError(f"confirmation must repeat the bitstream sha256 {gated.sha256}")
    return ProgramPlan(_loader_argv(programmer, gated.path.as_posix()), gated.path, gated.sha256)


def production_export(
    contract: FpgaContract, contract_path: Path, profile: DeviceProfile, out_dir: Path
) -> FpgaProduction:
    """The gated bitstream and loader options for the factory programming step."""
    programmer = _programmer(contract)
    gated = gated_bitstream(contract, contract_path, out_dir)
    relative = Path(os.path.relpath(gated.path.resolve(), out_dir.resolve())).as_posix()
    return FpgaProduction(
        design=contract.name,
        contract_sha256=sha256_file(contract_path),
        gate_report_sha256=gated.report_sha256,
        device_ref=contract.device.ref,
        device_profile=profile.id,
        part=profile.part,
        package=profile.package,
        bitstream=relative,
        bitstream_sha256=gated.sha256,
        bitstream_bytes=gated.path.stat().st_size,
        target="flash" if programmer.write_flash else "sram",
        board=programmer.board,
        cable=programmer.cable,
        argv=_loader_argv(programmer, relative),
    )
