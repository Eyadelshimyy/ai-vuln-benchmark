"""RTL simulation runs: NVC for VHDL, Icarus Verilog for Verilog/SV.

A run passes when the simulator exits 0 before the timeout, prints every
``expect`` line in order (substring match per line) and no ``forbid``
line. Assertion failures of severity ``failure``/``$fatal`` end the run
with a non-zero exit, so they fail the gate on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .contract import FpgaContract, Simulation, resolve
from .flow import TOOL_TIMEOUT_S, nvc_analyse, nvc_elaborate_argv
from .hdl import all_files, include_dirs, units
from .toolrun import run_tool


@dataclass
class SimResult:
    ok: bool
    detail: str
    argv: list[str]
    matched: list[str] = field(default_factory=list[str])
    missing: list[str] = field(default_factory=list[str])
    forbidden: list[str] = field(default_factory=list[str])
    transcript: Path | None = None
    waveform: Path | None = None
    seconds: float = 0.0


def progress(lines: list[str], expect: list[str]) -> int:
    """Number of ``expect`` entries matched in order (substring per line)."""
    index = 0
    for line in lines:
        if index < len(expect) and expect[index] in line:
            index += 1
    return index


def run_simulation(
    contract: FpgaContract, contract_path: Path, sim: Simulation, out_dir: Path
) -> SimResult:
    root = contract_path.parent
    work = resolve(contract_path, contract.build.dir) / f"sim-{sim.id}"
    transcript = out_dir / f"sim-{sim.id}.log"
    transcript.unlink(missing_ok=True)
    unit_list = units(contract, contract_path, sim.sources)
    waveform: Path | None = None
    if sim.runner == "nvc":
        lib_dir = work / "nvc"
        analysed = nvc_analyse(contract, unit_list, lib_dir, root, transcript)
        if not analysed.ok:
            return SimResult(False, f"analysis: {analysed.detail}", analysed.argv)
        argv = [*nvc_elaborate_argv(contract, lib_dir, sim.top), "-r", "--exit-severity=error"]
        if sim.stop_time:
            argv.append(f"--stop-time={sim.stop_time}")
        if sim.waveform:
            waveform = out_dir / f"sim-{sim.id}.vcd"
            argv += [f"--wave={waveform.as_posix()}", "--format=vcd"]
    else:
        work.mkdir(parents=True, exist_ok=True)
        image = work / f"{sim.top}.vvp"
        compile_argv = ["iverilog", "-g2012", "-o", image.as_posix(), "-s", sim.top]
        compile_argv += [f"-I{p.as_posix()}" for p in include_dirs(contract, contract_path)]
        compile_argv += [f"-D{k}={v}" for k, v in contract.build.defines.items()]
        compile_argv += [p.as_posix() for p in all_files(unit_list)]
        if sim.waveform:
            waveform = out_dir / f"sim-{sim.id}.vcd"
            dump = work / "fpga_wave_dump.v"
            dump.write_text(
                "module fpga_wave_dump; initial begin "
                f'$dumpfile("{waveform.resolve().as_posix()}"); '
                f"$dumpvars(0, {sim.top}); end endmodule\n",
                encoding="utf-8",
            )
            compile_argv += ["-s", "fpga_wave_dump", dump.as_posix()]
        compiled = run_tool(compile_argv, root, transcript, TOOL_TIMEOUT_S)
        if not compiled.ok:
            return SimResult(False, f"compile: {compiled.detail}", compile_argv)
        argv = ["vvp", "-n", image.as_posix()]
    run = run_tool(argv, root, transcript, sim.timeout_s)
    lines = run.output.splitlines()
    count = progress(lines, sim.expect)
    result = SimResult(
        True,
        "",
        argv,
        matched=sim.expect[:count],
        missing=sim.expect[count:],
        forbidden=[f for f in sim.forbid if any(f in line for line in lines)],
        transcript=transcript,
        waveform=waveform if waveform is not None and waveform.is_file() else None,
        seconds=run.seconds,
    )
    problems: list[str] = []
    if not run.ok:
        problems.append(run.detail)
    if result.missing:
        problems.append(f"expected output not seen: {result.missing[0]!r}")
    if result.forbidden:
        problems.append(f"forbidden output seen: {', '.join(map(repr, result.forbidden))}")
    if sim.waveform and result.waveform is None:
        problems.append("waveform was requested but not written")
    result.ok = not problems
    result.detail = "; ".join(problems)
    return result
