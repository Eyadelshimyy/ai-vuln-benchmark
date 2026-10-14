"""Structural coverage of RTL simulations against authored line/branch floors.

Verilog testbenches are rebuilt with Verilator ``--binary --timing
--coverage-line``: ``line`` points are executed blocks and ``branch`` points
are if/else and case arms. VHDL runs are elaborated with NVC
``--cover=statement,branch`` and exported as Cobertura XML: a branch is a
decision, covered only when every outcome was taken. Only points in the
``measure`` design sources count; testbench code never inflates the figure.
A missing tool, a failing run, unreadable coverage data or no point in the
measured sources fails the gate.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .contract import FpgaContract, Simulation, resolve
from .flow import TOOL_TIMEOUT_S, nvc_analyse, nvc_elaborate_argv
from .hdl import all_files, include_dirs, is_systemverilog, units
from .toolrun import run_tool

Counts = dict[str, list[int]]
_POINT = re.compile(r"^C '(?P<key>.*)' (?P<count>[0-9]+)$")


@dataclass
class CoverageResult:
    ok: bool
    detail: str
    evidence: list[str] = field(default_factory=list[str])


def _empty() -> Counts:
    return {"line": [0, 0], "branch": [0, 0]}


def _add(counts: Counts, kind: str, hit: bool) -> None:
    counts[kind][0] += 1
    counts[kind][1] += int(hit)


def verilator_counts(text: str, measured: set[Path]) -> Counts:
    """Tally ``coverage.dat`` line/branch points whose ``f`` is a measured file."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# SystemC::Coverage"):
        raise ValueError("not a Verilator coverage.dat file")
    counts = _empty()
    for line in lines[1:]:
        if not line.strip():
            continue
        match = _POINT.match(line)
        if match is None:
            raise ValueError(f"unreadable coverage point: {line[:80]!r}")
        fields = dict(
            item.split("\x02", 1) for item in match["key"].split("\x01") if "\x02" in item
        )
        kind = fields.get("t")
        if kind in counts and Path(fields.get("f", "")).resolve() in measured:
            _add(counts, kind, int(match["count"]) > 0)
    return counts


def cobertura_counts(text: str, root: Path, measured: set[Path]) -> Counts:
    """Tally NVC Cobertura lines and decisions of measured files (decision = all outcomes)."""
    try:
        tree = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"unreadable Cobertura XML: {exc}") from exc
    if tree.tag != "coverage":
        raise ValueError("not a Cobertura coverage document")
    counts = _empty()
    for cls in tree.iter("class"):
        name = Path(cls.get("filename", ""))
        if (name if name.is_absolute() else root / name).resolve() not in measured:
            continue
        for line in cls.iter("line"):
            hits = line.get("hits")
            if hits is None or not hits.isdigit():
                raise ValueError(f"{name}: line without integer hits")
            _add(counts, "line", int(hits) > 0)
            for condition in line.iter("condition"):
                _add(counts, "branch", condition.get("coverage", "").startswith("100"))
    return counts


def _pct(counts: Counts, kind: str) -> float:
    total, hit = counts[kind]
    return 100.0 if total == 0 else 100.0 * hit / total


def _verilator(
    contract: FpgaContract, contract_path: Path, sim: Simulation, work: Path, log: Path
) -> tuple[str | None, Path]:
    obj = work / "verilator"
    data = work / "coverage.dat"
    data.unlink(missing_ok=True)
    unit_list = units(contract, contract_path, sim.sources)
    build = [
        "verilator",
        "--binary",
        "--timing",
        "--coverage-line",
        "-Wno-fatal",
        *(["-sv"] if is_systemverilog(unit_list) else []),
        "--top-module",
        sim.top,
        "-Mdir",
        obj.as_posix(),
        "-o",
        "simv",
        *[f"-I{p.as_posix()}" for p in include_dirs(contract, contract_path)],
        *[f"-D{k}={v}" for k, v in contract.build.defines.items()],
        *[p.as_posix() for p in all_files(unit_list)],
    ]
    built = run_tool(build, contract_path.parent, log, TOOL_TIMEOUT_S)
    if not built.ok:
        return f"verilator build: {built.detail}", data
    ran = run_tool(
        [(obj / "simv").as_posix(), f"+verilator+coverage+file+{data.as_posix()}"],
        contract_path.parent,
        log,
        sim.timeout_s,
    )
    return (None if ran.ok else f"run: {ran.detail}"), data


def _nvc(
    contract: FpgaContract, contract_path: Path, sim: Simulation, work: Path, log: Path
) -> tuple[str | None, Path]:
    lib_dir = work / "nvc"
    database = work / f"{sim.id}.ncdb"
    xml = work / f"{sim.id}.cobertura.xml"
    xml.unlink(missing_ok=True)
    root = contract_path.parent
    unit_list = units(contract, contract_path, sim.sources)
    analysed = nvc_analyse(contract, unit_list, lib_dir, root, log)
    if not analysed.ok:
        return f"analysis: {analysed.detail}", xml
    argv = nvc_elaborate_argv(contract, lib_dir, sim.top)
    at = argv.index("-e") + 1
    argv[at:at] = ["--cover=statement,branch", f"--cover-file={database.as_posix()}"]
    argv += ["-r", "--exit-severity=error"]
    if sim.stop_time:
        argv.append(f"--stop-time={sim.stop_time}")
    ran = run_tool(argv, root, log, sim.timeout_s)
    if not ran.ok:
        return f"run: {ran.detail}", xml
    export = ["nvc", "--cover-export", "--format=cobertura", "-o", xml.as_posix()]
    exported = run_tool([*export, database.as_posix()], root, log, TOOL_TIMEOUT_S)
    return (None if exported.ok else f"export: {exported.detail}"), xml


def run_coverage(
    contract: FpgaContract, contract_path: Path, sim: Simulation, out_dir: Path
) -> CoverageResult:
    assert sim.coverage is not None
    cover = sim.coverage
    root = contract_path.parent
    measured = {resolve(contract_path, path).resolve() for path in cover.measure}
    absent = sorted(path for path in cover.measure if not (root / path).is_file())
    if absent:
        return CoverageResult(False, f"measured source(s) missing: {', '.join(absent)}")
    work = resolve(contract_path, contract.build.dir) / f"cov-{sim.id}"
    work.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"coverage-{sim.id}.log"
    log.unlink(missing_ok=True)
    runner = _nvc if sim.runner == "nvc" else _verilator
    problem, data = runner(contract, contract_path, sim, work, log)
    if problem is not None:
        return CoverageResult(False, problem)
    try:
        text = data.read_text(encoding="utf-8", errors="replace")
        counts = (
            cobertura_counts(text, root, measured)
            if sim.runner == "nvc"
            else verilator_counts(text, measured)
        )
    except (OSError, ValueError) as exc:
        return CoverageResult(False, f"coverage data: {exc}")
    line, branch = _pct(counts, "line"), _pct(counts, "branch")
    evidence = [
        f"tool={'nvc' if sim.runner == 'nvc' else 'verilator'}",
        f"lines={counts['line'][1]}/{counts['line'][0]} ({line:.1f}%)",
        f"branches={counts['branch'][1]}/{counts['branch'][0]} ({branch:.1f}%)",
    ]
    problems: list[str] = []
    if counts["line"][0] == 0:
        problems.append("no coverage points in the measured sources")
    if line < cover.min_line_pct:
        problems.append(f"line coverage {line:.1f}% < {cover.min_line_pct}%")
    if branch < cover.min_branch_pct:
        problems.append(f"branch coverage {branch:.1f}% < {cover.min_branch_pct}%")
    return CoverageResult(not problems, "; ".join(problems), evidence)
