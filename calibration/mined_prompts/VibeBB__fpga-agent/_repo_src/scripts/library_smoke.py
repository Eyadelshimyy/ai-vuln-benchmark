"""Synthesize every entity/module top of the declared HDL libraries.

For each external library this resolves every unit that nothing else in
the library references (a "top") through the same ``components`` resolver
the contract gates use, then runs ``yosys`` ``synth_<family>`` on it.
Units needing mandatory generics cannot elaborate bare and are recorded
as failures with their elaboration error — that is expected, and marks
them instantiation-only.

Usage:
    library_smoke.py [--contract CONTRACT] [--family gowin] [--out DIR]
                     [--libraries name[,name...]] [--timeout SECONDS]

Without ``--contract`` the registered libraries in
``scripts/hdl_libraries.json`` are swept (hdl-modules expands to one VHDL
library per module ``src/`` directory, ordered by inter-module deps).
Run on a networked host after ``fetch_hdl_libraries.py``, or inside the
fpga-tools image; only yosys + ghdl are needed (no P&R).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpga.contract import Language, Library, resolve
from fpga.flow import SYNTH
from fpga.libraries import resolve_library, scan_library

REGISTRY = Path(__file__).resolve().parent / "hdl_libraries.json"
REPO = Path(__file__).resolve().parents[1]
BUILTIN = {"ieee", "ieee_proposed", "std", "work"}


def _lib(name: str, root: str, language: Language, exclude: list[str] | None = None) -> Library:
    return Library(
        name=name.replace("-", "_"),
        root=root,
        language=language,
        exclude=exclude or [],
        license="smoke",
        source_url="https://smoke.invalid",
        revision="smoke00",
        components=["all"],
    )


def _hdl_modules_libs(root: Path) -> list[Library]:
    """One VHDL library per hdl-modules module, dependency-ordered."""
    modules = {d.name: d / "src" for d in (root / "modules").iterdir() if (d / "src").is_dir()}
    indexes = {name: scan_library(path, "vhdl") for name, path in modules.items()}
    order: list[str] = []
    seen: set[str] = set()

    def deps(name: str) -> set[str]:
        libs = {
            ref.library
            for refs in indexes[name].refs.values()
            for ref in refs
            if ref.library not in BUILTIN and ref.library != name
        }
        return libs & set(modules)

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dep in sorted(deps(name)):
            visit(dep)
        order.append(name)

    for name in sorted(modules):
        visit(name)
    return [_lib(name, f"third_party/hdl-modules/modules/{name}/src", "vhdl") for name in order]


def registry_libraries() -> list[Library]:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    libs: list[Library] = []
    for name in data:
        if name == "hdl-modules":
            libs += _hdl_modules_libs(REPO / "third_party" / name)
            continue
        root = f"third_party/{name}"
        if name == "open-logic":
            libs.append(_lib("olo", root, "vhdl"))
        elif name == "serv":
            libs.append(_lib(name, root, "verilog", ["bench/**", "servant/**", "data/**"]))
        elif name == "picorv32":
            libs.append(_lib(name, root, "verilog", ["scripts/**", "picosoc/**", "testbench*.v"]))
        elif name == "neorv32":
            libs.append(
                _lib(name, root, "vhdl", ["rtl/test_setups/**", "rtl/system_integration/**"])
            )
        elif name == "hazard3":
            libs.append(_lib(name, f"{root}/hdl", "verilog"))
        elif name == "vexriscv-pydata":
            libs.append(
                _lib(
                    "vexriscv",
                    f"{root}/pythondata_cpu_vexriscv/verilog",
                    "verilog",
                )
            )
        elif (REPO / root / "rtl").is_dir():
            libs.append(_lib(name, f"{root}/rtl", "verilog"))
        else:
            libs.append(_lib(name, root, "vhdl" if name == "colibri" else "verilog"))
    return libs


def contract_libraries(contract_path: Path) -> list[Library]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return [Library(**entry) for entry in contract.get("libraries", [])]


def entity_tops(lib: Library, root: Path, language: str) -> list[str]:
    """Units nothing in the library references (entities/modules only)."""
    index = scan_library(root, language, lib.exclude)
    kind_ok = {"entity"} if language == "vhdl" else {"module"}
    referenced = {ref.unit for refs in index.refs.values() for ref in refs}
    return sorted(
        unit
        for unit in index.providers
        if index.kinds.get(unit) in kind_ok and unit not in referenced
    )


def closure_groups(
    contract_path: Path,
    lib: Library,
    top: str,
    declared: list[str],
    libs_by_name: dict[str, Library],
) -> tuple[list[tuple[str, list[Path]]], list[str]]:
    """Resolve ``top`` plus every external library it needs.

    Mirrors how a contract compiles several ``libraries[]`` entries: each
    dependency library contributes its own resolved file list, ordered
    before the unit that references it (``declared`` order guarantees it).
    Returns ((vlib, files) groups in compile order, problems).
    """
    probe = lib.model_copy(update={"components": [top]})
    res = resolve_library(contract_path, probe, declared)
    if res.problems and any("ambiguous" in p for p in res.problems):
        # Variant libraries (e.g. pythondata-cpu-vexriscv: every file declares
        # `module VexRiscv`) are selected by file, not by unit name.
        root = resolve(contract_path, lib.root)
        index = scan_library(root, lib.language, lib.exclude)
        cands = index.ambiguous.get(top.lower()) or []
        if cands:
            rel = str(cands[0].resolve().relative_to(root))
            probe = lib.model_copy(update={"components": [], "files": [rel]})
            res = resolve_library(contract_path, probe, declared)
    if res.problems:
        return [], res.problems
    groups = [(lib.name, res.files)]
    problems: list[str] = []
    seen = {lib.name}
    queue = list(res.external)
    while queue:
        dep = queue.pop(0)
        if dep in seen:
            continue
        seen.add(dep)
        dep_lib = libs_by_name.get(dep)
        if dep_lib is None:
            problems.append(f"{lib.name}: needs undeclared library {dep!r}")
            continue
        dres = resolve_library(contract_path, dep_lib, declared)
        if dres.problems:
            problems += dres.problems
            continue
        groups.append((dep, dres.files))
        queue += sorted(dres.external - seen)
    position = {name: i for i, name in enumerate(declared)}
    groups.sort(key=lambda g: position.get(g[0], len(position)))
    return groups, problems


def synth_unit(
    lib: Library,
    top: str,
    res_files: list[tuple[str, list[Path]]],
    family: str,
    workdir: Path,
    log: Path,
    timeout: int,
    contract_path: Path,
) -> dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    if lib.language == "vhdl":
        parts = ["ghdl --std=08 -frelaxed"]
        for lname, lfiles in res_files:
            parts.append(f"--work={lname} " + " ".join(p.as_posix() for p in lfiles))
        read = " ".join(parts) + f" -e {lib.name}.{top}"
        synth = f"{SYNTH[family]} -top {top}"
    else:
        flat = [p for _, lfiles in res_files for p in lfiles]
        root_dir = resolve(contract_path, lib.root)
        incdirs = {root_dir, *(p.parent for p in flat)}
        inc = " ".join(f"-I{p}" for p in sorted(incdirs))
        real_top = top
        for path in flat:
            names = re.findall(r"^\s*module\s+([a-zA-Z_][a-zA-Z0-9_$]*)", path.read_text(), re.M)
            if any(n.lower() == top for n in names):
                real_top = next(n for n in names if n.lower() == top)
                break
        read = f"read_verilog {inc} " + " ".join(p.as_posix() for p in flat)
        read += f"; hierarchy -check -top {real_top}"
        synth = f"{SYNTH[family]} -top {real_top}"
    script = workdir / "synth.ys"
    script.write_text(f"{read}; {synth}; stat\n", encoding="utf-8")
    argv = ["yosys", "-s", "synth.ys"]
    if lib.language == "vhdl":
        argv = ["yosys", "-m", "ghdl", "-s", "synth.ys"]
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        output = proc.stdout + proc.stderr
        secs = round(time.time() - started, 1)
    except subprocess.TimeoutExpired:
        ok, output, secs = False, "TIMEOUT", float(timeout)
    log.write_text(output, encoding="utf-8")
    cells = re.findall(r"^\s*(\d+)\s+cells\s*$", output, re.M)
    tail = [ln for ln in output.splitlines() if ln.strip()]
    return {
        "lib": lib.name,
        "top": top,
        "files": sum(len(f) for _, f in res_files),
        "ok": ok,
        "cells": int(cells[-1]) if cells else None,
        "warnings": output.count("Warning:"),
        "seconds": secs,
        "error": "" if ok else (tail[-1] if tail else "")[:200],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, help="contract whose libraries[] to sweep")
    parser.add_argument("--family", default="gowin", choices=sorted(SYNTH))
    parser.add_argument("--out", type=Path, default=Path("library-smoke"))
    parser.add_argument("--libraries", type=str, help="comma-separated library name filter")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    base = args.contract.resolve().parent if args.contract else REPO
    contract_path = args.contract.resolve() if args.contract else (REPO / "smoke.fpga.json")
    libs = contract_libraries(args.contract.resolve()) if args.contract else registry_libraries()
    declared = [lib.name for lib in libs]
    roots = {lib.name: (base / lib.root).resolve() for lib in libs}
    sweep = libs
    if args.libraries:
        wanted = set(args.libraries.split(","))
        sweep = [lib for lib in libs if lib.name in wanted]

    results: list[dict[str, object]] = []
    libs_by_name = {lib.name: lib for lib in libs}
    for lib in sweep:
        root = roots[lib.name]
        if not root.is_dir():
            print(f"## {lib.name}: root {root} missing — run fetch_hdl_libraries.py")
            continue
        tops = entity_tops(lib, root, lib.language)
        print(f"## {lib.name}: {len(tops)} tops")
        for top in tops:
            groups, problems = closure_groups(contract_path, lib, top, declared, libs_by_name)
            workdir = args.out / lib.name / top
            if problems:
                results.append(
                    {
                        "lib": lib.name,
                        "top": top,
                        "files": 0,
                        "ok": False,
                        "cells": None,
                        "warnings": 0,
                        "seconds": 0.0,
                        "error": "; ".join(problems)[:200],
                    }
                )
                print(f"  {top}: RESOLVE-FAIL {problems[0][:80]}")
                continue
            row = synth_unit(
                lib,
                top,
                groups,
                args.family,
                workdir,
                workdir / "synth.log",
                args.timeout,
                contract_path,
            )
            results.append(row)
            status = "OK" if row["ok"] else "FAIL"
            print(f"  {top}: {status} cells={row['cells']} {row['seconds']}s")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    ok = sum(1 for r in results if r["ok"])
    print(f"DONE: {ok}/{len(results)} tops synthesized -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
