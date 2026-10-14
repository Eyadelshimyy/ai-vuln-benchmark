"""Advisory mutation testing for deterministic gate modules (stdlib only).

Structural coverage shows that tests execute a condition; mutation testing
shows that they would notice if the condition were wrong. Each mutant
changes one operator in a target module, the targeted tests run against it,
and the mutant is *killed* when they fail. A *surviving* mutant points at a
boundary or connective that no assertion pins.

Operators (the classic relational and logical replacements):

* ROR: ``<`` <-> ``<=``, ``>`` <-> ``>=``, ``==`` <-> ``!=``,
  ``in`` <-> ``not in``, ``is`` <-> ``is not``;
* LCR: ``and`` <-> ``or``;
* BCR: ``return True`` <-> ``return False``.

Mutants are compiled in memory through the same ``SourceFileLoader`` hook
the structural coverage runner uses, so the working tree is never written.
The report is advisory: survivors never fail the run. An unmeasurable run
(the unmutated tests fail) exits non-zero, because no score can be trusted.

Usage::

    python scripts/mutation_probe.py run [--json PATH] [--max N] [--jobs N]

Targets and tests live in ``[tool.vibebb-mutation]`` of ``pyproject.toml``.
The module is canonical across the VibeBB family.
"""

from __future__ import annotations

import argparse
import ast
import copy
import importlib.machinery
import json
import os
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MUTANT_ENV = "VIBEBB_MUTANT"
DEFAULT_MAX = 400
SKIP_MARK = "pragma: no mutate"
SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
SYMBOLS: dict[type[ast.AST], str] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.And: "and",
    ast.Or: "or",
}


@dataclass(frozen=True)
class Site:
    path: str
    index: int
    line: int
    col: int
    operator: str
    before: str
    after: str


def _candidates(tree: ast.AST) -> Iterator[tuple[ast.AST, int | None]]:
    """Yield mutable nodes in a deterministic order (``ast.walk`` order)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in SWAPS:
                    yield node, i
        elif isinstance(node, ast.BoolOp) or (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, bool)
        ):
            yield node, None


def sites(source: str, relpath: str) -> list[Site]:
    tree = ast.parse(source, filename=relpath)
    lines = source.splitlines()
    found: list[Site] = []
    for index, (node, op_index) in enumerate(_candidates(tree)):
        line = getattr(node, "lineno", 0)
        if 0 < line <= len(lines) and SKIP_MARK in lines[line - 1]:
            continue
        col = getattr(node, "col_offset", 0)
        if isinstance(node, ast.Compare) and op_index is not None:
            op = type(node.ops[op_index])
            found.append(
                Site(relpath, index, line, col, "ROR", SYMBOLS[op], SYMBOLS[SWAPS[op]])
            )
        elif isinstance(node, ast.BoolOp):
            before = SYMBOLS[type(node.op)]
            after = "or" if before == "and" else "and"
            found.append(Site(relpath, index, line, col, "LCR", before, after))
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            value = bool(node.value.value)
            found.append(
                Site(
                    relpath,
                    index,
                    line,
                    col,
                    "BCR",
                    f"return {value}",
                    f"return {not value}",
                )
            )
    return sorted(found, key=lambda s: (s.line, s.col, s.index))


def mutate(source: str, relpath: str, index: int) -> ast.Module:
    """Return the module tree with candidate ``index`` mutated."""
    tree = copy.deepcopy(ast.parse(source, filename=relpath))
    for i, (node, op_index) in enumerate(_candidates(tree)):
        if i != index:
            continue
        if isinstance(node, ast.Compare) and op_index is not None:
            node.ops[op_index] = SWAPS[type(node.ops[op_index])]()
        elif isinstance(node, ast.BoolOp):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            node.value = ast.Constant(value=not node.value.value)
        ast.fix_missing_locations(tree)
        return tree
    raise ValueError(f"{relpath}: no mutation site {index}")


def install(path: Path, index: int) -> None:
    """Serve the mutated module for ``path`` from any SourceFileLoader."""
    target = path.resolve()
    original = importlib.machinery.SourceFileLoader.get_code

    def get_code(self: importlib.machinery.SourceFileLoader, fullname: str) -> Any:
        if Path(self.get_filename(fullname)).resolve() != target:
            return original(self, fullname)
        tree = mutate(target.read_text(encoding="utf-8"), target.name, index)
        return compile(tree, str(target), "exec", dont_inherit=True)

    importlib.machinery.SourceFileLoader.get_code = get_code  # type: ignore[method-assign]


def _activate() -> None:
    raw = os.environ.get(MUTANT_ENV)
    if raw:
        spec = json.loads(raw)
        install(Path(spec["path"]), int(spec["index"]))


_activate()


# --------------------------------------------------------------------------
# Runner.


@dataclass(frozen=True)
class Config:
    root: Path
    targets: list[str]
    tests: list[str]
    max_mutants: int


def load_config(root: Path) -> Config:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    section = data.get("tool", {}).get("vibebb-mutation", {})
    targets = [str(t) for t in section.get("targets", [])]
    tests = [str(t) for t in section.get("tests", [])]
    if not targets or not tests:
        raise SystemExit("[tool.vibebb-mutation] needs non-empty targets and tests")
    return Config(root, targets, tests, int(section.get("max_mutants", DEFAULT_MAX)))


def sample(found: list[Site], limit: int) -> list[Site]:
    """Evenly spaced deterministic sample of at most ``limit`` sites."""
    if limit <= 0 or len(found) <= limit:
        return found
    step = len(found) / limit
    return [found[int(i * step)] for i in range(limit)]


def _pytest(config: Config, env: dict[str, str], timeout: float | None) -> str:
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-p",
        "mutation_probe",
        *config.tests,
    ]
    try:
        result = subprocess.run(
            argv,
            cwd=config.root,
            env=env,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if result.returncode == 0:
        return "survived"
    if result.returncode == 1:
        return "killed"
    return "error"


def _env(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    here = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (here, env.get("PYTHONPATH")) if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def score(statuses: list[str]) -> float | None:
    killed = sum(1 for s in statuses if s in {"killed", "timeout"})
    judged = killed + statuses.count("survived")
    return None if judged == 0 else 100.0 * killed / judged


def run(config: Config, limit: int, jobs: int, out: Path | None) -> int:
    started = time.monotonic()
    baseline = _pytest(config, _env({}), None)
    duration = time.monotonic() - started
    if baseline != "survived":
        print(f"mutation probe: unmutated tests did not pass ({baseline}); no score")
        return 1
    timeout = max(10.0, 3.0 * duration + 5.0)
    found: list[Site] = []
    for target in config.targets:
        path = config.root / target
        found.extend(sites(path.read_text(encoding="utf-8"), target))
    chosen = sample(found, limit)

    def one(site: Site) -> str:
        spec = json.dumps({"path": str(config.root / site.path), "index": site.index})
        return _pytest(config, _env({MUTANT_ENV: spec}), timeout)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        statuses = list(pool.map(one, chosen))
    survivors = [
        s for s, status in zip(chosen, statuses, strict=True) if status == "survived"
    ]
    value = score(statuses)
    counts = {k: statuses.count(k) for k in ("killed", "timeout", "survived", "error")}
    lines = [
        f"mutation probe (advisory): {len(chosen)} of {len(found)} sites, "
        f"tests: {' '.join(config.tests)}",
        "  " + ", ".join(f"{k} {v}" for k, v in counts.items()),
        "  score " + ("unknown" if value is None else f"{value:.2f}%"),
    ]
    lines += [
        f"  survived {s.path}:{s.line}:{s.col} {s.operator} {s.before} -> {s.after}"
        for s in survivors
    ]
    text = "\n".join(lines)
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "advisory": True,
            "sites_total": len(found),
            "sites_run": len(chosen),
            "counts": counts,
            "score_percent": value,
            "survivors": [s.__dict__ for s in survivors],
            "results": [
                {**s.__dict__, "status": status}
                for s, status in zip(chosen, statuses, strict=True)
            ],
        }
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a", encoding="utf-8") as handle:
            handle.write(f"```text\n{text}\n```\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advisory mutation probe.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="mutate the targets and run the tests")
    run_p.add_argument("--root", type=Path, default=Path.cwd())
    run_p.add_argument("--json", type=Path, default=None)
    run_p.add_argument("--max", type=int, default=None)
    run_p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    config = load_config(args.root.resolve())
    limit = config.max_mutants if args.max is None else args.max
    return run(config, limit, args.jobs, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
