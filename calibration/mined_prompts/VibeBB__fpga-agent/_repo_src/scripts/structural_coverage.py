"""Structural coverage beyond coverage.py: C2, MCC, MC/DC and boundary values.

coverage.py measures statement (C0) and branch (C1) coverage. This module adds
the criteria it cannot see, measured inside the normal pytest run:

* decision coverage including ternaries, comprehension filters, ``assert`` and
  ``match`` guards (coverage.py has no branch arcs for those),
* C2 condition coverage: every atomic condition evaluated true and false,
* MCC multiple-condition coverage: every feasible short-circuit evaluation
  vector of a decision observed (reported, never gated: it grows as 2^n),
* MC/DC: every condition shown to independently flip its decision, using
  unique-cause pairs where short-circuited conditions are don't-care (the
  definition GCC and Clang implement for C),
* 3-value boundary coverage of numeric ``<``/``<=``/``>``/``>=`` conditions:
  operands observed below, on and above the boundary (exact neighbours
  ``-1``/``+1`` when both operands are integers).

Usage::

    python scripts/structural_coverage.py run [--json PATH] [-- PYTEST_ARGS]
    python scripts/structural_coverage.py report --data DIR --coverage-json PATH

Floors live in ``[tool.vibebb-coverage]`` of ``pyproject.toml``; the source
directories come from ``[tool.coverage.run] source``. The module is canonical
across the VibeBB family; ``scripts/check_shared_hooks.py`` pins its hash.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.machinery
import itertools
import json
import math
import operator
import os
import re
import subprocess
import sys
import tempfile
import threading
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

DATA_ENV = "VIBEBB_STRUCTCOV_DATA"
SOURCE_ENV = "VIBEBB_STRUCTCOV_SOURCE"
ROOT_ENV = "VIBEBB_STRUCTCOV_ROOT"
RECORDER = "__vbcc__"
MCC_MAX_CONDITIONS = 10
CRITERIA = ("c0", "c1", "decision", "c2", "mcdc", "boundary")
EXCLUDE_LINE = re.compile(r"#\s*(pragma|PRAGMA)[:\s]?\s*(no|NO)\s*(cover|COVER)")
BOUNDARY_OPS: dict[type[ast.cmpop], str] = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


# --------------------------------------------------------------------------
# Static model: decisions, conditions and their boolean structure.


@dataclass(frozen=True)
class Shape:
    """Boolean structure of a decision: ``and``/``or``/``not`` over leaves.

    A leaf is a condition index or a constant truth value.
    """

    kind: str
    children: tuple[Shape, ...] = ()
    index: int = -1
    value: bool = False

    def to_json(self) -> Any:
        if self.kind == "cond":
            return self.index
        if self.kind == "const":
            return self.value
        return [self.kind, *(child.to_json() for child in self.children)]

    @staticmethod
    def from_json(raw: Any) -> Shape:
        if isinstance(raw, bool):
            return Shape("const", value=raw)
        if isinstance(raw, int):
            return Shape("cond", index=raw)
        kind, *children = raw
        return Shape(str(kind), tuple(Shape.from_json(child) for child in children))


@dataclass
class Decision:
    id: str
    file: str
    line: int
    context: str
    shape: Shape
    conditions: list[str]
    boundaries: dict[int, str] = field(default_factory=dict[int, str])

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file": self.file,
            "line": self.line,
            "context": self.context,
            "shape": self.shape.to_json(),
            "conditions": self.conditions,
            "boundaries": {str(k): v for k, v in self.boundaries.items()},
        }


def _call(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=RECORDER, ctx=ast.Load()), attr=name, ctx=ast.Load()
        ),
        args=list(args),
        keywords=[],
    )


def _is_test_context(node: ast.AST) -> list[tuple[ast.expr, str]]:
    if isinstance(node, ast.If | ast.While):
        return [(node.test, type(node).__name__.lower())]
    if isinstance(node, ast.IfExp):
        return [(node.test, "ternary")]
    if isinstance(node, ast.Assert):
        return [(node.test, "assert")]
    if isinstance(node, ast.comprehension):
        return [(test, "filter") for test in node.ifs]
    if isinstance(node, ast.match_case) and node.guard is not None:
        return [(node.guard, "guard")]
    return []


def _skippable_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Constant):
        return True
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
    )


class Instrumenter(ast.NodeTransformer):
    """Wrap every decision so the recorder sees each condition and outcome."""

    def __init__(self, relpath: str, source: str) -> None:
        self.relpath = relpath
        self.lines = source.splitlines()
        self.decisions: list[Decision] = []

    def _excluded(self, node: ast.AST) -> bool:
        line = getattr(node, "lineno", 0)
        return 0 < line <= len(self.lines) and bool(
            EXCLUDE_LINE.search(self.lines[line - 1])
        )

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.stmt) and self._excluded(node):
            return node
        tests = {id(test): context for test, context in _is_test_context(node)}
        for name, value in ast.iter_fields(node):
            if isinstance(value, list):
                new: list[Any] = []
                for item in value:  # pyright: ignore[reportUnknownVariableType]
                    new.append(self._visit_child(item, tests))
                setattr(node, name, new)
            elif isinstance(value, ast.AST):
                setattr(node, name, self._visit_child(value, tests))
        return node

    def _visit_child(self, child: Any, tests: dict[int, str]) -> Any:
        if not isinstance(child, ast.AST):
            return child
        if isinstance(child, ast.expr) and id(child) in tests:
            if _skippable_test(child):
                return self.visit(child)
            return self._decision(child, tests[id(child)])
        if isinstance(child, ast.BoolOp):
            leaves = list(_leaves(child))
            if sum(not isinstance(leaf, ast.Constant) for leaf in leaves) >= 2:
                return self._decision(child, "value")
        return self.visit(child)

    def _decision(self, expr: ast.expr, context: str) -> ast.expr:
        did = f"{self.relpath}:{expr.lineno}:{expr.col_offset}"
        decision = Decision(did, self.relpath, expr.lineno, context, Shape("const"), [])
        self.decisions.append(decision)
        decision.shape, body = self._rewrite(expr, decision)
        wrapped = _call("d", _call("b", ast.Constant(did)), body)
        return ast.copy_location(wrapped, expr)

    def _rewrite(self, expr: ast.expr, decision: Decision) -> tuple[Shape, ast.expr]:
        if isinstance(expr, ast.BoolOp):
            parts = [self._rewrite(value, decision) for value in expr.values]
            kind = "and" if isinstance(expr.op, ast.And) else "or"
            node = ast.BoolOp(op=expr.op, values=[part[1] for part in parts])
            return Shape(kind, tuple(part[0] for part in parts)), ast.copy_location(
                node, expr
            )
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            shape, operand = self._rewrite(expr.operand, decision)
            node = ast.UnaryOp(op=ast.Not(), operand=operand)
            return Shape("not", (shape,)), ast.copy_location(node, expr)
        if isinstance(expr, ast.Constant):
            return Shape("const", value=bool(expr.value)), expr
        index = len(decision.conditions)
        decision.conditions.append(ast.unparse(expr)[:120])
        did = ast.Constant(decision.id)
        if (
            isinstance(expr, ast.Compare)
            and len(expr.ops) == 1
            and type(expr.ops[0]) in BOUNDARY_OPS
        ):
            op = BOUNDARY_OPS[type(expr.ops[0])]
            decision.boundaries[index] = op
            left = self.visit(expr.left)
            right = self.visit(expr.comparators[0])
            leaf: ast.expr = _call(
                "r", did, ast.Constant(index), left, right, ast.Constant(op)
            )
        else:
            leaf = self.visit(expr)
        return Shape("cond", index=index), ast.copy_location(
            _call("c", did, ast.Constant(index), leaf), expr
        )


def _leaves(expr: ast.expr) -> Iterator[ast.expr]:
    if isinstance(expr, ast.BoolOp):
        for value in expr.values:
            yield from _leaves(value)
    elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        yield from _leaves(expr.operand)
    else:
        yield expr


def instrument(source: str, relpath: str) -> tuple[ast.Module, list[Decision]]:
    tree = ast.parse(source, filename=relpath)
    instrumenter = Instrumenter(relpath, source)
    tree = instrumenter.visit(tree)
    ast.fix_missing_locations(tree)
    return tree, instrumenter.decisions


# --------------------------------------------------------------------------
# Runtime recorder (pytest plugin side).


class Recorder:
    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self.vectors: dict[str, set[tuple[tuple[bool | None, ...], bool]]] = {}
        self.sizes: dict[str, int] = {}
        self.boundary: dict[str, set[str]] = {}

    def _stack(self) -> list[list[Any]]:
        stack: list[list[Any]] | None = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def b(self, did: str) -> int:
        stack = self._stack()
        stack.append([did, [None] * self.sizes.get(did, 0)])
        return len(stack) - 1

    def c(self, did: str, index: int, value: Any) -> Any:
        stack = self._stack()
        for entry in reversed(stack):
            if entry[0] == did:
                vector: list[bool | None] = entry[1]
                if index >= len(vector):
                    vector.extend([None] * (index + 1 - len(vector)))
                vector[index] = bool(value)
                break
        return value

    def r(self, did: str, index: int, left: Any, right: Any, op: str) -> Any:
        result = OPERATORS[op](left, right)
        kind = _boundary_class(left, right)
        if kind is not None:
            with self._lock:
                self.boundary.setdefault(f"{did}#{index}", set()).update(kind)
        return result

    def d(self, token: int, value: Any) -> Any:
        stack = self._stack()
        if token < len(stack):
            did, vector = stack[token]
            del stack[token:]
            with self._lock:
                self.vectors.setdefault(did, set()).add((tuple(vector), bool(value)))
        return value

    def dump(self, path: Path) -> None:
        payload = {
            "vectors": {
                did: sorted([list(vec), out] for vec, out in seen)
                for did, seen in self.vectors.items()
            },
            "boundary": {key: sorted(kinds) for key, kinds in self.boundary.items()},
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _boundary_class(left: Any, right: Any) -> set[str] | None:
    if not (_number(left) and _number(right)):
        return None
    try:
        delta = left - right
    except OverflowError:
        return None
    if isinstance(delta, float) and not math.isfinite(delta):
        return None
    kinds = {"below" if delta < 0 else "on" if delta == 0 else "above"}
    if not (isinstance(left, int) and isinstance(right, int)):
        kinds.add("float")
    else:
        kinds.add("int")
        if delta in (-1, 1):
            kinds.add("below1" if delta < 0 else "above1")
    return kinds


def _source_dirs(root: Path, raw: str) -> list[Path]:
    return [(root / part).resolve() for part in raw.split(os.pathsep) if part]


def install(root: Path, sources: list[Path]) -> Recorder:
    """Instrument every module loaded from ``sources`` by any loader path."""
    recorder = Recorder()
    setattr(builtins, RECORDER, recorder)
    original = importlib.machinery.SourceFileLoader.get_code

    def get_code(self: importlib.machinery.SourceFileLoader, fullname: str) -> Any:
        path = Path(self.get_filename(fullname)).resolve()
        if path.suffix != ".py" or not any(path.is_relative_to(src) for src in sources):
            return original(self, fullname)
        source = path.read_text(encoding="utf-8")
        tree, decisions = instrument(source, path.relative_to(root).as_posix())
        for decision in decisions:
            recorder.sizes[decision.id] = len(decision.conditions)
        return compile(tree, str(path), "exec", dont_inherit=True)

    importlib.machinery.SourceFileLoader.get_code = get_code  # type: ignore[method-assign]
    return recorder


def _activate() -> Recorder | None:
    if not (os.environ.get(DATA_ENV) and os.environ.get(SOURCE_ENV)):
        return None
    existing: Any = getattr(builtins, RECORDER, None)
    if existing is not None:
        # A second load of this file (e.g. by its own tests) shares the
        # recorder the pytest plugin installed; reinstalling would orphan it.
        return cast(Recorder, existing)
    root = Path(os.environ.get(ROOT_ENV) or Path.cwd()).resolve()
    return install(root, _source_dirs(root, os.environ[SOURCE_ENV]))


active_recorder = _activate()


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    del session, exitstatus
    if active_recorder is not None:
        out = Path(os.environ[DATA_ENV]) / f"structcov-{os.getpid()}.json"
        active_recorder.dump(out)


# --------------------------------------------------------------------------
# Analysis.


def evaluate(shape: Shape, values: dict[int, bool], vector: list[bool | None]) -> bool:
    """Short-circuit evaluation that marks only the conditions it reads."""
    if shape.kind == "const":
        return shape.value
    if shape.kind == "cond":
        vector[shape.index] = values[shape.index]
        return values[shape.index]
    if shape.kind == "not":
        return not evaluate(shape.children[0], values, vector)
    for child in shape.children:
        outcome = evaluate(child, values, vector)
        if outcome == (shape.kind == "or"):
            return outcome
    return shape.kind == "and"


Vector = tuple[tuple[bool | None, ...], bool]


def feasible_vectors(shape: Shape, size: int) -> set[Vector] | None:
    if size > MCC_MAX_CONDITIONS:
        return None
    found: set[Vector] = set()
    for bits in itertools.product((False, True), repeat=size):
        vector: list[bool | None] = [None] * size
        outcome = evaluate(shape, dict(enumerate(bits)), vector)
        found.add((tuple(vector), outcome))
    return found


def independence_pair(a: Vector, b: Vector, index: int) -> bool:
    if a[1] == b[1] or a[0][index] is None or b[0][index] is None:
        return False
    if a[0][index] == b[0][index]:
        return False
    return all(
        x == y or x is None or y is None
        for j, (x, y) in enumerate(zip(a[0], b[0], strict=True))
        if j != index
    )


def _pairs_for(vectors: set[Vector], index: int) -> list[tuple[Vector, Vector]]:
    ordered = sorted(vectors, key=repr)
    return [
        (a, b)
        for a, b in itertools.combinations(ordered, 2)
        if independence_pair(a, b, index)
    ]


@dataclass
class Tally:
    covered: int = 0
    total: int = 0

    def add(self, covered: int, total: int) -> None:
        self.covered += covered
        self.total += total

    def percent(self) -> float | None:
        return None if self.total == 0 else 100.0 * self.covered / self.total


@dataclass
class Analysis:
    tallies: dict[str, Tally] = field(default_factory=dict[str, Tally])
    files: dict[str, dict[str, Tally]] = field(
        default_factory=dict[str, dict[str, Tally]]
    )
    gaps: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    infeasible: list[str] = field(default_factory=list[str])

    def add(self, file: str, criterion: str, covered: int, total: int) -> None:
        self.tallies.setdefault(criterion, Tally()).add(covered, total)
        self.files.setdefault(file, {}).setdefault(criterion, Tally()).add(
            covered, total
        )


def _describe(vector: Vector, conditions: list[str]) -> str:
    parts = [
        f"{text}={value}"
        for text, value in zip(conditions, vector[0], strict=True)
        if value is not None
    ]
    return ", ".join(parts) + f" -> {vector[1]}"


def analyse_decision(
    decision: Decision,
    observed: set[Vector],
    boundary: dict[str, set[str]],
    analysis: Analysis,
) -> None:
    size = len(decision.conditions)
    file = decision.file
    outcomes = {outcome for _, outcome in observed}
    analysis.add(file, "decision", len(outcomes), 2)
    seen_values = {
        (index, value)
        for vector, _ in observed
        for index, value in enumerate(vector)
        if value is not None
    }
    analysis.add(file, "c2", len(seen_values), 2 * size)
    feasible = feasible_vectors(decision.shape, size)
    if feasible is not None:
        analysis.add(file, "mcc", len(observed & feasible), len(feasible))
    for index in range(size):
        possible = _pairs_for(feasible, index) if feasible is not None else None
        if possible is not None and not possible:
            analysis.infeasible.append(f"{decision.id} `{decision.conditions[index]}`")
            continue
        covered = bool(_pairs_for(observed, index))
        analysis.add(file, "mcdc", int(covered), 1)
        if not covered:
            analysis.gaps.append(_gap(decision, index, observed, possible or []))
    for index, op in sorted(decision.boundaries.items()):
        kinds = boundary.get(f"{decision.id}#{index}", set())
        exact = "int" in kinds and "float" not in kinds
        need = ("below1", "on", "above1") if exact else ("below", "on", "above")
        analysis.add(file, "boundary", len(set(need) & kinds), 3)
        missing = [k for k in need if k not in kinds]
        if missing:
            analysis.gaps.append(
                {
                    "criterion": "boundary",
                    "decision": decision.id,
                    "condition": decision.conditions[index],
                    "operator": op,
                    "missing": missing,
                }
            )


def _gap(
    decision: Decision,
    index: int,
    observed: set[Vector],
    possible: list[tuple[Vector, Vector]],
) -> dict[str, Any]:
    """Describe an MC/DC gap and the evaluation that would close it."""
    gap: dict[str, Any] = {
        "criterion": "mcdc",
        "decision": decision.id,
        "condition": decision.conditions[index],
    }
    for a, b in possible:
        for have, need in ((a, b), (b, a)):
            if have in observed:
                gap["have"] = _describe(have, decision.conditions)
                gap["add"] = _describe(need, decision.conditions)
                return gap
    if possible:
        a, b = possible[0]
        first = _describe(a, decision.conditions)
        second = _describe(b, decision.conditions)
        gap["add"] = f"{first} and {second}"
    return gap


def inventory(root: Path, sources: list[Path]) -> list[Decision]:
    decisions: list[Decision] = []
    for source in sources:
        files = [source] if source.is_file() else sorted(source.rglob("*.py"))
        for path in files:
            rel = path.resolve().relative_to(root).as_posix()
            _, found = instrument(path.read_text(encoding="utf-8"), rel)
            decisions.extend(found)
    return decisions


def load_observations(data: Path) -> tuple[dict[str, set[Vector]], dict[str, set[str]]]:
    vectors: dict[str, set[Vector]] = {}
    boundary: dict[str, set[str]] = {}
    for path in sorted(data.glob("structcov-*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for did, seen in raw.get("vectors", {}).items():
            bucket = vectors.setdefault(did, set())
            for vec, out in seen:
                bucket.add((tuple(vec), bool(out)))
        for key, kinds in raw.get("boundary", {}).items():
            boundary.setdefault(key, set()).update(kinds)
    return vectors, boundary


def _pad(vectors: set[Vector], size: int) -> set[Vector]:
    return {
        (tuple(list(vec) + [None] * (size - len(vec)))[:size], out)
        for vec, out in vectors
    }


def analyse(
    root: Path, sources: list[Path], data: Path, coverage_json: Path | None
) -> Analysis:
    analysis = Analysis()
    vectors, boundary = load_observations(data)
    for decision in inventory(root, sources):
        observed = _pad(vectors.get(decision.id, set()), len(decision.conditions))
        analyse_decision(decision, observed, boundary, analysis)
    if coverage_json is not None and coverage_json.is_file():
        raw = json.loads(coverage_json.read_text(encoding="utf-8"))
        for name, info in raw.get("files", {}).items():
            summary = info["summary"]
            rel = (
                Path(name).resolve().relative_to(root).as_posix()
                if Path(name).is_absolute()
                else name
            )
            analysis.add(rel, "c0", summary["covered_lines"], summary["num_statements"])
            analysis.add(
                rel,
                "c1",
                summary.get("covered_branches", 0),
                summary.get("num_branches", 0),
            )
    return analysis


# --------------------------------------------------------------------------
# Configuration, gate and reports.


@dataclass(frozen=True)
class Config:
    root: Path
    sources: list[Path]
    floors: dict[str, float]


def load_config(root: Path) -> Config:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    tool = data.get("tool", {})
    raw_sources = tool.get("coverage", {}).get("run", {}).get("source", [])
    sources = [(root / str(src)).resolve() for src in raw_sources]
    if not sources:
        raise SystemExit("structural_coverage: [tool.coverage.run] source is empty")
    raw_floors = tool.get("vibebb-coverage", {})
    floors = {key: float(raw_floors[key]) for key in CRITERIA if key in raw_floors}
    return Config(root, sources, floors)


def gate(analysis: Analysis, floors: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for criterion, floor in floors.items():
        percent = analysis.tallies.get(criterion, Tally()).percent()
        if percent is None:
            failures.append(
                f"{criterion}: unknown (nothing measured), floor {floor:.1f}%"
            )
        elif percent + 1e-9 < floor:
            failures.append(f"{criterion}: {percent:.2f}% below floor {floor:.1f}%")
    return failures


LABELS = {
    "c0": "C0 statement",
    "c1": "C1 branch",
    "decision": "decision (incl. ternary/filter/assert/guard)",
    "c2": "C2 condition",
    "mcc": "MCC multiple condition (reported only)",
    "mcdc": "MC/DC (unique-cause, short-circuit masking)",
    "boundary": "3-value boundary",
}


def to_json(
    analysis: Analysis, floors: dict[str, float], failures: list[str]
) -> dict[str, Any]:
    def tally(t: Tally) -> dict[str, Any]:
        return {"covered": t.covered, "total": t.total, "percent": t.percent()}

    return {
        "schema": "vibebb-structural-coverage/1",
        "totals": {k: tally(v) for k, v in sorted(analysis.tallies.items())},
        "floors": floors,
        "verdict": "fail" if failures else "pass",
        "failures": failures,
        "files": {
            f: {k: tally(v) for k, v in sorted(c.items())}
            for f, c in sorted(analysis.files.items())
        },
        "gaps": analysis.gaps,
        "infeasible_mcdc": analysis.infeasible,
    }


def summary_text(
    analysis: Analysis, floors: dict[str, float], failures: list[str]
) -> str:
    lines = ["structural coverage:"]
    for key in ("c0", "c1", "decision", "c2", "mcc", "mcdc", "boundary"):
        t = analysis.tallies.get(key, Tally())
        pct = t.percent()
        shown = "n/a" if pct is None else f"{pct:6.2f}%"
        floor = f"  floor {floors[key]:.1f}%" if key in floors else ""
        lines.append(f"  {LABELS[key]:<46} {t.covered:>6}/{t.total:<6} {shown}{floor}")
    mcdc_gaps = sum(g["criterion"] == "mcdc" for g in analysis.gaps)
    boundary_gaps = sum(g["criterion"] == "boundary" for g in analysis.gaps)
    lines.append(
        f"  gaps: {mcdc_gaps} MC/DC, {boundary_gaps} boundary; "
        f"{len(analysis.infeasible)} infeasible MC/DC conditions excluded"
    )
    lines.extend(f"  FAIL {failure}" for failure in failures)
    return "\n".join(lines)


def report(
    config: Config, data: Path, coverage_json: Path | None, out: Path | None
) -> int:
    analysis = analyse(config.root, config.sources, data, coverage_json)
    failures = gate(analysis, config.floors)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                to_json(analysis, config.floors, failures), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
    text = summary_text(analysis, config.floors, failures)
    print(text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a", encoding="utf-8") as handle:
            handle.write(f"```text\n{text}\n```\n")
    return 1 if failures else 0


def run(config: Config, pytest_args: list[str], out: Path | None) -> int:
    with tempfile.TemporaryDirectory(prefix="structcov-") as tmp:
        data = Path(tmp)
        coverage_json = data / "coverage.json"
        env = dict(os.environ)
        here = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (here, env.get("PYTHONPATH")) if p
        )
        env[DATA_ENV] = tmp
        env[ROOT_ENV] = str(config.root)
        env[SOURCE_ENV] = os.pathsep.join(str(src) for src in config.sources)
        argv = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "structural_coverage",
            "--cov",
            "--cov-branch",
            "--cov-report=term-missing:skip-covered",
            f"--cov-report=json:{coverage_json}",
            *pytest_args,
        ]
        code = subprocess.run(argv, cwd=config.root, env=env, check=False).returncode
        if code != 0:
            return code
        return report(config, data, coverage_json, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Structural coverage gate.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser(
        "run", help="run pytest with all structural criteria and gate"
    )
    run_p.add_argument("--json", type=Path, default=None)
    run_p.add_argument("pytest_args", nargs=argparse.REMAINDER)
    rep_p = sub.add_parser("report", help="analyse collected observations and gate")
    rep_p.add_argument("--data", type=Path, required=True)
    rep_p.add_argument("--coverage-json", type=Path, default=None)
    rep_p.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(Path.cwd().resolve())
    if args.command == "run":
        extra = [a for a in args.pytest_args if a != "--"]
        return run(config, extra, args.json)
    return report(config, args.data, args.coverage_json, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
