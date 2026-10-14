"""Tests for the family-canonical structural coverage gate.

The file is identical in every VibeBB repository; the script digest below
pins ``scripts/structural_coverage.py`` byte for byte.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "structural_coverage.py"
EXPECTED_SHA256 = "cfc9ae0569da4076a91a58a4a6c7d226ec94065c024ac4a34ee0e314a8499dc7"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vibebb_structural_coverage", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load()


def _exec(code: CodeType, namespace: dict[str, Any]) -> None:
    FunctionType(code, namespace)()


def _probed(expr: str) -> str:
    def probe(match: re.Match[str]) -> str:
        return f"p(log, {'abc'.index(match.group(0))}, {match.group(0)})"

    return re.sub(r"\b[abc]\b", probe, expr)


def _shape(source: str) -> tuple[Any, int]:
    _, decisions = sc.instrument(f"if {source}:\n    pass\n", "m.py")
    assert len(decisions) == 1
    return decisions[0].shape, len(decisions[0].conditions)


def _run(
    source: str, calls: list[tuple[Any, ...]], func: str = "f"
) -> tuple[Any, list[Any]]:
    """Execute instrumented ``source`` with a private recorder; return results."""
    recorder = sc.Recorder()
    tree, decisions = sc.instrument(source, "m.py")
    for decision in decisions:
        recorder.sizes[decision.id] = len(decision.conditions)
    namespace: dict[str, Any] = {sc.RECORDER: recorder}
    _exec(compile(tree, "m.py", "exec"), namespace)
    results = [namespace[func](*args) for args in calls]
    analysis = sc.Analysis()
    observations = {k: set(v) for k, v in recorder.vectors.items()}
    for decision in decisions:
        observed = sc._pad(
            observations.get(decision.id, set()), len(decision.conditions)
        )
        sc.analyse_decision(decision, observed, recorder.boundary, analysis)
    return analysis, results


def _pct(analysis: Any, criterion: str) -> float | None:
    return analysis.tallies.get(criterion, sc.Tally()).percent()


def test_script_matches_family_canon() -> None:
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_floors_are_configured_for_every_gated_criterion() -> None:
    config = sc.load_config(ROOT)
    assert set(config.floors) == {"c0", "c1", "decision", "c2", "mcdc", "boundary"}
    assert all(0.0 <= floor <= 100.0 for floor in config.floors.values())
    assert config.sources and all(source.exists() for source in config.sources)


# ---------------------------------------------------------------- semantics


SEMANTIC_CASES = [
    "a and b",
    "a or b",
    "not a and (b or c)",
    "(a or b) and not (c and a)",
    "a and (b or (c and not a))",
]


@pytest.mark.parametrize("expr", SEMANTIC_CASES)
def test_instrumentation_preserves_values_and_short_circuit(expr: str) -> None:
    source = (
        "def f(a, b, c, log):\n"
        f"    v = ({_probed(expr)})\n"
        "    return v\n"
        "def p(log, i, x):\n"
        "    log.append(i)\n"
        "    return x\n"
    )
    original: dict[str, Any] = {}
    _exec(compile(source, "o.py", "exec"), original)
    values = (0, 1, "", "s", None, [0])
    for a, b, c in itertools.product(values, repeat=3):
        log_o: list[int] = []
        log_i: list[int] = []
        expected = original["f"](a, b, c, log_o)
        _, (actual,) = _run(source, [(a, b, c, log_i)])
        assert actual == expected and type(actual) is type(expected)
        assert log_i == log_o


def test_ternary_filter_assert_guard_and_while_are_decisions() -> None:
    source = (
        "def f(xs):\n"
        "    out = [x for x in xs if x > 1]\n"
        "    assert len(out) >= 0\n"
        "    n = 0\n"
        "    while n < 2:\n"
        "        n += 1\n"
        "    match n:\n"
        "        case 2 if out:\n"
        "            return 'a' if out[0] else 'b'\n"
        "        case _:\n"
        "            return None\n"
    )
    _, decisions = sc.instrument(source, "m.py")
    assert sorted(d.context for d in decisions) == [
        "assert",
        "filter",
        "guard",
        "ternary",
        "while",
    ]
    _, results = _run(source, [([2, 0],), ([0],)])
    assert results == ["a", None]


def test_constant_type_checking_main_and_pragma_are_not_decisions() -> None:
    source = (
        "TYPE_CHECKING = False\n"
        "if TYPE_CHECKING:\n    pass\n"
        "while True:\n    break\n"
        "if __name__ == '__main__':\n    pass\n"
        "def f(a, b):  # pragma: no cover\n"
        "    return a and b\n"
        "x = y = 1\n"
        "z = x or 'default'\n"
    )
    _, decisions = sc.instrument(source, "m.py")
    assert decisions == []


def test_value_boolop_with_two_conditions_is_a_decision() -> None:
    _, decisions = sc.instrument("def f(a, b):\n    return a or b\n", "m.py")
    assert [d.context for d in decisions] == ["value"]


def test_nested_decision_inside_a_condition_is_separate() -> None:
    source = (
        "def f(a, b, c):\n    if g(a and b) or c:\n        return 1\n    return 0\n"
    )
    source += "def g(x):\n    return x\n"
    _, decisions = sc.instrument(source, "m.py")
    assert len(decisions) == 2
    _, results = _run(source, [(1, 1, 0), (0, 1, 0), (0, 0, 1)])
    assert results == [1, 0, 1]


def test_recursion_and_exceptions_keep_vectors_separate() -> None:
    source = (
        "def f(n, boom):\n"
        "    if n > 0 and f(n - 1, boom) >= 0:\n"
        "        return n\n"
        "    if boom and explode():\n"
        "        return -2\n"
        "    return 0\n"
        "def explode():\n"
        "    raise ValueError('x')\n"
        "def g(n):\n"
        "    try:\n"
        "        return f(n, True)\n"
        "    except ValueError:\n"
        "        return f(n, False)\n"
    )
    analysis, results = _run(source, [(3,), (0,)], func="g")
    assert results == [3, 0]
    assert _pct(analysis, "decision") is not None


# ---------------------------------------------------------------- criteria


def test_decision_table_admit_needs_three_vectors_for_mcdc() -> None:
    source = "def f(member, ticket):\n    return 'in' if member and ticket else 'out'\n"
    two, _ = _run(source, [(True, True), (False, False)])
    assert _pct(two, "decision") == 100.0
    assert _pct(two, "mcdc") == 50.0
    gap = next(g for g in two.gaps if g["criterion"] == "mcdc")
    assert gap["condition"] == "ticket"
    assert gap["add"] == "member=True, ticket=False -> False"
    three, _ = _run(source, [(True, True), (False, False), (True, False)])
    assert _pct(three, "mcdc") == 100.0
    assert _pct(three, "c2") == 100.0
    assert _pct(three, "mcc") == 100.0


def test_condition_coverage_does_not_imply_decision_coverage() -> None:
    source = "def f(a, b):\n    return 1 if a and not b else 0\n"
    analysis, _ = _run(source, [(True, True), (False, False)])
    assert _pct(analysis, "decision") == 50.0
    assert _pct(analysis, "c2") == 75.0


def test_mcc_counts_feasible_short_circuit_vectors_only() -> None:
    shape, size = _shape("a and b and c")
    feasible = sc.feasible_vectors(shape, size)
    assert feasible is not None and len(feasible) == 4
    shape, size = _shape("(a or b) and (c or d)")
    feasible = sc.feasible_vectors(shape, size)
    assert feasible is not None and len(feasible) == 7


def test_mcdc_minimum_is_n_plus_one_for_and_chain() -> None:
    source = "def f(a, b, c):\n    return 1 if a and b and c else 0\n"
    analysis, _ = _run(source, [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    assert _pct(analysis, "mcdc") == 100.0


def test_masked_condition_is_infeasible_and_excluded() -> None:
    source = "def f(a):\n    return 1 if a or True else 0\n"
    analysis, _ = _run(source, [(0,), (1,)])
    assert analysis.infeasible and "`a`" in analysis.infeasible[0]
    assert analysis.tallies.get("mcdc") is None


def test_independence_pair_treats_short_circuit_as_dont_care() -> None:
    a: Any = ((True, True, None), True)
    b: Any = ((False, None, None), False)
    assert sc.independence_pair(a, b, 0)
    c: Any = ((True, False, False), False)
    assert not sc.independence_pair(a, c, 0)
    assert not sc.independence_pair(a, a, 0)


def test_too_many_conditions_skips_mcc_but_measures_mcdc() -> None:
    names = [f"x{i}" for i in range(sc.MCC_MAX_CONDITIONS + 1)]
    source = (
        f"def f({', '.join(names)}):\n    return 1 if {' and '.join(names)} else 0\n"
    )
    ones = (1,) * len(names)
    analysis, _ = _run(source, [ones, (0, *ones[1:])])
    assert analysis.tallies.get("mcc") is None
    assert analysis.tallies["mcdc"].covered == 1


# ---------------------------------------------------------------- boundaries


@pytest.mark.parametrize(
    ("calls", "missing"),
    [
        ([(9,), (10,), (11,)], []),
        ([(10,), (11,)], ["below1"]),
        ([(8,), (10,), (12,)], ["below1", "above1"]),
        ([(9,), (11,)], ["on"]),
        ([], ["below", "on", "above"]),
    ],
)
def test_three_value_boundary_on_integers(
    calls: list[tuple[int]], missing: list[str]
) -> None:
    source = "def f(n):\n    return 'big' if n >= 10 else 'small'\n"
    analysis, _ = _run(source, calls)
    gaps = [g for g in analysis.gaps if g["criterion"] == "boundary"]
    if missing:
        assert gaps[0]["missing"] == missing
    else:
        assert not gaps


def test_boundary_on_floats_needs_both_sides_and_exact_value() -> None:
    source = "def f(v):\n    return 1 if v <= 3.3 else 0\n"
    partial, _ = _run(source, [(3.2,), (3.4,)])
    assert _pct(partial, "boundary") == pytest.approx(200 / 3)
    full, _ = _run(source, [(3.2,), (3.3,), (3.4,)])
    assert _pct(full, "boundary") == 100.0


def test_boundary_ignores_bools_strings_and_non_finite() -> None:
    assert sc._boundary_class(True, 1) is None
    assert sc._boundary_class("a", "b") is None
    assert sc._boundary_class(float("inf"), 1.0) is None
    assert sc._boundary_class(1, 2) == {"below", "below1", "int"}


def test_chained_comparison_is_one_condition_without_boundary() -> None:
    _, decisions = sc.instrument("if 0 < x < 5:\n    pass\n", "m.py")
    assert decisions[0].conditions == ["0 < x < 5"]
    assert decisions[0].boundaries == {}


# ---------------------------------------------------------------- gate


def test_gate_fails_closed_on_unknown_and_below_floor() -> None:
    analysis = sc.Analysis()
    analysis.add("m.py", "c0", 9, 10)
    failures = sc.gate(analysis, {"c0": 90.0, "mcdc": 10.0})
    assert failures == ["mcdc: unknown (nothing measured), floor 10.0%"]
    assert sc.gate(analysis, {"c0": 90.1}) == ["c0: 90.00% below floor 90.1%"]


def test_end_to_end_run_writes_report_and_gates(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "demo"
    pkg.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "structural_coverage.py").write_bytes(SCRIPT.read_bytes())
    (pkg / "__init__.py").write_text(
        "def admit(m, t):\n    return 'in' if m and t else 'out'\n", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_demo.py").write_text(
        "from demo import admit\n"
        "def test_admit():\n"
        "    assert admit(True, True) == 'in'\n"
        "    assert admit(False, False) == 'out'\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.coverage.run]\nsource = ["src/demo"]\n'
        "[tool.vibebb-coverage]\nc0 = 100\nmcdc = 60\n",
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("VIBEBB_STRUCTCOV")}
    env.pop("GITHUB_STEP_SUMMARY", None)
    env["PYTHONPATH"] = str(tmp_path / "src")
    out = tmp_path / "report.json"
    argv = [
        sys.executable,
        "scripts/structural_coverage.py",
        "run",
        "--json",
        str(out),
        "--",
    ]
    argv += [
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:xdist",
        "-p",
        "no:randomly",
        "tests",
    ]
    proc = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "fail"
    assert report["failures"] == ["mcdc: 50.00% below floor 60.0%"]
    assert report["totals"]["c0"]["percent"] == 100.0
    assert "pyc" not in {p.suffix for p in pkg.rglob("*")}


def test_instrumented_tree_compiles_for_every_source_file() -> None:
    config = sc.load_config(ROOT)
    for source in config.sources:
        for path in sorted(source.rglob("*.py")) if source.is_dir() else [source]:
            tree, _ = sc.instrument(path.read_text(encoding="utf-8"), path.name)
            compile(tree, str(path), "exec")
            assert isinstance(tree, ast.Module)
