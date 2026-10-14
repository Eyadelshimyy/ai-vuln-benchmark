"""Tests for scripts/mutation_probe.py (advisory mutation testing)."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mutation_probe.py"


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mutation_probe_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = textwrap.dedent(
    """
    def within(value, limit):
        return value <= limit and value >= 0

    def flag():
        return True

    def tagged(value):
        return value == 1  # pragma: no mutate
    """
)


def test_sites_cover_each_operator_class(probe: ModuleType) -> None:
    found = probe.sites(SOURCE, "m.py")
    assert [(s.operator, s.before, s.after) for s in found] == [
        ("LCR", "and", "or"),
        ("ROR", "<=", "<"),
        ("ROR", ">=", ">"),
        ("BCR", "return True", "return False"),
    ]


def test_mutate_changes_exactly_one_node(probe: ModuleType) -> None:
    for site in probe.sites(SOURCE, "m.py"):
        mutated = ast.unparse(probe.mutate(SOURCE, "m.py", site.index))
        original = ast.unparse(ast.parse(SOURCE))
        assert mutated != original
        assert len(probe.sites(mutated, "m.py")) == len(probe.sites(original, "m.py"))


def test_mutate_rejects_unknown_site(probe: ModuleType) -> None:
    with pytest.raises(ValueError, match="no mutation site"):
        probe.mutate(SOURCE, "m.py", 99)


def test_sample_is_deterministic_and_bounded(probe: ModuleType) -> None:
    items = list(range(10))
    assert probe.sample(items, 0) == items
    assert probe.sample(items, 20) == items
    assert probe.sample(items, 4) == [0, 2, 5, 7]
    assert probe.sample(items, 4) == probe.sample(items, 4)


def test_score_counts_timeouts_as_killed(probe: ModuleType) -> None:
    assert probe.score([]) is None
    assert probe.score(["error"]) is None
    assert probe.score(["killed", "timeout", "survived", "survived"]) == 50.0


def _project(tmp_path: Path, tests: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.vibebb-mutation]\ntargets = ["gate.py"]\ntests = ["test_gate.py"]\n',
        encoding="utf-8",
    )
    (tmp_path / "gate.py").write_text(
        "def ok(value, limit):\n    return value <= limit\n", encoding="utf-8"
    )
    (tmp_path / "test_gate.py").write_text(
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        "from gate import ok\n" + tests,
        encoding="utf-8",
    )
    return tmp_path


def test_on_boundary_assertion_kills_the_ror_mutant(
    probe: ModuleType, tmp_path: Path
) -> None:
    root = _project(tmp_path, "def test_on():\n    assert ok(5, 5)\n")
    out = tmp_path / "out.json"
    assert (
        probe.main(["run", "--root", str(root), "--json", str(out), "--jobs", "1"]) == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["advisory"] is True
    assert report["counts"]["killed"] == 1
    assert report["survivors"] == []
    assert report["score_percent"] == 100.0


def test_missing_boundary_assertion_reports_a_survivor(
    probe: ModuleType, tmp_path: Path
) -> None:
    root = _project(tmp_path, "def test_below():\n    assert ok(4, 5)\n")
    out = tmp_path / "out.json"
    assert (
        probe.main(["run", "--root", str(root), "--json", str(out), "--jobs", "1"]) == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [(s["line"], s["before"], s["after"]) for s in report["survivors"]] == [
        (2, "<=", "<")
    ]
    assert report["score_percent"] == 0.0


def test_failing_baseline_is_unmeasurable(probe: ModuleType, tmp_path: Path) -> None:
    root = _project(tmp_path, "def test_bad():\n    assert not ok(4, 5)\n")
    assert probe.main(["run", "--root", str(root), "--jobs", "1"]) == 1


def test_missing_config_is_rejected(probe: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.other]\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="vibebb-mutation"):
        probe.main(["run", "--root", str(tmp_path)])
