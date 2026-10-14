"""Phase 13 repository semantic graph.

Unique local symbols can cross files. Ambiguous, dynamic, and third-party
names stay unresolved. A zero limit never creates one accidental edge.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.domain.findings import FindingStatus
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.security.cross_file import CrossFileLimits, build_project
from app.security.engine import SecurityAnalysisEngine

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phase13_repo"


def _scan(root: Path):
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return SecurityAnalysisEngine().analyze_repository(root, paths)


def _lines(result, filename: str) -> list[int]:
    return sorted(
        obs.line
        for obs in result.observations
        if obs.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION
        and str(obs.file_path).endswith(filename)
    )


def _parse_tree(files: dict[str, str]) -> dict:
    graphs = {}
    for name, text in files.items():
        language = "python"
        if name.endswith((".js", ".mjs", ".jsx")):
            language = "javascript"
        elif name.endswith((".ts", ".tsx")):
            language = "typescript"
        graph = parse_source(language, Path(name), text)
        graph.file_path = name
        graphs[name] = graph
    return graphs


def _budgeted(project) -> list:
    return [edge for edge in project.edges if edge.kind != "method_owner"]


def test_zero_and_negative_limits_create_no_edges() -> None:
    graphs = _parse_tree(
        {
            "helpers.py": "def run_code(value):\n    eval(value)\n",
            "app.py": "from helpers import run_code\nrun_code(request.args.get('q'))\n",
        }
    )
    for limits in (
        CrossFileLimits(max_files=0),
        CrossFileLimits(max_files=-1),
        CrossFileLimits(max_edges=0),
        CrossFileLimits(max_edges=-4),
        CrossFileLimits(max_rounds=0),
        CrossFileLimits(max_rounds=-2),
        CrossFileLimits(max_import_depth=0),
        CrossFileLimits(max_import_depth=-1),
        CrossFileLimits(budget_ms=0),
        CrossFileLimits(budget_ms=-5),
    ):
        project = build_project(graphs, limits=limits)
        assert project.incomplete
        assert project.externals == {}
        assert project.edges == []
        assert project.diagnostics


def test_settings_reject_negative_taint_limits() -> None:
    with pytest.raises(ValidationError):
        Settings(taint_max_files=-1)
    with pytest.raises(ValidationError):
        Settings(taint_max_cross_file_edges=-1)


def test_max_files_boundaries() -> None:
    graphs = _parse_tree(
        {
            "a_helper.py": "def run_code(value):\n    eval(value)\n",
            "b_app.py": "from a_helper import run_code\nrun_code(request.args.get('q'))\n",
            "c_other.py": "x = 1\n",
        }
    )
    disabled = build_project(graphs, limits=CrossFileLimits(max_files=0))
    assert disabled.edges == []

    one = build_project(graphs, limits=CrossFileLimits(max_files=1))
    assert one.externals == {}
    assert any("stopped after 1 files" in item.message for item in one.diagnostics)

    exact = build_project(graphs, limits=CrossFileLimits(max_files=len(graphs)))
    assert "b_app.py" in exact.externals
    assert not any("stopped after" in item.message for item in exact.diagnostics)

    above = build_project(graphs, limits=CrossFileLimits(max_files=len(graphs) + 5))
    assert "b_app.py" in above.externals


def test_max_edges_never_exceeds_the_limit() -> None:
    graphs = _parse_tree(
        {
            "helpers.py": (
                "def run_a(value):\n    eval(value)\ndef run_b(value):\n    eval(value)\n"
            ),
            "app.py": (
                "from helpers import run_a, run_b\n"
                "run_a(request.args.get('q'))\n"
                "run_b(request.args.get('q'))\n"
            ),
        }
    )
    full = build_project(graphs, limits=CrossFileLimits())
    available = len(_budgeted(full))
    assert available >= 4
    assert len(full.externals["app.py"]) == 2

    for limit in (1, available - 1, available, available + 1):
        project = build_project(graphs, limits=CrossFileLimits(max_edges=limit))
        budgeted = _budgeted(project)
        assert len(budgeted) == min(limit, available)
        assert len(budgeted) <= limit
        if limit < available:
            assert project.incomplete
            assert any("edge limit" in item.message for item in project.diagnostics)
        else:
            assert not any("edge limit" in item.message for item in project.diagnostics)


def test_symbol_ids_are_stable_and_address_free() -> None:
    files = {
        "pkg/helpers.py": "class Executor:\n    def run(self, value):\n        eval(value)\n",
        "pkg/app.py": "from helpers import Executor\nExecutor().run(1)\n",
    }
    first = build_project(_parse_tree(files))
    second = build_project(_parse_tree(files))
    assert [(e.kind, e.source_id, e.target_id) for e in first.edges] == [
        (e.kind, e.source_id, e.target_id) for e in second.edges
    ]
    method_ids = [edge.source_id for edge in first.edges if edge.kind == "method_owner"]
    assert method_ids
    for edge in first.edges:
        assert edge.source_id.count("::") == 3
        assert "0x" not in edge.source_id
        assert "0x" not in edge.target_id
    assert any(edge.source_id.endswith("::Executor::run") for edge in first.edges)
    assert any(edge.target_id.endswith("::Executor::") for edge in first.edges)


def test_python_package_reexport_changes_the_observation() -> None:
    result = _scan(FIXTURE / "pkg_flow")
    assert _lines(result, "app.py") == [3, 4]
    metas = [
        obs.metadata
        for obs in result.observations
        if str(obs.file_path).endswith("app.py")
        and obs.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION
    ]
    assert {item["relationship"] for item in metas} == {"re_export"}
    assert {item["re_export"] for item in metas} == {"true"}
    assert any(item["defining_file"].endswith("pkg/helpers.py") for item in metas)
    assert any(item["defining_file"].endswith("pkg/internal/tools.py") for item in metas)
    assert all(item["class_method"] == "false" for item in metas)
    assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)


def test_ambiguous_package_does_not_flow() -> None:
    result = _scan(FIXTURE / "pkg_ambiguous")
    assert _lines(result, "app.py") == []
    assert any(item.kind == "ambiguous_import" for item in result.diagnostics)


@pytest.mark.parametrize(
    ("folder", "filename", "expected"),
    [
        ("js_default", "app.js", [2]),
        ("js_default_ident", "app.js", [2]),
        ("js_default_anon", "app.js", []),
        ("js_default_class", "app.js", [2]),
        ("js_named_alias", "app.js", [2]),
        ("js_namespace", "app.js", [2]),
        ("js_barrel", "app.js", [2]),
        ("js_export_from", "app.js", [2]),
        ("js_export_star", "app.js", []),
        ("js_index", "app.js", [2]),
        ("js_index_tsx", "app.ts", [2]),
        ("js_ambiguous", "app.js", []),
        ("ts_alias", "app.ts", [2]),
        ("ts_alias_ambiguous", "app.ts", []),
        ("ts_alias_unresolved", "app.ts", []),
        ("ts_alias_comment", "app.ts", []),
        ("ts_barrel", "app.ts", [2]),
        ("callback", "app.py", [4]),
        ("callback_local", "app.py", [6]),
        ("callback_reassign", "app.py", []),
        ("async_py", "app.py", [5, 6]),
        ("async_unknown", "app.py", []),
        ("async_reassign", "app.py", []),
        ("async_js", "app.js", [2, 3]),
        ("async_js_unknown", "app.js", []),
        ("third_party", "app.py", []),
        ("unresolved", "app.js", []),
        ("dynamic", "app.py", []),
        ("dynamic_js", "app.js", []),
        ("inheritance", "app.py", []),
    ],
)
def test_phase13_corpus_cases(folder: str, filename: str, expected: list[int]) -> None:
    result = _scan(FIXTURE / folder)
    assert _lines(result, filename) == expected
    assert all(item.status is not FindingStatus.VERIFIED for item in result.findings)


def test_class_methods_stay_distinct() -> None:
    result = _scan(FIXTURE / "classes")
    assert _lines(result, "app.py") == [4, 5, 6, 7, 8]
    metas = [
        obs.metadata
        for obs in result.observations
        if str(obs.file_path).endswith("app.py")
        and obs.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION
    ]
    assert {item["defining_symbol"] for item in metas} == {
        "Executor.run",
        "Executor.stat",
        "Executor.build",
    }
    assert all(item["class_method"] == "true" for item in metas)
    assert all("::Reporter::" not in item["semantic_id"] for item in metas)
    assert all(item["semantic_id"].count("::") == 3 for item in metas)


def test_default_and_star_diagnostics() -> None:
    anon = _scan(FIXTURE / "js_default_anon")
    assert any(item.kind == "unresolved_export" for item in anon.diagnostics)
    star = _scan(FIXTURE / "js_export_star")
    assert any(item.kind == "ambiguous_import" for item in star.diagnostics)
    ambiguous = _scan(FIXTURE / "js_ambiguous")
    assert any(item.kind == "ambiguous_import" for item in ambiguous.diagnostics)
    paths = _scan(FIXTURE / "ts_alias_ambiguous")
    assert any(item.kind == "ambiguous_import" for item in paths.diagnostics)


def test_partial_callee_still_has_no_summary() -> None:
    graphs = _parse_tree(
        {
            "helpers.py": "def run_code(value):\n    eval(value)\n",
            "app.py": "from helpers import run_code\nrun_code(request.args.get('q'))\n",
        }
    )
    graphs["helpers.py"].diagnostics = replace(
        graphs["helpers.py"].diagnostics, has_errors=True, error_count=1
    )
    project = build_project(graphs)
    assert project.externals == {}
    assert any(item.kind == "cross_file_partial" for item in project.diagnostics)

    graphs["helpers.py"].diagnostics = replace(
        graphs["helpers.py"].diagnostics, has_errors=False, error_count=0, truncated=True
    )
    truncated = build_project(graphs)
    assert truncated.externals == {}
    assert any(item.kind == "cross_file_partial" for item in truncated.diagnostics)


def test_phase13_corpus_is_deterministic_and_bounded() -> None:
    started = time.monotonic()
    engine = SecurityAnalysisEngine()
    paths = sorted(path for path in FIXTURE.rglob("*") if path.is_file())

    def signature():
        result = engine.analyze_repository(FIXTURE, paths)
        obs = tuple(
            sorted(
                (
                    item.file_path,
                    item.line,
                    item.rule_id,
                    item.vulnerability_class.value,
                    item.metadata.get("semantic_id", ""),
                    item.metadata.get("relationship", ""),
                    item.metadata.get("taint_scope", ""),
                )
                for item in result.observations
            )
        )
        diags = tuple((item.kind, item.file_path, item.message) for item in result.diagnostics)
        return obs, diags, result

    first_obs, first_diags, first = signature()
    second_obs, second_diags, _second = signature()
    assert first_obs == second_obs
    assert first_diags == second_diags
    assert time.monotonic() - started < 15
    assert all(item.status is not FindingStatus.VERIFIED for item in first.findings)
    assert any(item[0].endswith("pkg_flow/app.py") for item in first_obs)
    assert not any(item[0].endswith("pkg_ambiguous/app.py") for item in first_obs)
    assert any(kind == "ambiguous_import" for kind, _path, _message in first_diags)
    assert any(
        kind in {"parser_error", "parser_failure", "partial_analysis"} for kind, _, _ in first_diags
    )


def test_cross_file_metadata_names_the_caller(tmp_path: Path) -> None:
    root = tmp_path / "meta"
    root.mkdir()
    (root / "helpers.py").write_text("def run_code(value):\n    eval(value)\n")
    (root / "app.py").write_text(
        "from helpers import run_code\ndef handle():\n    run_code(request.args.get('q'))\n"
    )
    result = _scan(root)
    obs = next(
        item
        for item in result.observations
        if item.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION
    )
    assert obs.metadata["caller_file"].endswith("app.py")
    assert obs.metadata["caller_symbol"] == "handle"
    assert obs.metadata["defining_symbol"] == "run_code"
    assert obs.metadata["parameter_index"] == "0"
    assert obs.metadata["parser_complete"] == "true"
    assert obs.metadata["relationship"] == "import"
    assert "0x" not in obs.metadata["semantic_id"]


def test_file_limit_keeps_local_finding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "taint_max_files", 0)
    root = tmp_path / "local"
    root.mkdir()
    (root / "helpers.py").write_text("def run_code(value):\n    eval(value)\n")
    (root / "app.py").write_text(
        "from helpers import run_code\nrun_code(request.args.get('q'))\nq = request.args.get('q')\neval(q)\n"
    )
    result = _scan(root)
    assert _lines(result, "app.py") == [4]
    assert any(item.kind == "cross_file_incomplete" for item in result.diagnostics)


def test_conditional_alias_stays_unresolved() -> None:
    graphs = _parse_tree(
        {
            "helpers.py": "def run_code(value):\n    eval(value)\n",
            "app.py": (
                "from helpers import run_code\n"
                "if cond:\n"
                "    alias = run_code\n"
                "alias(request.args.get('q'))\n"
            ),
        }
    )
    project = build_project(graphs)
    assert "alias" not in project.externals.get("app.py", {})
    assert "run_code" in project.externals["app.py"]


def test_reassigned_default_export_stays_unresolved() -> None:
    graphs = _parse_tree(
        {
            "helpers.js": (
                "function runCode(value) { eval(value); }\n"
                "export default runCode;\n"
                "runCode = other;\n"
            ),
            "app.js": 'import runCode from "./helpers.js";\nrunCode(req.query.q);\n',
        }
    )
    project = build_project(graphs)
    assert project.externals == {}
    assert any(item.kind == "unresolved_export" for item in project.diagnostics)


def test_import_depth_records_an_unfollowed_hop() -> None:
    graphs = _parse_tree(
        {
            "d0.py": "def run_code(value):\n    eval(value)\n",
            "d1.py": "from d0 import run_code\n",
            "d2.py": "from d1 import run_code\n",
            "d3.py": "from d2 import run_code\n",
            "app.py": "from d3 import run_code\nrun_code(request.args.get('q'))\n",
        }
    )
    short = build_project(graphs, limits=CrossFileLimits(max_import_depth=1))
    assert "app.py" not in short.externals
    assert short.incomplete
    assert any(item.kind == "cross_file_depth_limited" for item in short.diagnostics)

    reached = build_project(graphs, limits=CrossFileLimits(max_import_depth=2))
    assert "run_code" in reached.externals.get("app.py", {})


def test_partial_importer_is_explicit() -> None:
    graphs = _parse_tree(
        {
            "helpers.py": "def run_code(value):\n    eval(value)\n",
            "app.py": "from helpers import run_code\nrun_code(request.args.get('q'))\n",
        }
    )
    graphs["app.py"].diagnostics = replace(
        graphs["app.py"].diagnostics, has_errors=True, error_count=1
    )
    project = build_project(graphs)
    ext = project.externals["app.py"]["run_code"]
    assert ext.partial
    assert project.incomplete
    assert any("partial importer" in item.message for item in project.diagnostics)
