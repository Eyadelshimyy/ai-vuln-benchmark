"""Phase 12 semantic, cross-file, and repository-corpus checks.

Static results stay potential. Partial graphs, ambiguous imports, and
unknown calls do not become high-confidence flows.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from app.core.config import settings
from app.domain.findings import FindingStatus
from app.domain.language import ParserTier
from app.domain.security import VulnerabilityClass
from app.parsing.engine import parse_source
from app.security.cross_file import CrossFileLimits, build_project
from app.security.engine import SecurityAnalysisEngine
from app.security.language_vocab import vocab_for
from app.security.rules.taint_rules import TaintFlowRule
from app.security.taint import analyze_taint

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phase12_repo"


def _scan(tmp_path: Path, files: dict[str, str]):
    paths: list[Path] = []
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths.append(path)
    return SecurityAnalysisEngine().analyze_repository(tmp_path, paths)


def _lines(result, vuln: VulnerabilityClass, filename: str | None = None) -> list[int]:
    return sorted(
        obs.line
        for obs in result.observations
        if obs.vulnerability_class is vuln
        and (filename is None or obs.file_path.endswith(filename))
    )


def test_engine_consumes_project_summaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.security.rules.taint_rules as rules

    seen: list[dict[str, object]] = []
    real = rules.analyze_taint

    def wrapped(graph, sources, externals=None):
        seen.append(dict(externals or {}))
        return real(graph, sources, externals)

    monkeypatch.setattr(rules, "analyze_taint", wrapped)
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def forward(value):\n    return value\n",
            "app.py": "from helpers import forward\nq = forward(request.args.get('q'))\neval(q)\n",
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [3]
    assert any(table for table in seen)


def test_second_pass_diagnostic_names_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = TaintFlowRule.check

    def boom(self, graph, frameworks=(), project=None):
        if project is not None and str(graph.file_path).endswith("early.py"):
            raise RuntimeError("boom-early")
        return original(self, graph, frameworks=frameworks, project=project)

    monkeypatch.setattr(TaintFlowRule, "check", boom)
    with caplog.at_level(logging.WARNING):
        result = _scan(
            tmp_path,
            {
                "helpers.py": "def forward(value):\n    return value\n",
                "early.py": "from helpers import forward\nq = forward(request.args.get('q'))\neval(q)\n",
                "late.py": "x = 1\n",
            },
        )
    matched = [item for item in result.diagnostics if "boom-early" in item.message]
    assert matched
    assert all(item.file_path.endswith("early.py") for item in matched)
    warnings = [rec.message for rec in caplog.records if "boom-early" in rec.message]
    assert warnings
    assert all("early.py" in message and "late.py" not in message for message in warnings)


def test_partial_callee_is_not_a_summary() -> None:
    helper = parse_source("python", Path("helpers.py"), "def run_code(value):\n    eval(value)\n")
    app = parse_source(
        "python",
        Path("app.py"),
        "from helpers import run_code\nrun_code(request.args.get('q'))\n",
    )
    helper.file_path = "helpers.py"
    app.file_path = "app.py"
    helper.diagnostics = replace(helper.diagnostics, has_errors=True, error_count=1)
    project = build_project({"helpers.py": helper, "app.py": app})
    assert project.externals == {}
    assert project.incomplete
    assert any(item.kind == "cross_file_partial" for item in project.diagnostics)


def test_truncated_callee_is_not_a_summary() -> None:
    helper = parse_source("python", Path("helpers.py"), "def run_code(value):\n    eval(value)\n")
    app = parse_source(
        "python",
        Path("app.py"),
        "from helpers import run_code\nrun_code(request.args.get('q'))\n",
    )
    helper.file_path = "helpers.py"
    app.file_path = "app.py"
    helper.diagnostics = replace(helper.diagnostics, truncated=True, message="truncated")
    project = build_project({"helpers.py": helper, "app.py": app})
    assert project.externals == {}
    assert any(item.kind == "cross_file_partial" for item in project.diagnostics)


def test_limits_stop_without_an_extra_edge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {
        "helpers.py": "def run_code(value):\n    eval(value)\n",
        "app.py": "from helpers import run_code\nrun_code(request.args.get('q'))\nq = request.args.get('q')\neval(q)\n",
    }
    monkeypatch.setattr(settings, "taint_max_cross_file_edges", 0)
    edges = _scan(tmp_path / "edges", files)
    assert any("edge limit" in item.message for item in edges.diagnostics)
    assert _lines(edges, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [4]

    monkeypatch.setattr(settings, "taint_max_cross_file_edges", 2000)
    monkeypatch.setattr(settings, "taint_max_cross_file_rounds", 0)
    rounds = _scan(tmp_path / "rounds", files)
    assert any("before any propagation round" in item.message for item in rounds.diagnostics)
    assert _lines(rounds, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [4]

    monkeypatch.setattr(settings, "taint_max_cross_file_rounds", 4)
    monkeypatch.setattr(settings, "taint_max_files", 1)
    limited = _scan(tmp_path / "files", files)
    assert any(item.kind == "cross_file_incomplete" for item in limited.diagnostics)
    assert 2 not in _lines(limited, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py")
    assert _lines(limited, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [4]


def test_round_limit_does_not_look_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "taint_max_cross_file_rounds", 1)
    result = _scan(
        tmp_path,
        {
            "c.py": "def run_code(value):\n    eval(value)\n",
            "b.py": "from c import run_code\n\ndef wrap(value):\n    run_code(value)\n",
            "a.py": "from b import wrap\n\nwrap(request.args.get('q'))\n",
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "a.py") == []
    assert any("max propagation rounds" in item.message for item in result.diagnostics)
    assert result.findings == [] or all(
        f.status is not FindingStatus.VERIFIED for f in result.findings
    )


def test_diagnostics_and_edges_are_deterministic() -> None:
    helper = parse_source("python", Path("helpers.py"), "def forward(value):\n    return value\n")
    app = parse_source(
        "python",
        Path("app.py"),
        "from helpers import forward\nq = forward(request.args.get('q'))\n",
    )
    helper.file_path = "helpers.py"
    app.file_path = "app.py"
    graphs = {"helpers.py": helper, "app.py": app}
    first = build_project(graphs, limits=CrossFileLimits())
    second = build_project(graphs, limits=CrossFileLimits())
    assert [(d.kind, d.file_path, d.message) for d in first.diagnostics] == [
        (d.kind, d.file_path, d.message) for d in second.diagnostics
    ]
    assert [(e.kind, e.source_id, e.target_id) for e in first.edges] == [
        (e.kind, e.source_id, e.target_id) for e in second.edges
    ]
    assert all(
        "::" in edge.target_id and not edge.target_id.startswith("0x") for edge in first.edges
    )


@pytest.mark.parametrize(
    "source",
    [
        "q = 'safe'\neval(q)\nq = request.args.get('q')\n",
        "q = request.args.get('q')\neval(q)\nq = 'safe'\n",
        "q = 'safe'\nq = request.args.get('q')\nq = 'safe'\neval(q)\n",
        "q = request.args.get('q')\nq = 'safe'\nq = request.args.get('q')\neval(q)\n",
    ],
)
def test_use_site_sequences(tmp_path: Path, source: str) -> None:
    result = _scan(tmp_path, {"a.py": source})
    lines = _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION)
    eval_line = source.splitlines().index("eval(q)") + 1
    if "eval(q)\nq = request" in source or "q = 'safe'\neval(q)" in source:
        assert lines == []
    else:
        assert lines == [eval_line]


def test_branch_merge_is_labeled(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {"a.py": "q = 'safe'\nif cond:\n    q = request.args.get('q')\neval(q)\n"},
    )
    obs = next(
        o
        for o in result.observations
        if o.vulnerability_class is VulnerabilityClass.DYNAMIC_EXECUTION
    )
    assert obs.metadata["branch_merge"] == "true"
    assert obs.metadata["path_sensitive"] == "false"
    assert "merge:" in obs.taint_path


def test_scope_shadow_and_sibling(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "a.py": (
                "q = request.args.get('q')\n"
                "def inner():\n"
                "    q = 'safe'\n"
                "    eval(q)\n"
                "eval(q)\n"
                "def other(value):\n"
                "    eval(value)\n"
            )
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "a.py") == [5]


def test_argument_mapping_and_helper_reassignment() -> None:
    helper = parse_source(
        "python",
        Path("helpers.py"),
        "def helper(value):\n    eval(value)\n    value = 'safe'\n    eval(value)\n",
    )
    app = parse_source(
        "python",
        Path("app.py"),
        "from helpers import helper\nhelper(request.args.get('q'))\n",
    )
    helper.file_path = "helpers.py"
    app.file_path = "app.py"
    project = build_project({"helpers.py": helper, "app.py": app})
    ext = project.externals["app.py"]["helper"]
    assert [sink[3] for sink in ext.param_sinks] == [2]


def test_argument_mapping_scan(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def run(label, value):\n    eval(value)\n",
            "app.py": (
                "from helpers import run\n"
                "run(request.args.get('q'), 'safe')\n"
                "run('ok', request.args.get('q'))\n"
            ),
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [3]


def test_class_method_and_not_same_name(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": (
                "class Executor:\n"
                "    def run(self, value):\n"
                "        eval(value)\n"
                "class Reporter:\n"
                "    def run(self, value):\n"
                "        return value\n"
            ),
            "app.py": (
                "from helpers import Executor, Reporter\n"
                "executor = Executor()\n"
                "executor.run(request.args.get('q'))\n"
                "reporter = Reporter()\n"
                "q = reporter.run(request.args.get('q'))\n"
                "eval('safe')\n"
            ),
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [3]
    obs = next(o for o in result.observations if o.line == 3)
    assert obs.metadata["callee_function"] == "Executor.run"
    assert obs.metadata["parameter_index"] == "0"
    assert "::" in obs.metadata["semantic_id"]


def test_explicit_receiver_and_static_method(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": (
                "class Executor:\n"
                "    def run(self, value):\n"
                "        eval(value)\n"
                "    @staticmethod\n"
                "    def stat(value):\n"
                "        eval(value)\n"
            ),
            "app.py": (
                "from helpers import Executor\n"
                "Executor.run(Executor(), request.args.get('q'))\n"
                "Executor.stat(request.args.get('q'))\n"
                "Executor().run(request.args.get('q'))\n"
            ),
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [2, 3, 4]


def test_inheritance_and_dynamic_dispatch_are_not_guessed(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": (
                "class Base:\n"
                "    def run(self, value):\n"
                "        eval(value)\n"
                "class Child(Base):\n"
                "    pass\n"
            ),
            "app.py": (
                "from helpers import Child\n"
                "child = Child()\n"
                "child.run(request.args.get('q'))\n"
                "obj = unknown()\n"
                "obj.run(request.args.get('q'))\n"
            ),
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == []


def test_python_package_reexport_and_relative(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "pkg/helpers.py": "def run_code(value):\n    eval(value)\n",
            "pkg/__init__.py": "from .helpers import run_code\n",
            "app.py": "from pkg import run_code\nrun_code(request.args.get('q'))\n",
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [2]
    assert any(
        obs.metadata.get("callee_file", "").endswith("pkg/helpers.py")
        for obs in result.observations
    )


def test_js_barrel_alias_and_namespace(tmp_path: Path) -> None:
    barrel = _scan(
        tmp_path / "barrel",
        {
            "helpers.js": "function runCode(value) { eval(value); }\n",
            "barrel.js": 'export { runCode as go } from "./helpers.js";\n',
            "app.js": 'import { go } from "./barrel.js";\ngo(req.query.q);\ngo("safe");\n',
        },
    )
    assert _lines(barrel, VulnerabilityClass.DYNAMIC_EXECUTION, "app.js") == [2]

    namespace = _scan(
        tmp_path / "ns",
        {
            "helpers.js": "function runCode(value) { eval(value); }\n",
            "app.js": 'import * as helpers from "./helpers.js";\nhelpers.runCode(req.query.q);\n',
        },
    )
    assert _lines(namespace, VulnerabilityClass.DYNAMIC_EXECUTION, "app.js") == [2]

    star = _scan(
        tmp_path / "star",
        {
            "helpers.js": "function runCode(value) { eval(value); }\n",
            "barrel.js": 'export * from "./helpers.js";\n',
            "app.js": 'import { runCode } from "./barrel.js";\nrunCode(req.query.q);\n',
        },
    )
    assert _lines(star, VulnerabilityClass.DYNAMIC_EXECUTION, "app.js") == []
    assert any(item.kind == "ambiguous_import" for item in star.diagnostics)


def test_third_party_and_unresolved_alias_do_not_flow(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "app.py": (
                "from flask import escape\n"
                "import missing_mod\n"
                "q = request.args.get('q')\n"
                "eval(missing_mod.clean(q))\n"
            )
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == []


def test_cross_file_sanitizer_effects(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": (
                "def clean_html(value):\n"
                "    return html.escape(value)\n"
                "def clean_sql(value):\n"
                '    return value.replace("\'", "")\n'
            ),
            "app.py": (
                "from helpers import clean_html\n"
                "raw = request.args.get('q')\n"
                "q = clean_html(raw)\n"
                "markupsafe.Markup(q)\n"
                "eval(q)\n"
                "cursor.execute(q)\n"
            ),
        },
    )
    assert _lines(result, VulnerabilityClass.XSS, "app.py") == []
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [5]
    assert _lines(result, VulnerabilityClass.SQL_INJECTION, "app.py") == [6]


def test_value_transforms_and_unknown_wrapper(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "helpers.py": "def mystery(value):\n    return other(value)\n",
            "app.py": (
                "from helpers import mystery\n"
                "q = request.args.get('q')\n"
                "s = '{}'.format(q)\n"
                "t = str(q)\n"
                "u = q + 'x'\n"
                "v = f'{q}'\n"
                "w = mystery(q)\n"
                "eval(s)\n"
                "eval(t)\n"
                "eval(u)\n"
                "eval(v)\n"
                "eval(w)\n"
                "n = 'safe' + 'ok'\n"
                "eval(n)\n"
            ),
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION, "app.py") == [8, 9, 10, 11]


def test_container_uses_are_whole_value(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "a.py": (
                "q = request.args.get('q')\n"
                "data = {'q': q}\n"
                "eval(data['q'])\n"
                "items = [q]\n"
                "eval(items[0])\n"
            )
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION) == [3, 5]


def test_comments_strings_and_constants_are_not_sources(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "a.py": (
                "# request.args.get('q')\nq = \"request.args.get('q')\"\neval(q)\neval('safe')\n"
            )
        },
    )
    assert _lines(result, VulnerabilityClass.DYNAMIC_EXECUTION) == []


def test_recursion_terminates_and_depth_is_explicit() -> None:
    mutual = parse_source(
        "python",
        Path("a.py"),
        "def a(value):\n    b(value)\ndef b(value):\n    a(value)\n    eval(value)\n",
    )
    state = analyze_taint(mutual, vocab_for("python").sources)
    assert state.incomplete is False

    deep = parse_source(
        "python",
        Path("deep.py"),
        (
            "def a(v):\n    return b(v)\n"
            "def b(v):\n    return c(v)\n"
            "def c(v):\n    return d(v)\n"
            "def d(v):\n    return e(v)\n"
            "def e(v):\n    return f(v)\n"
            "def f(v):\n    eval(v)\n"
            "q = a(request.args.get('q'))\n"
        ),
    )
    limited = analyze_taint(deep, vocab_for("python").sources)
    assert limited.incomplete
    assert "interprocedural depth limit" in limited.limit_reason


def test_profile_fallback_still_not_dataflow() -> None:
    helper = parse_source("python", Path("helpers.py"), "def run_code(value):\n    eval(value)\n")
    app = parse_source(
        "python",
        Path("app.py"),
        "from helpers import run_code\nrun_code(request.args.get('q'))\n",
    )
    helper.file_path = "helpers.py"
    app.file_path = "app.py"
    helper.parser_tier = ParserTier.PROFILE_FALLBACK
    assert build_project({"helpers.py": helper, "app.py": app}).externals == {}


def test_phase12_corpus_is_deterministic() -> None:
    paths = sorted(path for path in FIXTURE.rglob("*") if path.is_file())
    assert paths
    engine = SecurityAnalysisEngine()

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
                    item.metadata.get("taint_scope", ""),
                )
                for item in result.observations
            )
        )
        diags = tuple((item.kind, item.file_path, item.message) for item in result.diagnostics)
        statuses = tuple(item.status for item in result.findings)
        return obs, diags, statuses

    first = signature()
    second = signature()
    assert first == second
    obs, diags, statuses = first
    assert all(status is not FindingStatus.VERIFIED for status in statuses)
    files = {item[0] for item in obs}
    assert any(name.endswith("app.py") for name in files)
    assert any(name.endswith("app.js") for name in files)
    assert not any(name.endswith("generated/gen.py") for name in files)
    assert not any(name.endswith("ambiguous/app.py") for name in files)
    assert any(
        kind in {"parser_error", "parser_failure", "partial_analysis"} for kind, _, _ in diags
    )
    assert any(kind == "ambiguous_import" for kind, _, _ in diags)
