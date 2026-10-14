"""Regression tests for Phase 13–20 semantic integrity.

These checks lock use-site behavior. They do not add a second analysis engine.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.findings import FindingStatus, SecurityFinding, SourceLocation
from app.parsing.engine import parse_source
from app.repositories.security_finding_repo import _to_row, to_domain
from app.security.cross_file import CrossFileLimits, build_project
from app.security.engine import SecurityAnalysisEngine
from app.security.evidence_correlation import correlate_finding
from app.security.findings import attach_ai_hypothesis
from app.security.language_vocab import vocab_for
from app.security.taint import _interprocedural_edges, analyze_taint

EVAL = "dangerous_dynamic_execution"


def _write(root: Path, name: str, source: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _scan(root: Path):
    files = [
        path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".js", ".ts"}
    ]
    return SecurityAnalysisEngine().analyze_repository(root, files)


def _eval_lines(result, filename: str) -> list[int]:
    return sorted(
        obs.line
        for obs in result.observations
        if obs.vulnerability_class.value == EVAL and str(obs.file_path).endswith(filename)
    )


def _graphs(files: dict[str, str]) -> dict:
    graphs = {}
    for name, text in files.items():
        language = "javascript" if name.endswith(".js") else "python"
        graph = parse_source(language, Path(name), text)
        graph.file_path = name
        graphs[name] = graph
    return graphs


def _param_edges(source: str) -> list[tuple[str, str]]:
    graph = parse_source("python", Path("app.py"), source)
    vocab = vocab_for("python")
    patterns = tuple(pattern for src in vocab.sources for pattern in src.patterns)
    state = analyze_taint(graph, vocab.sources)
    return [
        (sid, reason)
        for sid, reason, _idx in _interprocedural_edges(
            graph, state.final_map(), patterns, state=state
        )
        if "source:request.args" in reason
    ]


def _finding(
    *,
    line: int,
    title: str = "Potential dynamic execution",
    finding_key: str = "",
    sink: str = "eval",
) -> SecurityFinding:
    evidence = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary=f"eval at {line}",
        artifact_path="app.py",
        metadata={
            "line": str(line),
            "vulnerability_class": "dynamic_execution",
            "sink": sink,
        },
    )
    return SecurityFinding.potential(
        title,
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class="dynamic_execution",
        source_location=SourceLocation(file_path="app.py", line=line),
        finding_key=finding_key,
    )


def test_builtin_call_before_later_definition_stays_a_sink(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'eval(request.args.get("q"))\n\ndef eval(value):\n    return value\n',
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [1]


def test_builtin_call_after_local_definition_is_not_a_sink(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'def eval(value):\n    return value\n\neval(request.args.get("q"))\n',
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == []


def test_module_assignment_shadows_only_later_calls(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "before.py",
        'def keep(value):\n    return value\neval(request.args.get("q"))\neval = keep\n',
    )
    _write(
        tmp_path,
        "after.py",
        'def keep(value):\n    return value\neval = keep\neval(request.args.get("q"))\n',
    )
    result = _scan(tmp_path)
    assert _eval_lines(result, "before.py") == [3]
    assert _eval_lines(result, "after.py") == []


def test_nested_function_shadows_only_its_enclosing_scope(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def outer():",
                "    def eval(value):",
                "        return value",
                '    eval(request.args.get("q"))',
                "",
                'eval(request.args.get("q"))',
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [6]


def test_qualified_eval_is_not_hidden_by_a_local_function(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'def eval(value):\n    return value\nobj.eval(request.args.get("q"))\n',
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [3]


def test_same_name_in_another_function_does_not_hide_the_builtin(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def outer():",
                "    def eval(value):",
                "        return value",
                "    return eval",
                "",
                'eval(request.args.get("q"))',
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [6]


def test_unknown_receiver_does_not_resolve_a_unique_method() -> None:
    edges = _param_edges(
        "\n".join(
            [
                "class Executor:",
                "    def run(self, value):",
                "        eval(value)",
                "",
                'mystery.run(request.args.get("q"))',
                "",
            ]
        )
    )
    assert edges == []


def test_known_receiver_and_constructor_map_the_user_argument() -> None:
    edges = _param_edges(
        "\n".join(
            [
                "class Executor:",
                "    def run(self, value):",
                "        eval(value)",
                "",
                "obj = Executor()",
                'obj.run(request.args.get("q"))',
                'Executor().run(request.args.get("q"))',
                'Executor.run(Executor(), request.args.get("q"))',
                "",
            ]
        )
    )
    targets = [sid for sid, _reason in edges]
    assert targets
    assert all(sid.endswith("::value") for sid in targets)
    assert not any(sid.endswith("::self") for sid in targets)
    assert len(targets) == 3


def test_staticmethod_does_not_shift_and_classmethod_does() -> None:
    static_edges = _param_edges(
        "\n".join(
            [
                "class Box:",
                "    @staticmethod",
                "    def run(value):",
                "        eval(value)",
                "",
                'Box.run(request.args.get("q"))',
                "box = Box()",
                'box.run(request.args.get("q"))',
                "",
            ]
        )
    )
    assert [sid for sid, _reason in static_edges] == [
        "module/class:Box/method:run::value",
        "module/class:Box/method:run::value",
    ]
    class_edges = _param_edges(
        "\n".join(
            [
                "class Box:",
                "    @classmethod",
                "    def run(cls, value):",
                "        eval(value)",
                "",
                'Box.run(request.args.get("q"))',
                "",
            ]
        )
    )
    assert [sid for sid, _reason in class_edges] == ["module/class:Box/method:run::value"]


def test_two_classes_resolve_only_the_known_receiver() -> None:
    edges = _param_edges(
        "\n".join(
            [
                "class Left:",
                "    def run(self, value):",
                "        eval(value)",
                "class Right:",
                "    def run(self, value):",
                "        return value",
                "",
                "left = Left()",
                'left.run(request.args.get("q"))',
                "right = Right()",
                'right.run("safe")',
                "",
            ]
        )
    )
    assert [sid for sid, _reason in edges] == ["module/class:Left/method:run::value"]


def test_ambiguous_receiver_and_inheritance_stay_unresolved() -> None:
    ambiguous = _param_edges(
        "\n".join(
            [
                "class Left:",
                "    def run(self, value):",
                "        eval(value)",
                "class Right:",
                "    def run(self, value):",
                "        eval(value)",
                "",
                "flag = True",
                "obj = Left() if flag else Right()",
                'obj.run(request.args.get("q"))',
                "",
            ]
        )
    )
    inherited = _param_edges(
        "\n".join(
            [
                "class Base:",
                "    def run(self, value):",
                "        eval(value)",
                "class Child(Base):",
                "    pass",
                "",
                "child = Child()",
                'child.run(request.args.get("q"))',
                "",
            ]
        )
    )
    assert ambiguous == []
    assert inherited == []


def test_multiline_assignment_uses_the_outer_call(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "helpers.py",
        "\n".join(
            [
                "def helper(value):",
                "    return value",
                "def wrap(value):",
                "    return (",
                "        helper(",
                "            value",
                "        )",
                "    )",
                "def clean(value):",
                "    return 'ok'",
                "",
            ]
        ),
    )
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "from helpers import helper, wrap, clean",
                'q = request.args.get("q")',
                "direct = helper(q)",
                "value = (",
                "    wrap(",
                "        q",
                "    )",
                ")",
                "safe = (",
                "    clean(",
                "        helper(q)",
                "    )",
                ")",
                "eval(direct)",
                "eval(value)",
                "eval(safe)",
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [14, 15]


def test_fake_route_names_are_not_http_routes(tmp_path: Path) -> None:
    python = "\n".join(
        [
            "class Cache:",
            "    def get(self, key):",
            "        return key",
            "",
            "cache = Cache()",
            "",
            '@cache.get("/not-a-real-route/{id}")',
            "def handler(id):",
            "    eval(id)",
            "",
        ]
    )
    _write(tmp_path, "app.py", python)
    _write(
        tmp_path,
        "cache.js",
        'function handler(id) { eval(id); }\ncache.get("/not-a-route", handler);\n',
    )
    graphs = _graphs({"app.py": python, "cache.js": 'cache.get("/not-a-route", handler);\n'})
    assert graphs["app.py"].routes == ()
    assert graphs["cache.js"].routes == ()
    assert _eval_lines(_scan(tmp_path), "app.py") == []

    routed = parse_source(
        "python",
        Path("route.py"),
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    assert len(routed.routes) == 1
    assert routed.routes[0].path == "/item/{id}"
    express = parse_source(
        "javascript",
        Path("app.js"),
        'const express = require("express");\nconst app = express();\napp.get("/search", (req, res) => { eval(req.query.q); });\n',
    )
    assert len(express.routes) == 1
    assert express.routes[0].path == "/search"


def test_evidence_attaches_only_to_the_matching_finding() -> None:
    first = _finding(line=3, title="first", finding_key="key-first")
    second = _finding(line=9, title="second", finding_key="key-second")
    exact = Evidence(
        kind=EvidenceKind.TEST_FAILURE,
        source="local-test",
        summary="reached line 3",
        artifact_path="app.py",
        metadata={"line": "3", "vulnerability_class": "dynamic_execution"},
    )
    wrong = Evidence(
        kind=EvidenceKind.TEST_FAILURE,
        source="local-test",
        summary="wrong line",
        artifact_path="app.py",
        metadata={"line": "4", "vulnerability_class": "dynamic_execution"},
    )
    missing = Evidence(
        kind=EvidenceKind.TEST_FAILURE,
        source="local-test",
        summary="no line",
        artifact_path="app.py",
        metadata={"vulnerability_class": "dynamic_execution"},
    )
    malformed = Evidence(
        kind=EvidenceKind.LOG,
        source="local-test",
        summary="bad line",
        artifact_path="app.py",
        metadata={"line": "not-a-number", "vulnerability_class": "dynamic_execution"},
    )
    identified = Evidence(
        kind=EvidenceKind.REPRODUCTION,
        source="reproducer",
        summary="identity match",
        artifact_path="app.py",
        metadata={"finding_key": "key-second", "line": "3"},
    )
    attached = correlate_finding(first, [exact, wrong, missing, malformed], peers=(second,))
    other = correlate_finding(second, [exact, missing, identified], peers=(first,))
    assert [
        item.summary
        for item in attached.evidence.items
        if item.kind is not EvidenceKind.STATIC_ANALYSIS
    ] == ["reached line 3"]
    # A client finding key does not override a conflicting line.
    assert all(item.summary != "identity match" for item in other.evidence.items)
    assert all(item.summary != "reached line 3" for item in other.evidence.items)
    assert all(
        "no line" not in item.summary for item in attached.evidence.items + other.evidence.items
    )
    assert all(
        "bad line" not in item.summary for item in attached.evidence.items + other.evidence.items
    )
    assert attached.status is FindingStatus.POTENTIAL
    assert other.status is FindingStatus.POTENTIAL


def test_ai_evidence_and_contradictions_do_not_verify() -> None:
    finding = _finding(line=3, finding_key="key-one")
    ai = Evidence(
        kind=EvidenceKind.AI_ANALYSIS,
        source="ai_provider",
        summary="looks exploitable",
        artifact_path="app.py",
        metadata={
            "finding_key": "key-one",
            "line": "3",
            "vulnerability_class": "dynamic_execution",
        },
    )
    contradiction = Evidence(
        kind=EvidenceKind.TEST_FAILURE,
        source="local-test",
        summary="not reached",
        artifact_path="app.py",
        metadata={
            "line": "3",
            "vulnerability_class": "dynamic_execution",
            "reached": "false",
            "contradicts": "true",
        },
    )
    ai_only = correlate_finding(finding, [ai])
    assert ai_only.status is FindingStatus.POTENTIAL
    assert any(item.kind is EvidenceKind.AI_ANALYSIS for item in ai_only.evidence.items)
    try:
        ai_only.verify()
    except ValueError:
        refused = True
    else:
        refused = False
    assert refused
    correlated = correlate_finding(finding, [contradiction])
    assert correlated.status is FindingStatus.POTENTIAL
    assert any(item.summary == "not reached" for item in correlated.evidence.items)


def test_evidence_metadata_survives_a_database_round_trip() -> None:
    metadata = {
        "line": 4,
        "vulnerability_class": "dynamic_execution",
        "rule_id": "sec.taint.dynamic_execution",
        "parser_backend": "cpython_ast",
        "taint_path": "source:request.args",
        "field_path": "obj.payload",
        "scope_id": "module",
        "sink": "eval",
        "argument_index": "0",
        "relationship": "import",
    }
    evidence = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary="eval may run request data",
        details="flow",
        artifact_path="app.py",
        metadata=dict(metadata),
    )
    finding = SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class="dynamic_execution",
        source_location=SourceLocation(file_path="app.py", line=4),
    )
    row = _to_row(finding, project_id=None, analysis_id=None)
    restored = to_domain(row)
    assert restored.evidence.items[0].metadata == metadata
    assert restored.evidence.items[0].summary == evidence.summary
    assert restored.evidence.items[0].artifact_path == "app.py"
    assert restored.status is FindingStatus.POTENTIAL

    row.evidence_json = json.dumps(
        [
            {
                "kind": "static_analysis",
                "source": "legacy",
                "summary": "old row",
                "details": "",
                "artifact_path": "app.py",
            },
            {"kind": "log", "source": "legacy", "summary": "bad meta", "metadata": "not-a-number"},
        ]
    )
    legacy = to_domain(row)
    assert legacy.evidence.items[0].metadata == {}
    assert legacy.evidence.items[1].metadata == {}
    row.evidence_json = "{not json"
    assert to_domain(row).evidence.items == ()


def test_ai_hypothesis_does_not_downgrade_verified_findings() -> None:
    from app.domain.trusted_evidence import issue_for_finding

    proof = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="exploit ran",
        details="shown",
        artifact_path="app.py",
    )
    shell = SecurityFinding.potential(
        "Potential dynamic execution",
        vulnerability_class="dynamic_execution",
    )
    verified = shell.verify(EvidenceBundle.from_items([issue_for_finding(shell, proof, "exec-v")]))
    updated = attach_ai_hypothesis(verified, hypothesis="maybe", analysis="model text")
    assert updated.status is FindingStatus.VERIFIED
    assert any(item.kind is EvidenceKind.HTTP_RESPONSE for item in updated.evidence.items)
    assert any(item.kind is EvidenceKind.AI_ANALYSIS for item in updated.evidence.items)
    accepted = updated.human_accept()
    kept = attach_ai_hypothesis(accepted, hypothesis="still maybe", analysis="more text")
    assert kept.status is FindingStatus.HUMAN_ACCEPTED
    assert any(item.summary == "exploit ran" for item in kept.evidence.items)


def test_cross_file_relationship_is_charged_once_across_rounds() -> None:
    files = {
        "d.py": "def sink(value):\n    eval(value)\n",
        "c.py": "from d import sink\ndef mid(value):\n    sink(value)\n",
        "b.py": "from c import mid\ndef helper(value):\n    mid(value)\n",
        "a.py": 'from b import helper\nhelper(request.args.get("q"))\n',
    }
    graphs = _graphs(files)
    project = build_project(
        graphs, limits=CrossFileLimits(max_edges=6, max_rounds=6, max_import_depth=3)
    )
    assert project.externals["a.py"]["helper"].param_sinks
    assert project.externals["b.py"]["mid"].param_sinks
    assert not project.incomplete
    assert not any("edge limit" in item.message for item in project.diagnostics)

    bounded = build_project(graphs, limits=CrossFileLimits(max_edges=1, max_rounds=6))
    assert bounded.incomplete
    assert any("edge limit" in item.message for item in bounded.diagnostics)
    disabled = build_project(graphs, limits=CrossFileLimits(max_edges=0))
    assert disabled.incomplete
    assert disabled.externals == {}
