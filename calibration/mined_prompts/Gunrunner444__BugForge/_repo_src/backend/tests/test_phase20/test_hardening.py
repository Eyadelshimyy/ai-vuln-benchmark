"""Final Phase 13–20 hardening: lexical binding, callee identity, routes, persistence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.mock_provider import MockLLMProvider
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.findings import FindingStatus, HumanReviewState, SecurityFinding, SourceLocation
from app.domain.security import VulnerabilityClass
from app.models.project import Project
from app.parsing.engine import parse_source
from app.repositories.security_finding_repo import SecurityFindingRepository, _to_row, to_domain
from app.security.agent import SecurityAnalysisAgent
from app.security.correlation import ObservationCluster
from app.security.engine import SecurityAnalysisEngine, SecurityScanResult
from app.security.evidence_correlation import correlate_finding
from app.security.finding_intelligence import semantic_clusters
from app.security.language_vocab import vocab_for
from app.security.rules.base import RuleDocumentation, SecurityObservation
from app.security.taint import _interprocedural_edges, analyze_taint

EVAL = "dangerous_dynamic_execution"
_DOC = RuleDocumentation("detects a sink", "evidence", "limited", "false positives")


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


def _obs(*, line: int, text: str, sink_occurrence: str = "") -> SecurityObservation:
    extra = {
        "sink": "eval",
        "taint": "source:request.args",
        "taint_source": "request.args",
        "callee_sink": "eval",
    }
    if sink_occurrence:
        extra["sink_occurrence"] = sink_occurrence
    return SecurityObservation(
        rule_id="sec.taint.eval",
        vulnerability_class=VulnerabilityClass.DYNAMIC_EXECUTION,
        title="Potential dangerous dynamic execution",
        summary="eval may receive tainted data",
        file_path="app.py",
        line=line,
        evidence_text=text,
        confidence="high",
        language="python",
        documentation=_DOC,
        scope_id="module",
        metadata=extra,
        taint_path="source:request.args",
    )


def _static_finding(
    *, key: str, line: int, title: str = "Potential dynamic execution"
) -> SecurityFinding:
    evidence = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary=f"eval at {line}",
        artifact_path="app.py",
        metadata={"line": str(line), "sink": "eval"},
    )
    return SecurityFinding.potential(
        title,
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class="dangerous_dynamic_execution",
        source_location=SourceLocation(file_path="app.py", line=line),
        finding_key=key,
        flow_summary="source request.args → sink eval",
        flow_source="request.args",
        flow_sink="eval",
        analyzer="security_rules",
    )


def test_module_later_def_does_not_hide_earlier_builtin(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'eval(request.args.get("q"))\n\ndef eval(value):\n    return value\n',
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [1]


def test_module_later_assignment_does_not_hide_earlier_builtin(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'def keep(value):\n    return value\neval(request.args.get("q"))\neval = keep\n',
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [3]


def test_function_local_later_assignment_is_not_a_builtin(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def keep(value):",
                "    return value",
                "def outer():",
                '    eval(request.args.get("q"))',
                "    eval = keep",
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == []


def test_function_local_later_nested_def_is_not_a_builtin(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def outer():",
                '    eval(request.args.get("q"))',
                "    def eval(value):",
                "        return value",
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == []


def test_nested_function_does_not_hide_enclosing_builtin(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def outer():",
                '    eval(request.args.get("q"))',
                "    def inner():",
                "        def eval(value):",
                "            return value",
                '        eval(request.args.get("q"))',
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [2]


def test_imported_eval_is_not_the_builtin(tmp_path: Path) -> None:
    _write(tmp_path, "helpers.py", "def eval(value):\n    return value\n")
    _write(
        tmp_path,
        "app.py",
        'from helpers import eval\neval(request.args.get("q"))\n',
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == []


def test_closure_sees_enclosing_function_local(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def keep(value):",
                "    return value",
                "def outer():",
                "    def inner():",
                '        eval(request.args.get("q"))',
                "    eval = keep",
                "    return inner",
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == []


def test_qualified_eval_ignores_local_binding(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def outer():",
                "    def eval(value):",
                "        return value",
                '    obj.eval(request.args.get("q"))',
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [4]


def test_direct_local_call_propagates_return_taint(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "def load():\n    return request.args.get('q')\nvalue = load()\neval(value)\n",
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == [4]


def test_unknown_receiver_does_not_inherit_local_return(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        "def load():\n    return request.args.get('q')\nvalue = external.load()\neval(value)\n",
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == []
    edges = _param_edges(
        "def load():\n    return request.args.get('q')\nvalue = helper.load()\neval(value)\n"
    )
    assert not any(sid.endswith("::value") and "return" in reason for sid, reason in edges)


def test_known_receiver_return_and_ambiguous_class_stay_precise(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "known.py",
        "\n".join(
            [
                "class Box:",
                "    def load(self):",
                "        return request.args.get('q')",
                "box = Box()",
                "value = box.load()",
                "eval(value)",
                "",
            ]
        ),
    )
    _write(
        tmp_path,
        "mystery.py",
        "\n".join(
            [
                "class Box:",
                "    def load(self):",
                "        return request.args.get('q')",
                "value = mystery.load()",
                "eval(value)",
                "",
            ]
        ),
    )
    result = _scan(tmp_path)
    assert _eval_lines(result, "known.py") == [6]
    assert _eval_lines(result, "mystery.py") == []


def test_two_classes_and_imported_same_name_do_not_guess(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "helpers.py",
        "def load():\n    return request.args.get('q')\n",
    )
    _write(
        tmp_path,
        "app.py",
        "\n".join(
            [
                "def load():",
                "    return request.args.get('q')",
                "from helpers import load",
                "value = load()",
                "eval(value)",
                "",
            ]
        ),
    )
    assert _eval_lines(_scan(tmp_path), "app.py") == []
    edges = _param_edges(
        "\n".join(
            [
                "class Left:",
                "    def load(self):",
                "        return request.args.get('q')",
                "class Right:",
                "    def load(self):",
                "        return 'safe'",
                "flag = True",
                "obj = Left() if flag else Right()",
                "value = obj.load()",
                "eval(value)",
                "",
            ]
        )
    )
    assert edges == []


def test_flask_and_fastapi_routes_require_construction() -> None:
    flask = parse_source(
        "python",
        Path("app.py"),
        "from flask import Flask\napp = Flask(__name__)\n@app.route('/item/<id>')\ndef item(id):\n    eval(id)\n",
    )
    fastapi = parse_source(
        "python",
        Path("app.py"),
        "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    assert flask.routes[0].path == "/item/<id>"
    assert fastapi.routes[0].path == "/item/{id}"


def test_express_route_requires_express_construction() -> None:
    graph = parse_source(
        "javascript",
        Path("app.js"),
        'const express = require("express");\nconst app = express();\napp.get("/search", handler);\n',
    )
    assert len(graph.routes) == 1
    assert graph.routes[0].path == "/search"


def test_unrelated_app_and_cache_are_not_routes() -> None:
    other = parse_source(
        "python",
        Path("app.py"),
        "class SomeOtherObject:\n    def get(self, key):\n        return key\napp = SomeOtherObject()\n@app.get('/not-really-a-route/{id}')\ndef handler(id):\n    eval(id)\n",
    )
    unknown = parse_source(
        "python",
        Path("app.py"),
        '@app.get("/item/{id}")\ndef item(id):\n    eval(id)\n',
    )
    cache = parse_source(
        "python",
        Path("app.py"),
        "class Cache:\n    def get(self, key):\n        return key\ncache = Cache()\n@cache.get('/not-a-real-route/{id}')\ndef handler(id):\n    eval(id)\n",
    )
    js_cache = parse_source(
        "javascript",
        Path("app.js"),
        'cache.get("/not-a-route", handler);\n',
    )
    assert other.routes == ()
    assert unknown.routes == ()
    assert cache.routes == ()
    assert js_cache.routes == ()


def test_wrong_language_constructor_is_not_a_route() -> None:
    python_express = parse_source(
        "python",
        Path("app.py"),
        "app = express()\n@app.get('/item/{id}')\ndef item(id):\n    eval(id)\n",
    )
    js_flask = parse_source(
        "javascript",
        Path("app.js"),
        'const app = Flask();\napp.get("/search", handler);\n',
    )
    assert python_express.routes == ()
    assert js_flask.routes == ()


def test_source_location_survives_database_round_trip() -> None:
    evidence = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary="eval may run request data",
        artifact_path="app.py",
        metadata={"line": "10", "sink": "eval"},
    )
    finding = SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class="dangerous_dynamic_execution",
        source_location=SourceLocation(file_path="app.py", line=10),
        finding_key="stable-key",
        flow_summary="source request.args → sink eval",
        flow_source="request.args",
        flow_sink="eval",
        field_path="",
        ai_analysis="model text",
        report_title="Potential dynamic execution",
        report_description="static hint",
    )
    restored = to_domain(_to_row(finding, project_id=None, analysis_id=None))
    assert restored.source_location is not None
    assert restored.source_location.file_path == "app.py"
    assert restored.source_location.line == 10
    assert restored.finding_key == "stable-key"
    assert restored.flow_summary == "source request.args → sink eval"
    assert restored.flow_source == "request.args"
    assert restored.flow_sink == "eval"
    assert restored.ai_analysis == "model text"
    assert restored.report_title == "Potential dynamic execution"
    assert restored.evidence.items[0].metadata["sink"] == "eval"
    row = _to_row(finding, project_id=None, analysis_id=None)
    row.intelligence_json = "{}"
    row.finding_key = None
    row.file_path = "legacy.py"
    row.line = 4
    legacy = to_domain(row)
    assert legacy.source_location is not None
    assert legacy.source_location.file_path == "legacy.py"
    assert legacy.source_location.line == 4
    assert legacy.finding_key == ""


@pytest.mark.asyncio
async def test_repeated_scan_does_not_duplicate_or_downgrade(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    first = _static_finding(key="same-key", line=3)
    other = _static_finding(key="other-key", line=9, title="Potential command injection")
    project = Project(name="p", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    created = await repo.bulk_create([first, other], project_id=project.id, analysis_id=None)
    assert len(created) == 2
    from app.domain.trusted_evidence import issue_for_finding

    proof = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="exploit ran",
        details="shown",
        artifact_path="app.py",
    )
    verified = first.verify(EvidenceBundle.from_items([issue_for_finding(first, proof, "exec-v")]))
    verified_row = await repo.get(created[0].id)
    assert verified_row is not None
    verified_row.status = FindingStatus.VERIFIED.value
    verified_row.evidence_json = _to_row(
        verified, project_id=project.id, analysis_id=None
    ).evidence_json
    verified_row.evidence_tier = verified.evidence_tier.value
    rejected = other.reject()
    rejected_row = await repo.get(created[1].id)
    assert rejected_row is not None
    rejected_row.status = FindingStatus.REJECTED.value
    rejected_row.evidence_json = _to_row(
        rejected, project_id=project.id, analysis_id=None
    ).evidence_json
    await db_session.flush()

    again = await repo.bulk_create(
        [
            _static_finding(key="same-key", line=8),
            _static_finding(key="other-key", line=12, title="Potential command injection"),
        ],
        project_id=project.id,
        analysis_id=None,
    )
    items, total = await repo.list_for_project(project.id)
    assert total == 2
    assert len(again) == 2
    by_key = {item.finding_key: item for item in items}
    assert by_key["same-key"].status == FindingStatus.VERIFIED.value
    assert by_key["same-key"].line == 8
    assert by_key["other-key"].status == FindingStatus.REJECTED.value
    restored = to_domain(by_key["same-key"])
    assert any(item.kind is EvidenceKind.HTTP_RESPONSE for item in restored.evidence.items)
    assert restored.source_location is not None
    assert restored.source_location.line == 8


@pytest.mark.asyncio
async def test_line_shift_reuses_finding_and_ai_survives(
    db_session: AsyncSession, tmp_path: Path
) -> None:
    project = Project(name="p2", repository_path=str(tmp_path))
    db_session.add(project)
    await db_session.flush()
    repo = SecurityFindingRepository(db_session)
    original = replace(
        _static_finding(key="flow-key", line=3),
        ai_analysis="first hypothesis",
        hypothesis="user data may reach eval",
    )
    await repo.bulk_create([original], project_id=project.id, analysis_id=None)
    shifted = _static_finding(key="flow-key", line=6)
    await repo.bulk_create([shifted], project_id=project.id, analysis_id=None)
    items, total = await repo.list_for_project(project.id)
    assert total == 1
    assert items[0].line == 6
    restored = to_domain(items[0])
    assert restored.ai_analysis == "first hypothesis"
    assert restored.hypothesis == "user data may reach eval"
    assert restored.id == original.id


def test_identical_sink_occurrences_stay_distinct(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "app.py",
        'eval(request.args.get("q"))\neval(request.args.get("q"))\n',
    )
    found = [item for item in _scan(tmp_path).findings if item.vulnerability_class == EVAL]
    assert len(found) == 2
    assert found[0].finding_key != found[1].finding_key
    _write(
        tmp_path,
        "app.py",
        '\n\neval(request.args.get("q"))\neval(request.args.get("q"))\n',
    )
    shifted = [item for item in _scan(tmp_path).findings if item.vulnerability_class == EVAL]
    assert {item.finding_key for item in found} == {item.finding_key for item in shifted}


def test_equivalent_rule_observations_without_occurrence_still_collapse() -> None:
    first = _obs(line=4, text='eval(obj["payload"])')
    second = _obs(line=9, text='eval(obj["payload"])')
    assert len(semantic_clusters([first, second])) == 1


@pytest.mark.asyncio
async def test_agent_does_not_downgrade_verified_or_accepted(tmp_path: Path) -> None:
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
        vulnerability_class="dangerous_dynamic_execution",
        source_location=SourceLocation(file_path="app.py", line=3),
        finding_key="verified-key",
    )
    verified = shell.verify([issue_for_finding(shell, proof, "exec-agent")])
    cluster = ObservationCluster(
        vulnerability_class=VulnerabilityClass.DYNAMIC_EXECUTION,
        file_path="app.py",
        observations=(_obs(line=3, text='eval(request.args.get("q"))'),),
    )

    class FrozenAgent(SecurityAnalysisAgent):
        def scan(self, repo_path: Path, file_paths: list[Path] | None = None) -> SecurityScanResult:
            return SecurityScanResult(findings=[verified], clusters=[cluster])

    result = await FrozenAgent(provider=MockLLMProvider()).analyze(tmp_path, use_ai=True)
    assert result.findings[0].status is FindingStatus.VERIFIED
    assert any(
        item.kind is EvidenceKind.HTTP_RESPONSE for item in result.findings[0].evidence.items
    )
    assert any(item.kind is EvidenceKind.AI_ANALYSIS for item in result.findings[0].evidence.items)

    accepted = verified.human_accept()

    class AcceptedAgent(SecurityAnalysisAgent):
        def scan(self, repo_path: Path, file_paths: list[Path] | None = None) -> SecurityScanResult:
            return SecurityScanResult(findings=[accepted], clusters=[cluster])

    kept = await AcceptedAgent(provider=MockLLMProvider()).analyze(tmp_path, use_ai=True)
    assert kept.findings[0].status is FindingStatus.HUMAN_ACCEPTED
    assert kept.findings[0].human_review_state is HumanReviewState.ACCEPTED


def test_contradiction_and_ai_still_do_not_verify() -> None:
    finding = _static_finding(key="k", line=3)
    ai = Evidence(
        kind=EvidenceKind.AI_ANALYSIS,
        source="ai_provider",
        summary="looks exploitable",
        artifact_path="app.py",
        metadata={"finding_key": "k"},
    )
    contradiction = Evidence(
        kind=EvidenceKind.TEST_FAILURE,
        source="local-test",
        summary="not reached",
        artifact_path="app.py",
        metadata={"line": "3", "reached": "false", "contradicts": "true"},
    )
    ai_only = correlate_finding(finding, [ai])
    assert ai_only.status is FindingStatus.POTENTIAL
    with pytest.raises(ValueError):
        ai_only.verify()
    correlated = correlate_finding(finding, [contradiction])
    assert correlated.status is FindingStatus.POTENTIAL
    assert any(item.summary == "not reached" for item in correlated.evidence.items)


def test_correlate_finding_is_not_imported_by_scan_lifecycle() -> None:
    from app.api.v1.endpoints import security as security_api
    from app.security import agent, findings
    from app.services import analysis_service

    for module in (security_api, agent, findings, analysis_service):
        assert "correlate_finding" not in dir(module)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "correlate_finding" not in source
        assert "evidence_correlation" not in source
