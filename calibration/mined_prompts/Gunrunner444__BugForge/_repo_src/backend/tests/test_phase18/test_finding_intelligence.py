"""Phase 18 finding identity, duplicate suppression, and explanations.

Static findings stay potential or corroborated. Explanations include only
relationships the observation recorded.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.security import VulnerabilityClass
from app.models.security_finding import DBSecurityFinding
from app.repositories.security_finding_repo import to_security_response
from app.security.engine import SecurityAnalysisEngine
from app.security.finding_intelligence import explain_cluster, semantic_clusters
from app.security.rules.base import RuleDocumentation, SecurityObservation

_DOC = RuleDocumentation("detects a sink", "evidence", "limited", "false positives")


def _obs(
    *,
    line: int = 3,
    rule_id: str = "sec.taint.eval",
    sink: str = "eval",
    field_path: str = "",
    text: str = 'eval(obj["payload"])',
    taint: str = 'field:obj["payload"]:source:request.args',
    **metadata: str,
) -> SecurityObservation:
    extra = {
        "sink": sink,
        "taint": taint,
        "taint_source": "request.args",
        "field_path": field_path,
        "parser_backend": "cpython_ast",
        "parser_tier": "full_ast",
    }
    extra.update(metadata)
    return SecurityObservation(
        rule_id=rule_id,
        vulnerability_class=VulnerabilityClass.DYNAMIC_EXECUTION,
        title="Potential dangerous dynamic execution",
        summary="eval may receive tainted data",
        file_path="app.py",
        line=line,
        evidence_text=text,
        confidence="high",
        language="python",
        documentation=_DOC,
        parser_backend="cpython_ast",
        parser_tier="full_ast",
        scope_id="module",
        metadata=extra,
        taint_path=taint,
    )


def _scan(root: Path, name: str, source: str):
    path = root / name
    path.write_text(source, encoding="utf-8")
    return SecurityAnalysisEngine().analyze_repository(root, [path])


def test_line_shift_keeps_finding_key(tmp_path: Path) -> None:
    source = 'obj = {}\nobj["payload"] = request.args.get("q")\neval(obj["payload"])\n'
    first = _scan(tmp_path, "app.py", source)
    second = _scan(tmp_path, "app.py", "\n\n" + source)
    before = _dynamic(first.findings)
    after = _dynamic(second.findings)
    assert len(before) == 1
    assert len(after) == 1
    assert before[0].finding_key
    assert before[0].finding_key == after[0].finding_key
    assert before[0].source_location is not None
    assert after[0].source_location is not None
    assert before[0].source_location.line != after[0].source_location.line
    assert before[0].status.value == "potential"
    assert "not verified" in before[0].evidence_summary
    assert "source request.args" in before[0].flow_summary
    assert "field" in before[0].flow_summary
    assert "sink eval" in before[0].flow_summary
    assert "alias" not in before[0].flow_summary
    assert before[0].field_path == 'obj["payload"]'


def test_distinct_sinks_stay_separate(tmp_path: Path) -> None:
    source = 'eval(request.args.get("q"))\neval(request.args.get("other"))\n'
    result = _scan(tmp_path, "app.py", source)
    found = _dynamic(result.findings)
    assert len(found) == 2
    assert found[0].finding_key != found[1].finding_key
    assert all(item.status.value == "potential" for item in found)


def test_equivalent_observations_collapse() -> None:
    first = _obs(rule_id="sec.taint.eval", line=4)
    second = _obs(rule_id="sec.indicator.eval", line=9)
    clusters = semantic_clusters([second, first])
    assert len(clusters) == 1
    explained = explain_cluster(clusters[0].observations)
    again = explain_cluster(clusters[0].observations)
    assert explained == again
    assert "alias" not in explained.summary
    assert explained.field_path == 'obj["payload"]'


def test_different_fields_stay_separate() -> None:
    left = _obs(field_path='obj["payload"]', text='eval(obj["payload"])')
    right = _obs(
        field_path="obj.other",
        text="eval(obj.other)",
        taint="field:obj.other:source:request.args",
    )
    clusters = semantic_clusters([left, right])
    assert len(clusters) == 2
    assert explain_cluster((left,)).related_group != explain_cluster((right,)).related_group


def test_recorded_alias_and_helper_are_explained() -> None:
    obs = _obs(
        relationship="alias",
        callee_file="helpers.py",
        callee_function="run_code",
        caller_file="app.py",
        re_export="false",
    )
    explained = explain_cluster((obs,))
    assert explained.summary == (
        'source request.args → field obj["payload"] → alias → helper helpers.py::run_code → sink eval'
    )
    assert explained.files_crossed == "app.py,helpers.py"
    assert "re_export" not in explained.summary


def test_partial_parser_is_explicit() -> None:
    obs = _obs(
        parser_complete="false",
        analysis_incomplete="partial_importer",
        parser_tier="profile_fallback",
    )
    explained = explain_cluster((obs,))
    assert "callee parse is partial" in explained.parser_completeness
    assert "profile fallback is not dataflow" in explained.parser_completeness
    assert explained.analysis_incomplete == "partial_importer"
    assert "not verified" in explained.evidence_summary


def test_cross_file_scan_explains_real_helper(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text(
        "def run_code(value):\n    eval(value)\n", encoding="utf-8"
    )
    app = tmp_path / "app.py"
    app.write_text(
        "\n".join(
            [
                "from helpers import run_code",
                "alias = run_code",
                "obj = {}",
                'obj["payload"] = request.args.get("q")',
                'alias(obj["payload"])',
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = SecurityAnalysisEngine().analyze_repository(tmp_path, [tmp_path / "helpers.py", app])
    found = [
        item
        for item in result.findings
        if item.source_location is not None and item.source_location.file_path == "app.py"
    ]
    assert found
    chosen = found[0]
    assert chosen.status.value in {"potential", "corroborated"}
    assert chosen.status.value != "verified"
    assert "alias" in chosen.flow_summary
    assert "helper helpers.py::run_code" in chosen.flow_summary
    assert "sink eval" in chosen.flow_summary
    assert "helpers.py" in chosen.files_crossed
    assert chosen.finding_key


def test_api_response_keeps_old_fields_and_adds_flow() -> None:
    row = DBSecurityFinding(
        id=uuid4(),
        title="Potential dangerous dynamic execution",
        status="potential",
        evidence_tier="static_indicator",
        description="Static analysis produced a potential security issue.",
        vulnerability_class="dangerous_dynamic_execution",
        confidence="high",
        file_path="app.py",
        line=3,
        rule_ids="sec.taint.eval",
        observation_refs="sec.taint.eval:app.py:3",
        created_at=datetime.now(UTC),
        finding_key="abc123",
        intelligence_json=json.dumps(
            {
                "finding_key": "abc123",
                "flow_summary": "source request.args → sink eval",
                "flow_source": "request.args",
                "flow_sink": "eval",
                "field_path": "",
                "evidence_summary": "static analysis; not verified",
            }
        ),
    )
    response = to_security_response(row)
    assert response.title == row.title
    assert response.status == "potential"
    assert response.file_path == "app.py"
    assert response.line == 3
    assert response.finding_key == "abc123"
    assert response.flow_source == "request.args"
    assert response.flow_sink == "eval"
    assert "verified" not in response.status


def _dynamic(findings: list) -> list:
    return [item for item in findings if item.vulnerability_class == "dangerous_dynamic_execution"]
