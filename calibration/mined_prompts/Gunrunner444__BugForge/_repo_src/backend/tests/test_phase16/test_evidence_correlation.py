"""Phase 16 evidence correlation.

Static, runtime, and AI evidence can be linked. None of those links marks a
finding verified. A contradiction does not delete the static result.
"""

from __future__ import annotations

from pathlib import Path

from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.findings import FindingStatus, SecurityFinding, SourceLocation
from app.domain.security import VulnerabilityClass
from app.security.engine import SecurityAnalysisEngine
from app.security.evidence_correlation import correlate_finding, explain_confidence

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phase14_repo" / "object_field"


def _static() -> SecurityFinding:
    evidence = Evidence(
        kind=EvidenceKind.STATIC_ANALYSIS,
        source="sec.taint.dynamic_execution",
        summary="eval may receive obj.payload",
        artifact_path="app.py",
        metadata={
            "taint": "field:obj.payload:source:request.args",
            "taint_source": "request.args",
            "sink": "eval",
            "field_path": "obj.payload",
            "parser_backend": "cpython_ast",
            "taint_scope": "local",
            "line": "3",
            "vulnerability_class": "dynamic_execution",
        },
    )
    return SecurityFinding.potential(
        "Potential dynamic execution",
        evidence=EvidenceBundle.from_items([evidence]),
        vulnerability_class="dynamic_execution",
        source_location=SourceLocation(file_path="app.py", line=3),
        confidence="high",
    )


def _test_evidence(
    *, reached: str = "true", line: int = 3, vuln: str = "dynamic_execution"
) -> Evidence:
    return Evidence(
        kind=EvidenceKind.TEST_FAILURE,
        source="local-test",
        summary="test reached eval" if reached == "true" else "test did not reach eval",
        artifact_path="app.py",
        metadata={
            "line": str(line),
            "vulnerability_class": vuln,
            "reached": reached,
            "contradicts": "false" if reached == "true" else "true",
        },
    )


def test_static_only_explanation_is_not_verified() -> None:
    finding = _static()
    explained = explain_confidence(finding)
    assert finding.status is FindingStatus.POTENTIAL
    assert explained.static_status == "potential"
    assert explained.runtime_support == "none"
    assert explained.field_path == "obj.payload"
    assert explained.sink == "eval"
    assert "request.args" in explained.source
    assert "parser cpython_ast" in explained.summary
    assert "no independent runtime evidence" in explained.summary


def test_runtime_support_is_attached_and_not_verified() -> None:
    finding = correlate_finding(_static(), [_test_evidence()])
    assert finding.status is FindingStatus.POTENTIAL
    assert not finding.is_verified
    explained = explain_confidence(finding)
    assert explained.runtime_support == "supports"
    assert "test or reproduction" in explained.summary
    kinds = {item.kind for item in finding.evidence.items}
    assert EvidenceKind.STATIC_ANALYSIS in kinds
    assert EvidenceKind.TEST_FAILURE in kinds


def test_contradiction_keeps_the_static_result() -> None:
    finding = correlate_finding(_static(), [_test_evidence(reached="false")])
    assert finding.status is FindingStatus.POTENTIAL
    assert any(item.kind is EvidenceKind.STATIC_ANALYSIS for item in finding.evidence.items)
    assert explain_confidence(finding).runtime_support == "contradicts"
    assert "static result is kept" in explain_confidence(finding).summary


def test_duplicate_evidence_is_not_repeated() -> None:
    item = _test_evidence()
    once = correlate_finding(_static(), [item, item])
    twice = correlate_finding(once, [item])
    failures = [e for e in twice.evidence.items if e.kind is EvidenceKind.TEST_FAILURE]
    assert len(failures) == 1


def test_unrelated_and_ai_evidence_do_not_verify() -> None:
    other = Evidence(
        kind=EvidenceKind.TEST_FAILURE,
        source="local-test",
        summary="different issue",
        artifact_path="other.py",
        metadata={"line": "3", "vulnerability_class": "dynamic_execution"},
    )
    wrong_class = _test_evidence(vuln="sql_injection")
    ai = Evidence.from_ai("This looks exploitable", source="ai_provider")
    ai_same = Evidence(
        kind=EvidenceKind.AI_ANALYSIS,
        source="ai_provider",
        summary="Hypothesis about app.py:3",
        artifact_path="app.py",
        metadata={"line": "3", "vulnerability_class": "dynamic_execution"},
    )
    finding = correlate_finding(_static(), [other, wrong_class, ai, ai_same])
    assert finding.status is FindingStatus.POTENTIAL
    assert not finding.is_verified
    assert all(item.artifact_path != "other.py" for item in finding.evidence.items)
    assert not any("sql" in item.summary for item in finding.evidence.items)
    assert explain_confidence(finding).runtime_support == "ai_only"
    hypothesis = SecurityFinding.from_hypothesis("maybe", "model text")
    assert correlate_finding(hypothesis, [ai]).status is FindingStatus.POTENTIAL


def test_status_transitions_stay_on_the_existing_model() -> None:
    finding = correlate_finding(_static(), [_test_evidence()])
    assert finding.status is FindingStatus.POTENTIAL
    assert not finding.is_verified
    ai = Evidence(
        kind=EvidenceKind.AI_ANALYSIS,
        source="ai_provider",
        summary="AI agrees",
        artifact_path="app.py",
        metadata={"line": "3", "vulnerability_class": "dynamic_execution"},
    )
    ai_only = correlate_finding(_static(), [ai])
    try:
        ai_only.verify()
    except ValueError:
        ai_refused = True
    else:
        ai_refused = False
    assert ai_refused
    executed = Evidence(
        kind=EvidenceKind.HTTP_RESPONSE,
        source="lab-http",
        summary="exploit ran",
        details="shown",
        artifact_path="app.py",
        metadata={
            "line": "3",
            "vulnerability_class": "dynamic_execution",
            "execution_id": "http-1",
        },
    )
    from app.domain.trusted_evidence import issue_for_finding

    attached = correlate_finding(finding, [executed])
    assert attached.status is FindingStatus.POTENTIAL
    verified = attached.verify([issue_for_finding(attached, executed, "http-1")])
    assert verified.status is FindingStatus.VERIFIED
    assert verified.is_verified


def test_correlation_order_is_stable() -> None:
    first = Evidence(
        kind=EvidenceKind.LOG,
        source="b-log",
        summary="second",
        artifact_path="app.py",
        metadata={"line": "3", "vulnerability_class": "dynamic_execution"},
    )
    second = Evidence(
        kind=EvidenceKind.LOG,
        source="a-log",
        summary="first",
        artifact_path="app.py",
        metadata={"line": "3", "vulnerability_class": "dynamic_execution"},
    )
    left = correlate_finding(_static(), [first, second])
    right = correlate_finding(_static(), [second, first])
    assert [item.source for item in left.evidence.items] == [
        item.source for item in right.evidence.items
    ]
    assert [item.source for item in left.evidence.items if item.kind is EvidenceKind.LOG] == [
        "a-log",
        "b-log",
    ]


def test_scanned_finding_explanation_uses_real_metadata() -> None:
    root = FIXTURE
    result = SecurityAnalysisEngine().analyze_repository(root, [root / "app.py"])
    assert result.findings
    finding = result.findings[0]
    assert finding.status is not FindingStatus.VERIFIED
    explained = explain_confidence(finding)
    assert explained.field_path == "obj.payload"
    assert explained.sink == "eval"
    assert explained.parser
    assert finding.vulnerability_class == VulnerabilityClass.DYNAMIC_EXECUTION.value
