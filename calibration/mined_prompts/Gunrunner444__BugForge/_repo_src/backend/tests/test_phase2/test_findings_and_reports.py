"""Phase 2 finding persistence, reports, and HackerOne draft structures."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.adapters.reports import LocalReportProvider
from app.domain.findings import FindingStatus, SecurityFinding
from app.domain.reports import VulnerabilityReportDraft, draft_from_finding
from app.domain.security import EvidenceTier
from app.plugins.errors import AdapterNotImplementedError


def test_draft_from_finding_is_not_a_submission() -> None:
    finding = SecurityFinding.potential(
        "Possible SQLi",
        hypothesis="user input reaches execute",
        vulnerability_class="sql_injection",
        impact="data disclosure (hypothesized)",
        report_title="Possible SQL injection in search",
        report_description="Static observations only.",
        asset="app.py",
    )
    draft = draft_from_finding(finding)
    assert isinstance(draft, VulnerabilityReportDraft)
    assert draft.status == "draft"
    assert draft.reproduction_evidence == ()
    assert "sql" in draft.vulnerability_class


def test_local_report_includes_corroborated() -> None:
    findings = [
        SecurityFinding.potential("A"),
        replace(
            SecurityFinding.potential("B"),
            status=FindingStatus.CORROBORATED,
            evidence_tier=EvidenceTier.CORROBORATED,
        ),
    ]
    report = LocalReportProvider().render(findings)
    assert report.potential_count == 1
    assert report.corroborated_count == 1
    assert report.submitted_remotely is False
    assert "[corroborated]" in report.body
    with pytest.raises(AdapterNotImplementedError):
        LocalReportProvider().submit(report)


def test_ai_hypothesis_tier() -> None:
    finding = SecurityFinding.from_hypothesis("X", "y")
    assert finding.evidence_tier is EvidenceTier.AI_HYPOTHESIS
    assert finding.status is FindingStatus.POTENTIAL
