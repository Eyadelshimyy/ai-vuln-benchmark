"""Structured security report — vendor-neutral."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.domain.findings import FindingStatus, HumanReviewState, SecurityFinding


@dataclass
class SecurityReport:
    """Local or remote report payload.

    ``submitted_remotely`` is always False for local rendering. Remote
    program submission (HackerOne, etc.) is reserved for a later phase.
    """

    title: str
    findings: tuple[SecurityFinding, ...]
    generated_by: str
    body: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    destination: str = "local"
    submitted_remotely: bool = False

    @property
    def verified_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status is FindingStatus.VERIFIED)

    @property
    def potential_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status is FindingStatus.POTENTIAL)

    @property
    def corroborated_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status is FindingStatus.CORROBORATED)

    @property
    def reproduced_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status is FindingStatus.REPRODUCED)

    @property
    def human_accepted_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status is FindingStatus.HUMAN_ACCEPTED)

    @property
    def rejected_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status is FindingStatus.REJECTED)

    def review_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {state.value: 0 for state in HumanReviewState}
        for finding in self.findings:
            counts[finding.human_review_state.value] += 1
        return counts


def iter_findings(findings: Sequence[SecurityFinding]) -> tuple[SecurityFinding, ...]:
    return tuple(findings)


@dataclass(frozen=True)
class ReportAsset:
    """An asset or target referenced by a later report. Not a live scan target."""

    identifier: str
    kind: str = "source"  # source | host | endpoint | package


@dataclass(frozen=True)
class VulnerabilityReportDraft:
    """Prepared report fields for a later submission phase.

    This is a data structure only. BugForge does not submit to HackerOne (or
    any other program) in this phase.
    """

    title: str
    description: str
    vulnerability_class: str
    asset: ReportAsset | None = None
    source_evidence: tuple[str, ...] = ()
    reproduction_evidence: tuple[str, ...] = ()
    impact: str | None = None
    status: str = "draft"

    def to_mapping(self) -> dict[str, str]:
        return {
            "title": self.title,
            "description": self.description,
            "vulnerability_class": self.vulnerability_class,
            "asset": self.asset.identifier if self.asset else "",
            "asset_kind": self.asset.kind if self.asset else "",
            "impact": self.impact or "",
            "status": self.status,
        }


def draft_from_finding(finding: SecurityFinding) -> VulnerabilityReportDraft:
    """Populate draft fields from a potential/corroborated finding."""
    location = finding.source_location
    asset = None
    if location is not None:
        asset = ReportAsset(identifier=location.file_path, kind="source")
    elif finding.asset:
        asset = ReportAsset(identifier=finding.asset, kind="source")
    elif finding.target:
        asset = ReportAsset(identifier=finding.target, kind="host")
    source_evidence = tuple(item.summary for item in finding.evidence.items)
    return VulnerabilityReportDraft(
        title=finding.report_title or finding.title,
        description=finding.report_description or finding.description or finding.hypothesis or "",
        vulnerability_class=finding.vulnerability_class or "unknown",
        asset=asset,
        source_evidence=source_evidence,
        reproduction_evidence=(),
        impact=finding.impact,
        status="draft",
    )
