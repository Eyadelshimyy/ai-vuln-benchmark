"""Local structured security report renderer. Does not upload anywhere."""

from __future__ import annotations

from collections.abc import Sequence

from app.adapters.reports.base import ReportProvider
from app.domain.evidence import EvidenceBundle
from app.domain.findings import FindingStatus, HumanReviewState, SecurityFinding
from app.domain.reports import SecurityReport
from app.plugins.errors import AdapterNotImplementedError


class LocalReportProvider(ReportProvider):
    @property
    def provider_id(self) -> str:
        return "local"

    def render(
        self,
        findings: Sequence[SecurityFinding],
        *,
        evidence: EvidenceBundle | None = None,
    ) -> SecurityReport:
        lines = [
            "# BugForge Security Report",
            "",
            "This is a **local rendering**. It has not been submitted to any remote",
            "program (HackerOne or otherwise).",
            "",
            "Hypotheses are not verified findings. Potential and corroborated",
            "static hypotheses still require independent observational or",
            "executable evidence before they can be marked verified.",
            "",
        ]
        if not findings:
            lines.append("No findings.")
        for finding in findings:
            lines.append(f"## [{finding.status.value}] {finding.title}")
            lines.append(f"- Human review: {finding.human_review_state.value}")
            if finding.vulnerability_class:
                lines.append(f"- Class: {finding.vulnerability_class}")
            if finding.target:
                lines.append(f"- Target: {finding.target}")
            if finding.hypothesis:
                lines.append(f"- Hypothesis: {finding.hypothesis}")
            if finding.description:
                lines.append(f"- Description: {finding.description}")
            if finding.evidence:
                lines.append(f"- Evidence items: {len(finding.evidence)}")
                for item in finding.evidence.items:
                    provenance = item.provenance.value if item.provenance else "unknown"
                    lines.append(f"  - [{item.kind.value} / {provenance}] {item.summary}")
            lines.append("")

        extra = evidence or EvidenceBundle()
        potential = sum(1 for f in findings if f.status is FindingStatus.POTENTIAL)
        corroborated = sum(1 for f in findings if f.status is FindingStatus.CORROBORATED)
        reproduced = sum(1 for f in findings if f.status is FindingStatus.REPRODUCED)
        verified = sum(1 for f in findings if f.status is FindingStatus.VERIFIED)
        human_accepted = sum(1 for f in findings if f.status is FindingStatus.HUMAN_ACCEPTED)
        rejected = sum(1 for f in findings if f.status is FindingStatus.REJECTED)
        unreviewed = sum(1 for f in findings if f.human_review_state is HumanReviewState.UNREVIEWED)
        metadata = {
            "destination": "local",
            "submitted_remotely": "false",
            "finding_count": str(len(findings)),
            "potential_count": str(potential),
            "corroborated_count": str(corroborated),
            "reproduced_count": str(reproduced),
            "verified_count": str(verified),
            "human_accepted_count": str(human_accepted),
            "rejected_count": str(rejected),
            "unreviewed_count": str(unreviewed),
            "standalone_evidence_count": str(len(extra)),
        }
        return SecurityReport(
            title="BugForge Security Report",
            findings=tuple(findings),
            generated_by=self.provider_id,
            body="\n".join(lines).strip() + "\n",
            metadata=metadata,
            destination="local",
            submitted_remotely=False,
        )

    def submit(self, report: SecurityReport) -> str:
        raise AdapterNotImplementedError(
            "LocalReportProvider does not submit reports remotely. "
            "Use render() for local output. HackerOne submission is reserved "
            "for a later phase."
        )
