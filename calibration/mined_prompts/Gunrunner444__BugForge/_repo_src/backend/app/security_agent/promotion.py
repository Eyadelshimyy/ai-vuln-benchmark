"""Deterministic hypothesis → finding promotion. The agent cannot verify."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind
from app.domain.findings import SecurityFinding
from app.domain.lifecycle_policy import (
    _RESEARCH_OBSERVATION_KINDS,
    can_corroborate,
    positive_reproduction,
)
from app.domain.target_identity import semantic_target_identity
from app.domain.trusted_evidence import issue_for_finding
from app.security_agent.correlation import finding_fingerprint
from app.security_agent.schemas import ResearchHypothesis
from app.security_agent.states import HypothesisStatus, ReproductionOutcome
from app.security_testing.errors import RestrictedActivityError


def promote_hypothesis(session: Any, hypothesis: ResearchHypothesis) -> SecurityFinding | None:
    """Potential → corroborated → reproduction required → reproduced.

    Verification is owned by SecurityFinding.verify() with observational evidence.
    The agent never calls verify.
    """
    existing = _existing(session, hypothesis)
    evidence = _evidence_from_graph(session, hypothesis)
    if existing is None:
        finding = SecurityFinding.from_hypothesis(
            hypothesis.title,
            hypothesis.reason,
            vulnerability_class=hypothesis.vulnerability_class,
            target=hypothesis.target,
            confidence=hypothesis.confidence,
            impact=hypothesis.impact or None,
        )
    else:
        finding = existing
    project_id = str(getattr(session, "project_id", "") or "")
    if project_id and finding.project_id != project_id:
        finding = replace(finding, project_id=project_id)
    if evidence:
        stamped = _stamp_research_evidence(finding, evidence)
        merged = (
            finding.evidence.extend(stamped)
            if finding.evidence
            else EvidenceBundle.from_items(stamped)
        )
        finding = replace(finding, evidence=merged)
    if (
        can_corroborate(
            finding.evidence,
            target_id=semantic_target_identity(finding),
            finding_id=str(finding.id),
            finding_key=finding.finding_key,
            project_id=finding.project_id,
        )
        and finding.status.value == "potential"
    ):
        finding = finding.corroborate()
        if hypothesis.status is HypothesisStatus.OPEN:
            hypothesis.status = HypothesisStatus.SUPPORTED
    elif (
        hypothesis.status is HypothesisStatus.REQUIRES_REPRODUCTION
        and finding.status.value == "potential"
        and can_corroborate(
            finding.evidence,
            target_id=semantic_target_identity(finding),
            finding_id=str(finding.id),
            finding_key=finding.finding_key,
            project_id=finding.project_id,
        )
    ):
        finding = finding.corroborate()
    if existing is None:
        session.findings.append(finding)
    else:
        _replace_finding(session, finding)
    return finding


def apply_reproduction(
    session: Any,
    hypothesis: ResearchHypothesis,
    *,
    outcome: ReproductionOutcome,
    evidence: list[Evidence] | None = None,
) -> SecurityFinding | None:
    finding = promote_hypothesis(session, hypothesis)
    if finding is None:
        return None
    if outcome is ReproductionOutcome.REPRODUCED:
        positive = [item for item in (evidence or []) if positive_reproduction(item)]
        if positive:
            finding = finding.reproduce(positive)
        else:
            # Reproduction without a successful reproduction record cannot advance.
            hypothesis.status = HypothesisStatus.REQUIRES_REPRODUCTION
            return finding
        # AI still cannot set VERIFIED. Reproduced findings wait for BugForge verify().
        hypothesis.status = HypothesisStatus.SUPPORTED
        hypothesis.reproducibility = "reproduced"
        _replace_finding(session, finding)
        return finding
    if outcome is ReproductionOutcome.NOT_REPRODUCED:
        hypothesis.status = HypothesisStatus.WEAKENED
        hypothesis.reproducibility = "not_reproduced"
        return finding
    if outcome is ReproductionOutcome.BLOCKED:
        hypothesis.reproducibility = "blocked"
        return finding
    hypothesis.reproducibility = "inconclusive"
    return finding


def require_not_verify(kind: str) -> None:
    if kind in {"verify_finding", "mark_verified"}:
        raise RestrictedActivityError("verify_finding")


def _existing(session: Any, hypothesis: ResearchHypothesis) -> SecurityFinding | None:
    fingerprint = finding_fingerprint(
        vulnerability_class=hypothesis.vulnerability_class,
        target=hypothesis.target,
        endpoint=hypothesis.target,
    )
    for finding in session.findings:
        other = finding_fingerprint(
            vulnerability_class=finding.vulnerability_class or "",
            target=finding.target or "",
            endpoint=finding.endpoint or finding.target or "",
        )
        if isinstance(finding, SecurityFinding) and other == fingerprint:
            return finding
    return None


def _replace_finding(session: Any, finding: SecurityFinding) -> None:
    session.findings = [item if item.id != finding.id else finding for item in session.findings]


def _stamp_research_evidence(finding: SecurityFinding, items: list[Evidence]) -> list[Evidence]:
    """Runtime research records become server observations. Static records stay static."""
    stamped: list[Evidence] = []
    for item in items:
        if item.kind not in _RESEARCH_OBSERVATION_KINDS:
            stamped.append(item)
            continue
        stamped.append(
            issue_for_finding(finding, item, f"research-{item.kind.value}-{item.summary}")
        )
    return stamped


def _evidence_from_graph(session: Any, hypothesis: ResearchHypothesis) -> list[Evidence]:
    items: list[Evidence] = []
    for evidence_id in hypothesis.supporting_evidence_ids:
        node = session.graph.nodes.get(evidence_id)
        if node is None:
            continue
        if str(getattr(node, "provenance", "") or "") in {"replay", "ai_hypothesis"}:
            continue
        kind = _kind_from_node(node.kind)
        items.append(
            Evidence(
                kind=kind,
                source=node.source or "research-agent",
                summary=node.summary or "research observation",
            )
        )
    return items


def _kind_from_node(kind: str) -> EvidenceKind:
    mapping = {
        "source": EvidenceKind.SOURCE_CODE,
        "request": EvidenceKind.HTTP_REQUEST,
        "response": EvidenceKind.HTTP_RESPONSE,
        "scanner_result": EvidenceKind.SCANNER,
        "browser_observation": EvidenceKind.BROWSER,
        "reproduction": EvidenceKind.REPRODUCTION,
        "observation": EvidenceKind.LOG,
    }
    if kind in mapping:
        return mapping[kind]
    try:
        return EvidenceKind(kind)
    except ValueError:
        return EvidenceKind.LOG
