"""Correlate independent evidence. Never treat shared vuln-class as proof."""

from __future__ import annotations

from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

from app.domain.evidence import VERIFICATION_PROVENANCE, EvidenceProvenance
from app.security_agent.schemas import ResearchHypothesis
from app.security_agent.states import HypothesisStatus

_INDEPENDENT = {
    EvidenceProvenance.STATIC_ANALYSIS.value,
    EvidenceProvenance.HTTP_OBSERVATION.value,
    EvidenceProvenance.BROWSER_OBSERVATION.value,
    EvidenceProvenance.SCANNER_RESULT.value,
    EvidenceProvenance.SCANNER_OBSERVATION.value,
    EvidenceProvenance.SOURCE_OBSERVATION.value,
    EvidenceProvenance.REPRODUCTION.value,
    EvidenceProvenance.API_TEST.value,
    EvidenceProvenance.FUZZING_RESULT.value,
    EvidenceProvenance.EXECUTION.value,
}

_AI_ONLY = {
    EvidenceProvenance.AI_HYPOTHESIS.value,
}

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
_REPRO_RANK = {"reproduced": 3, "requires_reproduction": 1, "unknown": 0, "not_reproduced": 0}


def independent_provenances(hypothesis: ResearchHypothesis, graph: Any) -> set[str]:
    found: set[str] = set()
    for evidence_id in hypothesis.supporting_evidence_ids:
        node = graph.nodes.get(evidence_id)
        if node is None:
            continue
        provenance = str(node.provenance)
        if provenance in _INDEPENDENT or provenance in {
            item.value for item in VERIFICATION_PROVENANCE
        }:
            found.add(provenance)
    return found


def correlate_hypothesis(hypothesis: ResearchHypothesis, graph: Any) -> HypothesisStatus:
    supporting = independent_provenances(hypothesis, graph)
    contradicting = set()
    for evidence_id in hypothesis.contradicting_evidence_ids:
        node = graph.nodes.get(evidence_id)
        if node is not None:
            contradicting.add(str(node.provenance))
    hypothesis.evidence_strength = len(supporting)
    if contradicting and supporting:
        hypothesis.status = HypothesisStatus.WEAKENED
        return hypothesis.status
    if contradicting and not supporting:
        hypothesis.status = HypothesisStatus.DISPROVED
        return hypothesis.status
    if len(supporting) >= 2:
        hypothesis.status = HypothesisStatus.REQUIRES_REPRODUCTION
        return hypothesis.status
    if len(supporting) == 1:
        hypothesis.status = HypothesisStatus.SUPPORTED
        return hypothesis.status
    # Two AI hypotheses with the same class are not independent evidence.
    return (
        HypothesisStatus.OPEN if hypothesis.status is HypothesisStatus.OPEN else hypothesis.status
    )


def correlate_all(hypotheses: list[ResearchHypothesis], graph: Any) -> list[ResearchHypothesis]:
    for item in hypotheses:
        correlate_hypothesis(item, graph)
    return hypotheses


def prioritize(hypotheses: list[ResearchHypothesis]) -> list[ResearchHypothesis]:
    """Deterministic ranking. Confidence is not severity."""

    def score(item: ResearchHypothesis) -> tuple[int, int, int, int, int]:
        severity = _SEVERITY_RANK.get((item.severity or "").lower(), 0)
        evidence = item.evidence_strength or len(item.supporting_evidence_ids)
        repro = _REPRO_RANK.get((item.reproducibility or "").lower(), 0)
        if item.status is HypothesisStatus.REQUIRES_REPRODUCTION:
            repro = max(repro, 1)
        if item.status is HypothesisStatus.SUPPORTED:
            repro = max(repro, 0)
        confidence = _CONFIDENCE_RANK.get((item.confidence or "").lower(), 0)
        impact = 1 if (item.impact or "").strip() else 0
        # Weighted: 8*severity + 5*evidence + 4*repro + 2*impact + 1*confidence
        total = 8 * severity + 5 * evidence + 4 * repro + 2 * impact + confidence
        return (total, severity, evidence, repro, confidence)

    return sorted(hypotheses, key=score, reverse=True)


def finding_fingerprint(
    *,
    vulnerability_class: str,
    target: str,
    endpoint: str = "",
    source_location: str = "",
    evidence_keys: tuple[str, ...] = (),
) -> str:
    parsed = urlparse(target)
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or "/"
    resource = endpoint or f"{host}{path}"
    material = "|".join(
        [
            (vulnerability_class or "").strip().lower(),
            host,
            resource.strip().lower(),
            (source_location or "").strip().lower(),
            ",".join(sorted(evidence_keys)),
        ]
    )
    return sha256(material.encode("utf-8")).hexdigest()


def stable_finding_identity(
    *,
    vulnerability_class: str,
    target: str,
    endpoint: str = "",
    source_location: str = "",
    asset: str = "",
) -> str:
    """Scanner-independent identity for the same underlying issue."""
    return finding_fingerprint(
        vulnerability_class=vulnerability_class,
        target=target,
        endpoint=endpoint or asset,
        source_location=source_location,
    )


def propose_merge(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic merge *proposal*. Never auto-merges findings."""
    left_id = stable_finding_identity(
        vulnerability_class=str(left.get("vulnerability_class") or ""),
        target=str(left.get("target") or ""),
        endpoint=str(left.get("endpoint") or ""),
        source_location=str(left.get("source_location") or ""),
        asset=str(left.get("asset") or ""),
    )
    right_id = stable_finding_identity(
        vulnerability_class=str(right.get("vulnerability_class") or ""),
        target=str(right.get("target") or ""),
        endpoint=str(right.get("endpoint") or ""),
        source_location=str(right.get("source_location") or ""),
        asset=str(right.get("asset") or ""),
    )
    same = left_id == right_id
    reasons: list[str] = []
    if (
        str(left.get("vulnerability_class") or "").lower()
        == str(right.get("vulnerability_class") or "").lower()
    ):
        reasons.append("same_vulnerability_class")
    if str(left.get("endpoint") or left.get("target")) == str(
        right.get("endpoint") or right.get("target")
    ):
        reasons.append("same_endpoint")
    if str(left.get("source_location") or "") and str(left.get("source_location")) == str(
        right.get("source_location") or ""
    ):
        reasons.append("same_source_location")
    if str(left.get("asset") or "") and str(left.get("asset")) == str(right.get("asset") or ""):
        reasons.append("same_hackerone_asset")
    return {
        "same_identity": same,
        "left_id": left_id,
        "right_id": right_id,
        "reasons": reasons,
        "auto_merged": False,
    }


def cross_session_matches(
    current: list[dict[str, Any]],
    previous: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for left in current:
        for right in previous:
            proposal = propose_merge(left, right)
            if proposal["same_identity"] or proposal["reasons"]:
                proposals.append(proposal)
    return proposals
