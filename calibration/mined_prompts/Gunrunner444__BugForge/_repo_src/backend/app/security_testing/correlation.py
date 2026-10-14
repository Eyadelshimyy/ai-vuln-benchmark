"""Correlate evidence from multiple tools. AI-only never becomes verified."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace

from app.domain.evidence import EvidenceBundle, EvidenceProvenance
from app.domain.findings import FindingStatus, SecurityFinding
from app.domain.security import EvidenceTier, VulnerabilityClass

_ID_SEGMENT = re.compile(r"/(\d+|[0-9a-fA-F-]{8,}|[0-9a-f]{32})(?=/|$)")


def canonicalize_endpoint(url: str) -> str:
    if not url:
        return ""
    path = url.split("?", 1)[0]
    return _ID_SEGMENT.sub("/{id}", path)


@dataclass(frozen=True)
class CorrelatedFinding:
    key: str
    findings: tuple[SecurityFinding, ...]
    evidence: EvidenceBundle
    independent_sources: int
    informational: bool

    @property
    def confidence(self) -> str:
        if self.informational:
            return "informational"
        if self.independent_sources >= 3:
            return "high"
        if self.independent_sources == 2:
            return "medium"
        return "low"


class FindingCorrelationEngine:
    """Deduplicate scanner noise and merge related observations."""

    def correlate(self, findings: list[SecurityFinding]) -> list[CorrelatedFinding]:
        grouped: dict[str, list[SecurityFinding]] = defaultdict(list)
        for finding in findings:
            grouped[self._key(finding)].append(finding)
        clusters: list[CorrelatedFinding] = []
        for key, items in grouped.items():
            bundle = EvidenceBundle()
            sources: set[str] = set()
            for finding in items:
                bundle = bundle.extend(finding.evidence.items)
                sources.update(finding.tools)
                for item in finding.evidence.items:
                    if item.provenance is not EvidenceProvenance.AI_HYPOTHESIS:
                        sources.add(item.source)
            independent = len(sources)
            informational = independent <= 1 and all(
                f.status is FindingStatus.POTENTIAL
                and f.evidence_tier in {EvidenceTier.AI_HYPOTHESIS, EvidenceTier.STATIC_INDICATOR}
                for f in items
            )
            clusters.append(
                CorrelatedFinding(
                    key=key,
                    findings=tuple(items),
                    evidence=bundle,
                    independent_sources=independent,
                    informational=informational,
                )
            )
        clusters.sort(key=lambda c: (-c.independent_sources, c.key))
        return clusters

    def merge_evidence(self, findings: list[SecurityFinding]) -> list[SecurityFinding]:
        clusters = self.correlate(findings)
        merged: list[SecurityFinding] = []
        for cluster in clusters:
            primary = cluster.findings[0]
            extra = [item for item in cluster.evidence.items if item not in primary.evidence.items]
            if extra:
                primary = replace(
                    primary,
                    evidence=primary.evidence.extend(extra),
                    confidence=(
                        cluster.confidence
                        if cluster.confidence != "informational"
                        else primary.confidence
                    ),
                    tools=tuple(dict.fromkeys(t for f in cluster.findings for t in f.tools)),
                    rule_ids=tuple(dict.fromkeys(r for f in cluster.findings for r in f.rule_ids)),
                    observation_refs=tuple(
                        dict.fromkeys(r for f in cluster.findings for r in f.observation_refs)
                    ),
                )
            merged.append(primary)
        return merged

    def _key(self, finding: SecurityFinding) -> str:
        endpoint = canonicalize_endpoint(finding.endpoint or finding.target or "")
        klass = finding.vulnerability_class or VulnerabilityClass.MISSING_SECURITY_CONTROL.value
        return f"{klass}|{endpoint}|{finding.title.lower().strip()}"


def ai_only(finding: SecurityFinding) -> bool:
    if not finding.evidence:
        return bool(finding.hypothesis or finding.ai_analysis)
    return not finding.evidence.verifying_items() and all(
        item.provenance is EvidenceProvenance.AI_HYPOTHESIS
        or item.provenance is EvidenceProvenance.STATIC_ANALYSIS
        or item.provenance is EvidenceProvenance.SOURCE_OBSERVATION
        for item in finding.evidence.items
    )
