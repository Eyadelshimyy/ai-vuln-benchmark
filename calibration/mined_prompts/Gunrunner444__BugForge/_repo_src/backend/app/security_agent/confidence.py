"""Transparent finding confidence, severity, impact, evidence, reproducibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.findings import SecurityFinding
from app.security_agent.schemas import ResearchHypothesis


@dataclass(frozen=True)
class FindingScore:
    confidence: str
    severity: str
    impact: str
    evidence_strength: int
    reproducibility: str
    total: int
    explanation: tuple[str, ...]

    def snapshot(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "severity": self.severity,
            "impact": self.impact,
            "evidence_strength": self.evidence_strength,
            "reproducibility": self.reproducibility,
            "total": self.total,
            "explanation": list(self.explanation),
        }


_CONF = {"high": 3, "medium": 2, "low": 1}
_SEV = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
_REPRO = {"reproduced": 3, "requires_reproduction": 1, "unknown": 0, "not_reproduced": 0}


def score_finding(
    finding: SecurityFinding | None = None,
    hypothesis: ResearchHypothesis | None = None,
) -> FindingScore:
    confidence = (finding.confidence if finding else None) or (
        hypothesis.confidence if hypothesis else "low"
    )
    severity = (
        (finding.impact if finding and finding.impact in _SEV else None)
        or (hypothesis.severity if hypothesis else "medium")
        or "medium"
    )
    impact = (finding.impact if finding else None) or (hypothesis.impact if hypothesis else "")
    evidence = 0
    if hypothesis is not None:
        evidence = hypothesis.evidence_strength or len(hypothesis.supporting_evidence_ids)
    elif finding is not None:
        evidence = len(finding.evidence.items)
    repro = (finding.reproducibility if finding else None) or (
        hypothesis.reproducibility if hypothesis else "unknown"
    )
    conf_n = _CONF.get(str(confidence).lower(), 1)
    sev_n = _SEV.get(str(severity).lower(), 2)
    repro_n = _REPRO.get(str(repro).lower(), 0)
    impact_n = 1 if str(impact).strip() else 0
    total = 8 * sev_n + 5 * evidence + 4 * repro_n + 2 * impact_n + conf_n
    explanation = (
        f"severity={severity} contributes {8 * sev_n}",
        f"evidence_strength={evidence} contributes {5 * evidence}",
        f"reproducibility={repro} contributes {4 * repro_n}",
        f"impact_present={bool(str(impact).strip())} contributes {2 * impact_n}",
        f"confidence={confidence} contributes {conf_n}",
        "AI suggestions never override this score",
    )
    return FindingScore(
        confidence=str(confidence),
        severity=str(severity),
        impact=str(impact),
        evidence_strength=evidence,
        reproducibility=str(repro or "unknown"),
        total=total,
        explanation=explanation,
    )
