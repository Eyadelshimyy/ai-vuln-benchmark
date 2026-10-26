"""Torre.ai job board portal detector."""

from __future__ import annotations

from dataclasses import dataclass

from jobbot.portals.detect import AtsKind


@dataclass
class PortalDetectionResult:
    """Result of portal detection."""

    is_job_portal: bool
    confidence: float
    evidence: str
    ats_kind: str | None


class TorrePortalDetector:
    """Detect Torre.ai as a job board portal.
    
    Torre is a LATAM/remote-first job board with a public API.
    """

    def detect(self, url: str, html: str | None = None) -> PortalDetectionResult:
        """Detect if URL is Torre.ai.
        
        Torre URLs:
        - https://torre.ai/
        - https://torre.co/
        - https://search.torre.co/
        """
        url_lower = url.lower()
        
        if "torre.ai" in url_lower or "torre.co" in url_lower:
            return PortalDetectionResult(
                is_job_portal=True,
                confidence=1.0,
                evidence="Torre.ai job board (LATAM/remote)",
                ats_kind=AtsKind.TORRE.value,
            )
        
        return PortalDetectionResult(
            is_job_portal=False,
            confidence=0.0,
            evidence="Not Torre.ai",
            ats_kind=None,
        )
