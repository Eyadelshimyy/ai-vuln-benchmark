"""Torre application adapter - delegates to external ATS or manual Torre profile."""

from __future__ import annotations

from jobbot.adapters.base import ApplicationPackage, ApplyMethod, PrefillResult
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind


class TorreApplicationAdapter:
    """Apply to Torre opportunities.
    
    Torre is a job board that often redirects to external ATS (Greenhouse, Workday, etc).
    This adapter detects the destination and delegates appropriately.
    """

    name = "torre"

    def detect_method(self, job: JobPosting) -> ApplyMethod:
        """Detect how to apply to a Torre opportunity.
        
        Torre opportunities can be:
        1. External ATS redirect (Greenhouse, Workday, etc) → EXTERNAL_ATS
        2. Direct Torre application → UNKNOWN (manual via Torre profile)
        """
        # If the job's ATS is not Torre, it's an external redirect
        if job.ats_kind and job.ats_kind != AtsKind.TORRE.value:
            return ApplyMethod.EXTERNAL_ATS
        
        # Direct Torre applications are manual (no auto-apply)
        return ApplyMethod.UNKNOWN

    def prefill(
        self,
        candidate: Candidate,
        job: JobPosting,
        package: ApplicationPackage,
    ) -> PrefillResult:
        """Prefill Torre application or detect external ATS.
        
        If Torre redirects to external ATS, detection happens here.
        Otherwise, Torre applications are manual via profile.
        """
        # Torre opportunities with external redirects
        if job.ats_kind and job.ats_kind != AtsKind.TORRE.value:
            # Delegate to the appropriate adapter (handled by registry)
            return PrefillResult(
                filled=[],
                needs_review=[
                    f"Torre redirects to {job.ats_kind}",
                    f"Open URL: {job.ats_url}",
                ],
            )
        
        # Direct Torre applications
        return PrefillResult(
            filled=[],
            needs_review=[
                "Torre application via profile",
                "Open Torre opportunity URL",
                "Complete application manually in Torre",
                f"URL: {job.url}",
            ],
        )

    def attach_cv(self, package: ApplicationPackage) -> None:
        """Torre doesn't support direct CV attachment via adapter.
        
        CV must be uploaded manually or through Torre's profile system.
        """
        # No-op: Torre CV attachment is manual
        pass
