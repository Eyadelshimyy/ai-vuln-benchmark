"""Workday ATS adapter stub — open + known-field map only (HITL submit)."""

from __future__ import annotations

from jobbot.adapters.ats.apply import describe_prefill, open_ats_in_browser, resolve_ats_url
from jobbot.adapters.base import ApplicationPackage, ApplyMethod, PrefillResult
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind, detect_ats


class WorkdayAdapter:
    """Prefill helper for Workday-hosted applications (HITL; no auto-submit)."""

    name = "workday"

    def detect_method(self, job: JobPosting) -> ApplyMethod:
        url, kind = resolve_ats_url(job)
        if url and (kind == AtsKind.WORKDAY or detect_ats(url) == AtsKind.WORKDAY):
            return ApplyMethod.EXTERNAL_ATS
        return ApplyMethod.UNKNOWN

    def prefill(
        self,
        candidate: Candidate,
        job: JobPosting,
        package: ApplicationPackage,
    ) -> PrefillResult:
        _ = package
        result = describe_prefill(candidate, job)
        review = list(result.needs_review)
        review.append("Workday multi-step forms — review each page before Submit")
        return PrefillResult(filled=result.filled, needs_review=review)

    def attach_cv(self, package: ApplicationPackage) -> None:
        _ = package

    def open(self, job: JobPosting) -> str | None:
        url, _kind = resolve_ats_url(job)
        if not url:
            return None
        open_ats_in_browser(url)
        return url
