"""Lever ATS adapter — open + known-field map only (HITL submit)."""

from __future__ import annotations

from jobbot.adapters.ats.apply import describe_prefill, open_ats_in_browser, resolve_ats_url
from jobbot.adapters.base import ApplicationPackage, ApplyMethod, PrefillResult
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind, detect_ats


class LeverAdapter:
    """Prefill helper for Lever-hosted applications."""

    name = "lever"

    def detect_method(self, job: JobPosting) -> ApplyMethod:
        url, kind = resolve_ats_url(job)
        if url and (kind == AtsKind.LEVER or detect_ats(url) == AtsKind.LEVER):
            return ApplyMethod.EXTERNAL_ATS
        return ApplyMethod.UNKNOWN

    def prefill(
        self,
        candidate: Candidate,
        job: JobPosting,
        package: ApplicationPackage,
    ) -> PrefillResult:
        _ = package
        return describe_prefill(candidate, job)

    def attach_cv(self, package: ApplicationPackage) -> None:
        _ = package

    def open(self, job: JobPosting) -> str | None:
        url, _kind = resolve_ats_url(job)
        if not url:
            return None
        open_ats_in_browser(url)
        return url
