"""Indeed apply handoff — open the right page and stop before submit."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from jobbot.adapters.ats.apply import describe_prefill, open_ats_in_browser
from jobbot.adapters.base import ApplicationPackage, ApplyMethod, PrefillResult
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting


class IndeedApplyAdapter:
    """Open Indeed Apply or the external ATS. Never submits."""

    name = "indeed"

    def detect_method(self, job: JobPosting) -> ApplyMethod:
        if apply_target(job):
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
        """CV attach stays in the Indeed or ATS UI."""
        _ = package

    def open(self, job: JobPosting) -> str | None:
        url = apply_target(job)
        if not url:
            return None
        open_ats_in_browser(url)
        return url


def apply_target(job: JobPosting) -> str | None:
    """External ATS when the posting leaves Indeed; otherwise Indeed Apply."""
    if job.ats_url and not _is_indeed_url(job.ats_url):
        return job.ats_url
    return _applystart(job)


def _applystart(job: JobPosting) -> str | None:
    jk = job.source_job_id
    host: str | None = None
    scheme = "https"
    for candidate in (job.url, job.ats_url):
        if not candidate or not _is_indeed_url(candidate):
            continue
        parsed = urlparse(candidate)
        host = parsed.hostname
        scheme = parsed.scheme or "https"
        if not jk:
            match = re.search(r"[?&]jk=([a-f0-9]+)", candidate, flags=re.I)
            if match:
                jk = match.group(1)
        break
    if not host or not jk:
        return None
    return f"{scheme}://{host}/applystart?jk={jk}"


def _is_indeed_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    return host == "indeed.com" or host.endswith(".indeed.com")
