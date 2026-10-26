"""Dispatch ApplicationPortalAdapter by ATS kind."""

from __future__ import annotations

from jobbot.adapters.ats.ashby import AshbyAdapter
from jobbot.adapters.ats.getonboard import GetOnBoardAdapter
from jobbot.adapters.ats.greenhouse import GreenhouseAdapter
from jobbot.adapters.ats.indeed_apply import IndeedApplyAdapter
from jobbot.adapters.ats.lever import LeverAdapter
from jobbot.adapters.ats.workday import WorkdayAdapter
from jobbot.adapters.base import ApplicationPortalAdapter
from jobbot.adapters.torre.apply import TorreApplicationAdapter
from jobbot.models.job import JobPosting
from jobbot.portals.detect import AtsKind, detect_ats

_ADAPTERS: dict[AtsKind, ApplicationPortalAdapter] = {
    AtsKind.GREENHOUSE: GreenhouseAdapter(),
    AtsKind.LEVER: LeverAdapter(),
    AtsKind.ASHBY: AshbyAdapter(),
    AtsKind.GETONBOARD: GetOnBoardAdapter(),
    AtsKind.WORKDAY: WorkdayAdapter(),
    AtsKind.INDEED: IndeedApplyAdapter(),
    AtsKind.TORRE: TorreApplicationAdapter(),
}


def supported_kinds() -> frozenset[AtsKind]:
    """Platforms with an application adapter (prefill of known fields)."""
    return frozenset(_ADAPTERS)


def adapter_for_kind(kind: AtsKind) -> ApplicationPortalAdapter | None:
    return _ADAPTERS.get(kind)


def adapter_for_job(job: JobPosting) -> ApplicationPortalAdapter | None:
    if job.ats_kind:
        try:
            kind = AtsKind(job.ats_kind)
        except ValueError:
            kind = AtsKind.UNKNOWN
    else:
        kind = AtsKind.UNKNOWN
    if kind == AtsKind.UNKNOWN:
        for url in (job.ats_url, job.url):
            if url:
                kind = detect_ats(url)
                if kind != AtsKind.UNKNOWN:
                    break
    return adapter_for_kind(kind)
