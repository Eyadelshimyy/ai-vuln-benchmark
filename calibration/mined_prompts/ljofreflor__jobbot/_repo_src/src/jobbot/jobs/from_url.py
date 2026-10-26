"""Ingest a hard job URL: know the portal, fetch the JD, store, match, prepare."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from jobbot.adapters.getonboard.jobs import job_from_hard_link, remember_portal
from jobbot.applications.manager import (
    ApplicationRepository,
    prepare_application_package,
)
from jobbot.config import JobbotConfig
from jobbot.cv.build import build_adapted_cv
from jobbot.jobs.career_page import (
    CareerPageParseError,
    ClosedPostingError,
    job_from_career_html,
)
from jobbot.jobs.repository import JobRepository, write_job_json
from jobbot.matching.analyzer import RuleBasedJobAnalyzer
from jobbot.models.application import ApplicationStatus
from jobbot.models.candidate import Candidate
from jobbot.models.job import JobPosting
from jobbot.models.match import JobMatch
from jobbot.portals.detect import AtsKind
from jobbot.portals.knowledge import PortalKnowledge, lookup_portal

# Re-export so CLI/tests catch filled career fixtures without a second import.
__all__ = [
    "ClosedPostingError",
    "GetFromUrlResult",
    "UnknownPortalError",
    "UnsupportedPortalFetchError",
    "ingest_hard_link",
]


class UnknownPortalError(ValueError):
    """The URL's host is not in local portals, the seed, or built-in ATS rules."""


class UnsupportedPortalFetchError(ValueError):
    """Portal is known but JobBot cannot fetch a JD from that host yet."""


@dataclass(frozen=True)
class GetFromUrlResult:
    job: JobPosting
    portal: PortalKnowledge
    match: JobMatch | None
    package_dir: Path | None
    job_json: Path


def ingest_hard_link(
    config: JobbotConfig,
    session: Session,
    url: str,
    *,
    candidate: Candidate,
    html: str | None = None,
    build: bool = True,
    prepare: bool = True,
    cdp_url: str | None = None,
    on_chunk: Callable[[int, int | None], None] | None = None,
) -> GetFromUrlResult:
    """Know the portal → fetch → store → match → CV + application package.

    Does not open a browser or submit. Callers use ``application apply --apply``
    (or the CLI ``get --apply``) for HITL send.

    Unknown hosts need ``html`` (a saved career page). Without a fixture they
    stay refused. Known Get on Board and Indeed URLs can fetch live; Indeed
    also accepts a saved viewjob ``html``. Other known ATS hosts still need a
    fetcher or a parseable career fixture.
    """
    portal = lookup_portal(config, url)
    if not portal.known and html is None:
        domain = portal.domain or "(no host)"
        msg = (
            f"Unknown employment domain: {domain}. "
            "Pass --fixture with saved career HTML, or add the host with "
            "`jobbot portals add` / `jobbot companies learn`."
        )
        raise UnknownPortalError(msg)

    job = _fetch_job(portal, html=html, cdp_url=cdp_url, on_chunk=on_chunk)
    if portal.known:
        remember_portal(
            config,
            url=portal.url,
            ats_kind=portal.ats_kind,
            notes=f"hard link ({portal.source.value})",
        )

    repo = JobRepository(session)
    stored = repo.upsert_external(job)
    job_json = write_job_json(stored, config.output_dir)

    match: JobMatch | None = None
    package_dir: Path | None = None
    if build or prepare:
        match = RuleBasedJobAnalyzer().analyze(candidate, stored)
        repo.update_match_score(stored.id, match.score)
        stored.match_score = match.score
        if build:
            _build_adapted_cv(config, candidate, stored, match)
        if prepare:
            job_dir = config.output_dir / "jobs" / stored.id
            package_dir = prepare_application_package(
                stored, config.output_dir, job_dir=job_dir, candidate=candidate
            )
            ApplicationRepository(session).upsert_for_job(
                stored.id,
                status=ApplicationStatus.PREPARED,
                package_dir=str(package_dir),
            )

    return GetFromUrlResult(
        job=stored,
        portal=portal,
        match=match,
        package_dir=package_dir,
        job_json=job_json,
    )


def _fetch_job(
    portal: PortalKnowledge,
    *,
    html: str | None,
    cdp_url: str | None = None,
    on_chunk: Callable[[int, int | None], None] | None = None,
) -> JobPosting:
    if portal.ats_kind == AtsKind.GETONBOARD:
        return job_from_hard_link(portal.url, html=html, on_chunk=on_chunk)
    if portal.ats_kind == AtsKind.INDEED:
        return _fetch_indeed_job(portal, html=html, cdp_url=cdp_url)
    if portal.ats_kind == AtsKind.EMPLEOS_PUBLICOS:
        return _fetch_empleos_publicos_job(portal, html=html)
    if html is not None:
        try:
            return job_from_career_html(html, url=portal.url)
        except ClosedPostingError:
            raise
        except CareerPageParseError as exc:
            if portal.known:
                msg = (
                    f"No hard-link fetcher for {portal.ats_kind.value} yet "
                    f"({portal.domain}). Fixture HTML was not a career job page."
                )
                raise UnsupportedPortalFetchError(msg) from exc
            raise UnknownPortalError(
                f"Fixture HTML is not a recognizable career job page "
                f"({portal.domain or portal.url})."
            ) from exc
    msg = (
        f"No hard-link fetcher for {portal.ats_kind.value} yet "
        f"({portal.domain}). Use `jobbot jobs add --file`, a Get on Board URL, "
        "or pass --fixture with saved career HTML."
    )
    raise UnsupportedPortalFetchError(msg)


def _fetch_empleos_publicos_job(
    portal: PortalKnowledge,
    *,
    html: str | None,
) -> JobPosting:
    """Parse an Empleos Públicos ficha (fixture-first; live board often 403)."""
    from jobbot.adapters.empleospublicos.jobs import (
        EmpleosPublicosParseError,
        canonical_ficha_url,
        job_from_ficha_html,
    )

    try:
        canonical = canonical_ficha_url(portal.url)
    except EmpleosPublicosParseError as exc:
        raise UnsupportedPortalFetchError(str(exc)) from exc
    if html is None:
        msg = (
            "Empleos Públicos ficha needs --fixture with saved HTML "
            "(live portal is JS-loaded and often refuses automated fetch)."
        )
        raise UnsupportedPortalFetchError(msg)
    try:
        job = job_from_ficha_html(html, url=canonical)
    except EmpleosPublicosParseError as exc:
        raise UnsupportedPortalFetchError(str(exc)) from exc
    job.url = canonical
    return job


def _fetch_indeed_job(
    portal: PortalKnowledge,
    *,
    html: str | None,
    cdp_url: str | None = None,
) -> JobPosting:
    """Parse an Indeed viewjob (fixture or live). Tracking params are stripped."""
    from jobbot.adapters.indeed.jobs import (
        IndeedJobClosed,
        IndeedJobSource,
        card_to_job_posting,
    )
    from jobbot.jobs.indeed_url import (
        IndeedUrlError,
        canonical_indeed_job_url,
        extract_indeed_jk,
    )

    try:
        canonical = canonical_indeed_job_url(portal.url)
        jk = extract_indeed_jk(canonical)
    except IndeedUrlError as exc:
        raise UnsupportedPortalFetchError(str(exc)) from exc

    card = {
        "source_job_id": jk,
        "url": canonical,
        "title": "",
        "company": "",
        "location": None,
        "snippet": "",
    }
    try:
        if html is not None:
            job = card_to_job_posting(card, detail_html=html, placeholder_id="TMP")
        else:
            job = IndeedJobSource(cdp_url=cdp_url).get_job(jk)
    except IndeedJobClosed as exc:
        raise ClosedPostingError(str(exc)) from exc
    job.url = canonical
    job.source_job_id = jk
    return job


def _build_adapted_cv(
    config: JobbotConfig,
    candidate: Candidate,
    job: JobPosting,
    match: JobMatch,
) -> None:
    build_adapted_cv(config, candidate, job, match)
