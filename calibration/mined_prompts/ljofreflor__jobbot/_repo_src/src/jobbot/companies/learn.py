"""Turn URLs seen during normal use into candidate company knowledge."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jobbot.companies.discovery import career_root_url, classify_url
from jobbot.companies.models import CareerSiteType, DiscoverySource
from jobbot.companies.registry import (
    CompanyRegistry,
    ObserveOutcome,
    default_companies_path,
    load_companies,
    save_companies,
)
from jobbot.companies.urls import (
    PrivateRouteRejected,
    ReservedDomainRejected,
    company_hint_from_url,
)
from jobbot.config import JobbotConfig
from jobbot.models.job import JobPosting

logger = logging.getLogger("jobbot.companies.learn")

_SKIPPED_HOSTS = ("linkedin.com", "lnkd.in")


@dataclass(frozen=True)
class LearnResult:
    company_id: str
    url: str
    site_type: CareerSiteType
    ats: str
    created: bool
    merged: bool
    conflict: str | None = None


def learn_from_url(
    registry: CompanyRegistry,
    *,
    company: str,
    url: str,
    source: DiscoverySource,
    country: str | None = None,
    resolve: bool = False,
    notes: str | None = None,
) -> ObserveOutcome | None:
    """Classify a URL and record it as candidate knowledge. Boards are ignored."""
    classification = classify_url(url, resolve=resolve)
    if not classification.is_company_specific:
        return None
    root = career_root_url(classification.url, classification.ats)
    site_type = classification.site_type
    if site_type == CareerSiteType.JOB_POSTING:
        # We store the portal that lists openings, not the single vacancy.
        site_type = (
            CareerSiteType.ATS_INSTANCE
            if classification.ats.value != "unknown"
            else CareerSiteType.COMPANY_CAREER_PORTAL
        )
    return registry.observe(
        company=company,
        url=root,
        source=source,
        site_type=site_type,
        ats=classification.ats,
        evidence=classification.evidence,
        country=country,
        reached_from=(classification.requested_url if classification.redirects_to else None),
        notes=notes,
    )


def company_name_for_job(job: JobPosting, url: str) -> str | None:
    """Employer name safe to store: on post sweeps ``company`` is the recruiter."""
    hint = company_hint_from_url(url)
    if job.source == "linkedin_post":
        return hint or None
    return job.company or hint or None


def learn_from_job(
    config: JobbotConfig,
    job: JobPosting,
    *,
    source: DiscoverySource = DiscoverySource.JOB_SOURCE,
) -> LearnResult | None:
    """Best-effort learning from a stored job. Never breaks the calling flow."""
    url = job.ats_url or job.url
    if not url or not job.company:
        return None
    if any(host in url.casefold() for host in _SKIPPED_HOSTS):
        return None
    path = default_companies_path(config.root)
    try:
        company = company_name_for_job(job, url)
        if not company:
            return None
        registry = load_companies(path)
        outcome = learn_from_url(
            registry,
            company=company,
            url=url,
            source=source,
            notes=f"seen via {job.source}",
        )
    except ReservedDomainRejected as exc:
        logger.warning("Company knowledge skipped: %s", exc)
        return None
    except PrivateRouteRejected as exc:
        logger.debug("Not shareable company knowledge (%s): %s", url, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — discovery must not break the job flow
        logger.debug("Company learning skipped for %s: %s", url, exc)
        return None
    if outcome is None or outcome.site is None:
        return None
    save_companies(registry, path)
    return LearnResult(
        company_id=outcome.company.id,
        url=outcome.site.url,
        site_type=outcome.site.site_type,
        ats=outcome.site.ats.value,
        created=outcome.created_site,
        merged=outcome.merged,
        conflict=outcome.conflict,
    )
