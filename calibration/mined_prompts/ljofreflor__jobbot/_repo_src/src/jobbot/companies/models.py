"""Company ↔ career platform knowledge model (shareable; never candidate PII)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from jobbot.companies.urls import canonical_key
from jobbot.portals.detect import AtsKind


def utc_now() -> datetime:
    return datetime.now(UTC)


class CareerSiteType(StrEnum):
    """What a discovered URL actually is."""

    COMPANY_CAREER_PORTAL = "company_career_portal"
    ATS_INSTANCE = "ats_instance"
    JOB_POSTING = "job_posting"
    JOB_BOARD = "job_board"
    UNKNOWN = "unknown"


class KnowledgeStatus(StrEnum):
    """Discovery lifecycle: nothing becomes truth without an explicit promote."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    STALE = "stale"
    REJECTED = "rejected"


class DiscoverySource(StrEnum):
    MANUAL = "manual"
    USER_OBSERVATION = "user_observation"
    LINKEDIN_POST = "linkedin_post"
    JOB_SOURCE = "job_source"
    WEB_DISCOVERY = "web_discovery"
    OFFICIAL_SITE = "official_site"


class Observation(BaseModel):
    """One independent sighting of a career site. Kept even when they disagree."""

    source: DiscoverySource
    checked_at: datetime = Field(default_factory=utc_now)
    ats: AtsKind = AtsKind.UNKNOWN
    site_type: CareerSiteType = CareerSiteType.UNKNOWN
    evidence: str = ""
    observed_url: str | None = None


class CareerSite(BaseModel):
    url: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    site_type: CareerSiteType = CareerSiteType.UNKNOWN
    ats: AtsKind = AtsKind.UNKNOWN
    status: KnowledgeStatus = KnowledgeStatus.CANDIDATE
    first_seen: datetime = Field(default_factory=utc_now)
    last_verified: datetime | None = None
    # Public URL that redirected here (e.g. empresa.cl/careers → this ATS instance).
    reached_from: str | None = None
    observations: list[Observation] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    notes: str | None = None

    @property
    def key(self) -> str:
        return canonical_key(self.url)

    @property
    def confidence(self) -> int:
        """How many independent sources reported this site."""
        return len({obs.source for obs in self.observations})


class CompanyRecord(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    country: str | None = None
    sector: str | None = None
    domains: list[str] = Field(default_factory=list)
    career_sites: list[CareerSite] = Field(default_factory=list)

    def find_site(self, url: str) -> CareerSite | None:
        key = canonical_key(url)
        return next((site for site in self.career_sites if site.key == key), None)

    def sites_with_status(self, status: KnowledgeStatus) -> list[CareerSite]:
        return [site for site in self.career_sites if site.status == status]
