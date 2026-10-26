"""Job source adapter protocol (Indeed, LinkedIn posts, GetOnBoard, …)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from jobbot.config import JobbotConfig
from jobbot.models.job import JobPosting

JobSourceName = Literal[
    "indeed", "linkedin_post", "getonboard", "torre", "empleos_publicos"
]


class JobSearchQuery(BaseModel):
    query: str
    location: str | None = None
    remote: bool = False
    limit: int = Field(default=30, ge=1, le=50)
    countries: tuple[str, ...] = ()
    allow_remote: bool = True
    copy_permalinks: bool = True
    max_age_days: int = 0  # 0 = keep every age; the CLI passes the configured default


class JobSourceAdapter(Protocol):
    def search_jobs(self, query: JobSearchQuery) -> list[JobPosting]: ...

    def get_job(self, job_id: str) -> JobPosting: ...


def get_job_source(
    name: JobSourceName,
    config: JobbotConfig,
    *,
    cdp_url: str | None = None,
    fixture: Path | None = None,
) -> JobSourceAdapter:
    """Factory for job discovery adapters."""
    if name == "indeed":
        from jobbot.adapters.indeed.jobs import IndeedJobSource

        return IndeedJobSource(config, cdp_url=cdp_url)
    if name == "linkedin_post":
        from jobbot.adapters.linkedin.posts_source import LinkedInPostJobSource

        return LinkedInPostJobSource(config, cdp_url=cdp_url)
    if name == "getonboard":
        from jobbot.adapters.getonboard.jobs import GetOnBoardJobSource

        return GetOnBoardJobSource(config)
    if name == "torre":
        from jobbot.adapters.torre.jobs import TorreJobSource

        return TorreJobSource(config)
    if name == "empleos_publicos":
        from jobbot.adapters.empleospublicos.jobs import EmpleosPublicosJobSource

        return EmpleosPublicosJobSource(config, fixture=fixture)
    msg = f"Unknown job source: {name}"
    raise ValueError(msg)
