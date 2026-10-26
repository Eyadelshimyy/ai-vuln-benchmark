"""Job posting domain model."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, HttpUrl


class JobPosting(BaseModel):
    id: str = Field(min_length=1)
    source: str = "manual"
    source_job_id: str | None = None
    url: str | None = None
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    location: str | None = None
    description: str = ""
    raw_description: str = ""
    requirements: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    seniority: str | None = None
    language_requirements: list[str] = Field(default_factory=list)
    employment_type: str | None = None
    remote_type: str | None = None
    ats_url: str | None = None
    ats_kind: str | None = None
    posted_at: datetime | None = None
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str | None = None
    match_score: float | None = None
    invalid_reason: str | None = None

    def validate_url(self) -> None:
        if self.url:
            HttpUrl(self.url)
        if self.ats_url and not self.ats_url.lower().startswith("mailto:"):
            HttpUrl(self.ats_url)
