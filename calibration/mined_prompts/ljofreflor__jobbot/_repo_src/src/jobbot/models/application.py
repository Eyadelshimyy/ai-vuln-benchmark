"""Application tracking models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ApplicationStatus(StrEnum):
    DISCOVERED = "discovered"
    SHORTLISTED = "shortlisted"
    PREPARED = "prepared"
    APPLIED = "applied"
    SCREENING = "screening"
    INTERVIEW = "interview"
    TECHNICAL_INTERVIEW = "technical_interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class Application(BaseModel):
    id: str
    job_id: str
    status: ApplicationStatus = ApplicationStatus.DISCOVERED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    package_dir: str | None = None


class ApplicationEvent(BaseModel):
    id: str
    application_id: str
    event_type: str
    detail: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
