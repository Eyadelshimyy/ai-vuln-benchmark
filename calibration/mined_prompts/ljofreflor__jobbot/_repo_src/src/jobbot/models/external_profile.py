"""External profile snapshot for portal diffs."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ExternalExperience(BaseModel):
    company: str | None = None
    title: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: str | None = None


class ExternalEducation(BaseModel):
    institution: str | None = None
    degree: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class ExternalProfile(BaseModel):
    source: str
    headline: str | None = None
    summary: str | None = None
    location: str | None = None
    experience: list[ExternalExperience] = Field(default_factory=list)
    education: list[ExternalEducation] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
