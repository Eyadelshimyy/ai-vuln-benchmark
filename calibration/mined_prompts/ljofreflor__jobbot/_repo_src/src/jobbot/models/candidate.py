"""Candidate domain model — single source of truth representation."""

from __future__ import annotations

import re

from pydantic import BaseModel, EmailStr, Field, HttpUrl, TypeAdapter, field_validator

from jobbot.models.education import Education
from jobbot.models.experience import Experience
from jobbot.models.project import Project
from jobbot.models.skill import Publication, SkillGroups

_HTTP_URL = TypeAdapter(HttpUrl)
_ORCID_SHAPE = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]")


def orcid_is_valid(value: str) -> bool:
    """ORCID iD shape plus its ISO 7064 11,2 check digit."""
    if not _ORCID_SHAPE.fullmatch(value):
        return False
    digits = value.replace("-", "")
    total = 0
    for char in digits[:-1]:
        total = (total + int(char)) * 2
    remainder = (12 - total % 11) % 11
    return digits[-1] == ("X" if remainder == 10 else str(remainder))


class PersonalInfo(BaseModel):
    name: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    city: str | None = None
    country: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    orcid: str | None = None

    @field_validator("linkedin", "github", "orcid", mode="before")
    @classmethod
    def strip_empty(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("linkedin", "github")
    @classmethod
    def validate_url_like(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not (value.startswith("http://") or value.startswith("https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        _HTTP_URL.validate_python(value)
        return value

    @field_validator("orcid")
    @classmethod
    def validate_orcid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not orcid_is_valid(value):
            msg = "ORCID must be the bare iD 0000-0000-0000-000X with a valid check digit"
            raise ValueError(msg)
        return value

    @property
    def orcid_url(self) -> str | None:
        return f"https://orcid.org/{self.orcid}" if self.orcid else None

    def location_line(self) -> str | None:
        parts = [p for p in (self.city, self.country) if p]
        return ", ".join(parts) if parts else None


class Candidate(BaseModel):
    personal: PersonalInfo
    summary: str | None = None
    specialties: list[str] = Field(default_factory=list)
    # Job-board searches the candidate keeps: a preference, not a professional fact.
    search_queries: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: SkillGroups = Field(default_factory=SkillGroups)
    publications: list[Publication] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)

    @field_validator("search_queries", mode="before")
    @classmethod
    def clean_queries(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            text = " ".join(str(item or "").split())
            if text and text.casefold() not in seen:
                seen.add(text.casefold())
                cleaned.append(text)
        return cleaned

    def achievement_count(self) -> int:
        return sum(len(exp.achievements) for exp in self.experience)
