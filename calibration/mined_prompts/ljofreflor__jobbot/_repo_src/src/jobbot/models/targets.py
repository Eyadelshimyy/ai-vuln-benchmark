"""Shared types for target renderers and constraints."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field


class ProfileTarget(StrEnum):
    CV = "cv"
    ATS = "ats"
    INDEED = "indeed"
    LINKEDIN = "linkedin"


class TargetConstraints(BaseModel):
    headline_max: int | None = None
    summary_max: int | None = None
    experience_description_max: int | None = None


class ExperienceRenderer(Protocol):
    """Render an experience for a given target without changing facts."""

    def render(self, experience: object, target: ProfileTarget) -> str: ...


DEFAULT_CONSTRAINTS: dict[ProfileTarget, TargetConstraints] = {
    ProfileTarget.CV: TargetConstraints(),
    ProfileTarget.ATS: TargetConstraints(),
    ProfileTarget.INDEED: TargetConstraints(
        headline_max=120,
        summary_max=4000,
        experience_description_max=4000,
    ),
    ProfileTarget.LINKEDIN: TargetConstraints(
        headline_max=220,
        summary_max=2600,
        experience_description_max=2000,
    ),
}


class YearMonth(BaseModel):
    """Validated YYYY-MM date used in profile YAML."""

    value: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")

    def __str__(self) -> str:
        return self.value

    @property
    def year(self) -> int:
        return int(self.value[:4])

    @property
    def month(self) -> int:
        return int(self.value[5:7])

    def as_tuple(self) -> tuple[int, int]:
        return self.year, self.month
