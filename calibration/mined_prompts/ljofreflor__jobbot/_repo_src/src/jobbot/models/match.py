"""Job match result models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class MatchStrength(StrEnum):
    STRONG = "strong_match"
    PARTIAL = "partial_match"
    MISSING = "missing"
    UNKNOWN = "unknown"


class MatchItem(BaseModel):
    label: str
    strength: MatchStrength
    detail: str | None = None


class JobMatch(BaseModel):
    job_id: str
    score: float = Field(ge=0, le=100)
    items: list[MatchItem] = Field(default_factory=list)

    def by_strength(self, strength: MatchStrength) -> list[MatchItem]:
        return [i for i in self.items if i.strength == strength]

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "score": self.score,
            "strong_match": [i.label for i in self.by_strength(MatchStrength.STRONG)],
            "partial_match": [i.label for i in self.by_strength(MatchStrength.PARTIAL)],
            "missing": [i.label for i in self.by_strength(MatchStrength.MISSING)],
            "unknown": [i.label for i in self.by_strength(MatchStrength.UNKNOWN)],
            "items": [i.model_dump() for i in self.items],
        }
