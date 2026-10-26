"""Experience and achievement models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Achievement(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, int | float | str] = Field(default_factory=dict)


class Experience(BaseModel):
    id: str = Field(min_length=1)
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    location: str | None = None
    start_date: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    current: bool = False
    description: str | None = None
    achievements: list[Achievement] = Field(default_factory=list)

    @field_validator("end_date")
    @classmethod
    def empty_end_to_none(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        return value

    @model_validator(mode="after")
    def check_dates(self) -> Experience:
        if self.current and self.end_date is not None:
            msg = f"{self.id}: current experience must not set end_date"
            raise ValueError(msg)
        if not self.current and self.end_date is None:
            msg = f"{self.id}: non-current experience requires end_date"
            raise ValueError(msg)
        if self.end_date is not None and self.end_date < self.start_date:
            msg = f"{self.id}: end_date is before start_date"
            raise ValueError(msg)
        return self

    def format_date_range(self) -> str:
        start = self.start_date
        end = "Present" if self.current else (self.end_date or "")
        return f"{start} – {end}"


MetricValue = int | float | str


def format_metric(key: str, value: MetricValue) -> str:
    """Format a metric for display without changing its numeric value."""
    if isinstance(value, bool):
        return f"{key}: {value}"
    if isinstance(value, int | float):
        abs_val = abs(value)
        if abs_val >= 1_000_000_000:
            scaled = value / 1_000_000_000
            text = f"{scaled:g}B"
        elif abs_val >= 1_000_000:
            scaled = value / 1_000_000
            text = f"{scaled:g}M"
        elif abs_val >= 1_000:
            scaled = value / 1_000
            text = f"{scaled:g}K"
        else:
            text = f"{value:g}"
        if key.endswith("_percent") or key.endswith("_pct"):
            return f"{value:g}%"
        if "usd" in key.lower() or "dollar" in key.lower():
            return f"USD {text}"
        return f"{text} {key.replace('_', ' ')}"
    return f"{value}"


def metrics_as_display(metrics: dict[str, Any]) -> list[str]:
    return [format_metric(k, v) for k, v in metrics.items()]
