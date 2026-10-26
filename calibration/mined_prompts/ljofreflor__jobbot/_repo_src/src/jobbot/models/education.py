"""Education model."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Education(BaseModel):
    id: str = Field(min_length=1)
    institution: str = Field(min_length=1)
    degree: str = Field(min_length=1)
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    details: str | None = None

    @model_validator(mode="after")
    def check_dates(self) -> Education:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            msg = f"{self.id}: end_date is before start_date"
            raise ValueError(msg)
        return self
