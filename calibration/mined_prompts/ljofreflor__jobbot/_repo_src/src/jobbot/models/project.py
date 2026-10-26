"""Personal project model."""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl, TypeAdapter, field_validator

_HTTP_URL = TypeAdapter(HttpUrl)


class Project(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    url: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("url", mode="before")
    @classmethod
    def strip_empty(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not (value.startswith("http://") or value.startswith("https://")):
            msg = "URL must start with http:// or https://"
            raise ValueError(msg)
        _HTTP_URL.validate_python(value)
        return value
