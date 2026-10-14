"""Normalized API specification model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AuthType(StrEnum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    COOKIE = "cookie"


@dataclass(frozen=True)
class Parameter:
    name: str
    location: str  # path | query | header | cookie | body
    required: bool = False
    schema_type: str = "string"
    example: str | None = None


@dataclass(frozen=True)
class RequestSchema:
    content_type: str | None = None
    parameters: tuple[Parameter, ...] = ()
    body_example: str | None = None


@dataclass(frozen=True)
class ResponseSchema:
    status: int = 200
    content_type: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Authentication:
    auth_type: AuthType = AuthType.NONE
    name: str | None = None
    in_location: str | None = None


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    operation_id: str | None = None
    summary: str = ""
    parameters: tuple[Parameter, ...] = ()
    request: RequestSchema | None = None
    responses: tuple[ResponseSchema, ...] = ()
    authentication: Authentication = Authentication()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class APISpec:
    title: str
    version: str = "0.0.0"
    base_url: str = ""
    endpoints: tuple[Endpoint, ...] = ()
    authentication: Authentication = Authentication()
    source_format: str = "unknown"
