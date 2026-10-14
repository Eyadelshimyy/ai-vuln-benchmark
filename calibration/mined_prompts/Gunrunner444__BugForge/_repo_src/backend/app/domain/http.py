"""Vendor-neutral HTTP evidence types for proxy and browser adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4


@dataclass(frozen=True)
class HttpHeader:
    name: str
    value: str


@dataclass(frozen=True)
class HttpBodyMeta:
    """Metadata about a request/response body without always storing raw bytes."""

    content_type: str | None = None
    byte_length: int = 0
    truncated: bool = False
    sha256: str | None = None


@dataclass(frozen=True)
class HttpExchange:
    """One captured request/response pair.

    Bodies may be truncated by collectors. This type is evidence, not an
    invitation to replay requests against live targets.
    """

    method: str
    url: str
    request_id: str = ""
    timestamp: datetime | None = None
    scheme: str | None = None
    host: str | None = None
    port: int | None = None
    path: str | None = None
    request_headers: tuple[HttpHeader, ...] = ()
    request_body: str | None = None
    request_body_meta: HttpBodyMeta | None = None
    query: str | None = None
    cookies: tuple[str, ...] = ()
    response_status: int | None = None
    response_headers: tuple[HttpHeader, ...] = ()
    response_body: str | None = None
    response_body_meta: HttpBodyMeta | None = None
    elapsed_ms: float | None = None
    source_tool: str | None = None
    scope_decision: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id:
            object.__setattr__(self, "request_id", uuid4().hex)
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now(UTC))
        parsed = urlparse(self.url)
        if not self.scheme:
            object.__setattr__(self, "scheme", (parsed.scheme or "").lower() or None)
        if not self.host:
            host = (parsed.hostname or "").lower().rstrip(".")
            object.__setattr__(self, "host", host or None)
        if self.port is None:
            object.__setattr__(self, "port", parsed.port)
        if not self.path:
            object.__setattr__(self, "path", parsed.path or "/")
        if self.query is None and parsed.query:
            object.__setattr__(self, "query", parsed.query)

    @property
    def endpoint_key(self) -> str:
        host = self.host or ""
        path = self.path or "/"
        return f"{self.method.upper()} {host}{path}"
