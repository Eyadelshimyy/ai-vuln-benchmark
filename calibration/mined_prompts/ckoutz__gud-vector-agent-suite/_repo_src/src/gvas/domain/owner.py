"""The business owner's web dashboard: its sign-in records, the calendar it
shows and the settings an owner may change from it.

Owner credentials live apart from customer portal credentials: an owner
session can never authenticate a customer route and a customer session can
never authenticate an owner route. Both are bound to one business.
"""

import ipaddress
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gvas.domain.identifiers import BusinessId

OWNER_LOGIN_TOKEN_TTL = timedelta(minutes=15)
OWNER_SESSION_TTL = timedelta(days=30)
CALENDAR_FEED_URL_MAX_CHARS = 2048
CALENDAR_WINDOW_MAX_DAYS = 62
DISPLAY_NAME_MAX_CHARS = 255


class OwnerModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("owner timestamps must be timezone-aware")
    return value


class OwnerLoginToken(OwnerModel):
    """One single-use owner magic link; only the digest is stored."""

    token_hash: str = Field(min_length=64, max_length=64)
    business_id: BusinessId
    email: str = Field(min_length=3, max_length=320)
    expires_at: datetime
    used_at: datetime | None = None
    created_at: datetime

    _aware_expires = field_validator("expires_at")(_aware)

    def is_usable(self, now: datetime) -> bool:
        return self.used_at is None and now < self.expires_at


class OwnerSession(OwnerModel):
    """A bearer credential for the owner of one business."""

    token_hash: str = Field(min_length=64, max_length=64)
    business_id: BusinessId
    email: str = Field(min_length=3, max_length=320)
    expires_at: datetime
    revoked_at: datetime | None = None
    created_at: datetime

    _aware_expires = field_validator("expires_at")(_aware)

    def is_active(self, now: datetime) -> bool:
        return self.revoked_at is None and now < self.expires_at


class OwnerLoginTokenRepository(Protocol):
    async def add(self, token: OwnerLoginToken) -> None: ...

    async def find_by_hash(self, token_hash: str) -> OwnerLoginToken | None: ...

    async def mark_used(self, token_hash: str, now: datetime) -> bool:
        """``True`` exactly once per token."""
        ...


class OwnerSessionRepository(Protocol):
    async def add(self, session: OwnerSession) -> None: ...

    async def find_by_hash(self, token_hash: str) -> OwnerSession | None: ...

    async def revoke(self, token_hash: str, now: datetime) -> bool: ...


class CalendarEventSource(StrEnum):
    """Where a dashboard calendar entry came from."""

    #: A Calendly booking (an estimate or appointment customers scheduled).
    BOOKING = "booking"
    #: The owner's own calendar (Google, Apple, Outlook, ...) via its feed.
    CALENDAR = "calendar"
    #: A website booking request still waiting for the owner's decision.
    REQUEST = "request"


class CalendarEvent(OwnerModel):
    source: CalendarEventSource
    title: str
    start: datetime
    end: datetime | None = None
    all_day: bool = False
    location: str | None = None
    invitee_name: str | None = None
    invitee_email: str | None = None
    reference: str | None = None

    _aware_start = field_validator("start")(_aware)


class OwnerCalendarError(RuntimeError):
    """A calendar source failed; the message is safe to show the owner."""


def normalize_booking_link(value: str) -> str:
    """A booking link customers follow: absolute https (http only for a local
    host); a path is allowed."""

    candidate = value.strip()
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
        parts.port  # noqa: B018 - property access raises on a malformed port
    except ValueError as error:
        raise ValueError("is not a parseable URL") from error
    if parts.scheme.lower() not in ("http", "https") or not host:
        raise ValueError("must be an absolute http(s) URL")
    if parts.scheme.lower() == "http" and not _is_local(host):
        raise ValueError("must use https outside local development")
    return candidate


def normalize_calendar_feed_url(value: str) -> str:
    """The owner's private calendar subscription link (an iCal/.ics URL).

    ``webcal://`` — what Apple and Outlook hand out — is the same feed over
    https. Only public https hosts are accepted so the server can never be
    pointed at itself or a private network.
    """

    candidate = value.strip()
    if len(candidate) > CALENDAR_FEED_URL_MAX_CHARS:
        raise ValueError(f"must be at most {CALENDAR_FEED_URL_MAX_CHARS} characters")
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
        port = parts.port
    except ValueError as error:
        raise ValueError("is not a parseable URL") from error
    scheme = parts.scheme.lower()
    if scheme == "webcal":
        scheme = "https"
    if scheme != "https" or not host:
        raise ValueError("must be an https:// or webcal:// calendar link")
    if parts.username or parts.password:
        raise ValueError("must not carry a username or password")
    if port not in (None, 443):
        raise ValueError("must use the standard https port")
    if not is_public_host(host):
        raise ValueError("must point at a public calendar service")
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, ""))


def calendar_feed_host(url: str | None) -> str | None:
    """What the dashboard may show about a stored feed: its host only. The
    rest of a private calendar link is a credential."""

    if not url:
        return None
    return urlsplit(url).hostname


def is_public_host(host: str) -> bool:
    """A literal address must be globally routable; a name must not be a
    local one. Names are re-checked after resolution by the feed adapter."""

    name = host.strip().rstrip(".").lower()
    if not name or _is_local(name):
        return False
    try:
        address = ipaddress.ip_address(name.strip("[]"))
    except ValueError:
        return "." in name
    return address.is_global


def _is_local(host: str) -> bool:
    name = host.strip().rstrip(".").lower()
    return name == "localhost" or name.endswith(".localhost") or name.endswith(".local")


__all__ = [
    "CALENDAR_FEED_URL_MAX_CHARS",
    "CALENDAR_WINDOW_MAX_DAYS",
    "DISPLAY_NAME_MAX_CHARS",
    "OWNER_LOGIN_TOKEN_TTL",
    "OWNER_SESSION_TTL",
    "CalendarEvent",
    "CalendarEventSource",
    "OwnerCalendarError",
    "OwnerLoginToken",
    "OwnerLoginTokenRepository",
    "OwnerSession",
    "OwnerSessionRepository",
    "calendar_feed_host",
    "is_public_host",
    "normalize_booking_link",
    "normalize_calendar_feed_url",
]
