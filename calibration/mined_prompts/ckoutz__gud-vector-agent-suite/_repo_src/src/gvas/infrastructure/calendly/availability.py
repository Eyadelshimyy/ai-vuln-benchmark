"""Calendly availability + booking for the intake chat.

Reads the business's configured Calendly user: the first active event type is
the bookable kind, ``GET /event_type_available_times`` reports real openings,
and an approval books via ``POST /invitees`` — the direct-invitee Scheduling
API. When that call is refused (the Scheduling API needs a paid Calendly
plan) the adapter falls back to a single-use scheduling link
(``POST /scheduling_links``, ``max_event_count=1``) prefilled with the
customer's name, email and chosen date. Errors surface as
``AvailabilityError`` with a fixed message — the token and provider payloads
never leave this module.
"""

import logging
import time
from datetime import date, datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gvas.domain.calendar_blocks import MINUTES_PER_DAY, DayHours, Interval
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import (
    DEFAULT_SLOT_MINUTES,
    AvailabilityError,
    AvailableSlot,
    BookingKind,
    BookingRequest,
    BookingResult,
)
from gvas.infrastructure.calendly.api import _iso_utc, _localize
from gvas.infrastructure.calendly.config import (
    CalendlyInstallation,
    CalendlySettings,
    parse_calendly_installations,
)

logger = logging.getLogger(__name__)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _EventType(_Model):
    uri: str
    duration: int | None = None


class _EventTypesResponse(_Model):
    collection: tuple[_EventType, ...]


class _User(_Model):
    timezone: str | None = None


class _UserResponse(_Model):
    resource: _User | None = None


class _Pagination(_Model):
    next_page_token: str | None = None


class _AvailableTime(_Model):
    status: str
    start_time: str


class _AvailableTimesResponse(_Model):
    collection: tuple[_AvailableTime, ...]
    pagination: _Pagination = _Pagination()


class _ScheduledEvent(_Model):
    status: str
    start_time: str
    event_type: str | None = None


class _ScheduledEventsResponse(_Model):
    collection: tuple[_ScheduledEvent, ...]


class _SchedulingLink(_Model):
    booking_url: str


class _SchedulingLinkResponse(_Model):
    resource: _SchedulingLink | None = None


class _Interval(_Model):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    start: str = Field(alias="from")
    end: str = Field(alias="to")


class _Rule(_Model):
    type: str
    wday: str | None = None
    date: str | None = None
    intervals: list[_Interval] = Field(default_factory=list)


class _ScheduleRule(_Model):
    timezone: str
    rules: list[_Rule] = Field(default_factory=list)


class _EventTypeSchedule(_Model):
    availability_rule: _ScheduleRule


class _EventTypeSchedulesResponse(_Model):
    collection: list[_EventTypeSchedule]


class CalendlyAvailability:
    """Implements ``AvailabilityPort`` for businesses in ``installations``."""

    def __init__(
        self,
        settings: CalendlySettings,
        client: httpx.AsyncClient,
        installations: tuple[CalendlyInstallation, ...] | None = None,
    ) -> None:
        if not settings.token:
            raise AvailabilityError("calendly token is not configured")
        self._settings = settings
        self._client = client
        resolved = installations or parse_calendly_installations(settings.installations)
        self._users: dict[BusinessId, str] = {
            installation.business_id: installation.user_uri for installation in resolved
        }
        # Resolved lazily per business and refreshed after a TTL so a changed
        # or deleted event type does not stay cached until restart. One dict
        # publishes the event type, its timezone and the timestamp together —
        # a mid-refresh reader never sees a new type with a stale timezone.
        self._event_types: dict[BusinessId, tuple[_EventType, str | None, float]] = {}

    def serves(self, business_id: BusinessId) -> bool:
        return business_id in self._users

    async def available_slots(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[AvailableSlot, ...]:
        spec = await self._event_type(business_id)
        if spec is None:
            return ()
        event_type, timezone, minutes = spec
        openings: list[AvailableSlot] = []
        for entry in await self._available_times(event_type, start, end):
            if entry.status != "available":
                continue
            try:
                slot_start = _parse_calendly_time(entry.start_time)
            except ValueError:
                continue
            local = _localize(slot_start, timezone)
            openings.append(AvailableSlot(start=local, end=local + timedelta(minutes=minutes)))
        return tuple(openings)

    async def find_booking(self, request: BookingRequest) -> BookingResult | None:
        """Reconciliation after a crashed attempt: the booking is an active
        invitee event for this email at the exact requested start on the
        business's event type — anything looser could claim an unrelated
        appointment the same customer happens to hold."""

        user_uri = self._users.get(request.business_id)
        if user_uri is None:
            raise AvailabilityError("no calendly event type is configured")
        # A pinned type wins: a retried attempt must match the event the
        # original call created, even if the first active type has changed.
        event_type_uri = request.event_type_uri
        if event_type_uri is None:
            spec = await self._event_type(request.business_id)
            if spec is None:
                raise AvailabilityError("no calendly event type is configured")
            event_type_uri = spec[0]
        payload = await self._get(
            "/scheduled_events",
            {
                "user": user_uri,
                "invitee_email": request.invitee_email,
                "min_start_time": _iso_utc(request.slot_start),
                "max_start_time": _iso_utc(request.slot_end),
                "status": "active",
            },
        )
        try:
            events = _ScheduledEventsResponse.model_validate(payload)
        except ValidationError as error:
            raise _unreadable(error) from error
        for event in events.collection:
            if event.status != "active":
                continue
            if event.event_type is not None and event.event_type != event_type_uri:
                continue
            try:
                start = _parse_calendly_time(event.start_time)
            except ValueError:
                continue
            if start == request.slot_start:
                return BookingResult(kind=BookingKind.BOOKED, event_type_uri=event_type_uri)
        return None

    async def cancel_booking(self, business_id: BusinessId, event_uri: str) -> None:
        if business_id not in self._users:
            raise AvailabilityError("no calendly event type is configured")
        base = self._settings.api_base_url.rstrip("/")
        if not event_uri.startswith(f"{base}/"):
            raise AvailabilityError("unexpected calendly event uri")
        path = event_uri[len(base) :]
        try:
            response = await self._client.post(
                f"{base}{path}/cancellation",
                json={"reason": "the owner declined this booking"},
                headers={"Authorization": f"Bearer {self._settings.token}"},
                timeout=self._settings.api_timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("calendly cancellation failed: %s", type(error).__name__)
            raise AvailabilityError("calendly was unreachable") from error
        if response.status_code == 404:
            # Already canceled or never existed — the desired end state.
            return
        if response.status_code >= 400:
            logger.warning("calendly cancellation returned http %s", response.status_code)
            raise AvailabilityError(f"calendly returned http {response.status_code}")

    async def book(self, request: BookingRequest) -> BookingResult:
        spec = await self._event_type(request.business_id)
        if spec is None:
            raise AvailabilityError("no calendly event type is configured")
        _resolved_type, timezone, _minutes = spec
        event_type = request.event_type_uri or _resolved_type
        try:
            await self._create_invitee(event_type, request, timezone)
            return BookingResult(kind=BookingKind.BOOKED, event_type_uri=event_type)
        except _DirectBookingRejectedError:
            link = await self._scheduling_link(event_type)
            return BookingResult(
                kind=BookingKind.LINK,
                link=_prefilled_link(link, request, timezone),
                event_type_uri=event_type,
            )

    async def booking_event_type_uri(self, business_id: BusinessId) -> str | None:
        spec = await self._event_type(business_id)
        if spec is None:
            raise AvailabilityError("no calendly event type is configured")
        return spec[0]

    async def _event_type(self, business_id: BusinessId) -> tuple[str, str | None, int] | None:
        user_uri = self._users.get(business_id)
        if user_uri is None:
            return None
        cached = self._event_types.get(business_id)
        if cached is None or time.monotonic() - cached[2] >= EVENT_TYPE_TTL_SECONDS:
            payload = await self._get("/event_types", {"user": user_uri, "active": "true"})
            try:
                listed = _EventTypesResponse.model_validate(payload)
            except ValidationError as error:
                raise _unreadable(error) from error
            if not listed.collection:
                raise AvailabilityError("calendly has no active event type")
            timezone = await self._user_timezone(user_uri)
            self._event_types[business_id] = (
                listed.collection[0],
                timezone,
                time.monotonic(),
            )
        event_type, timezone, _resolved_at = self._event_types[business_id]
        duration = event_type.duration or DEFAULT_SLOT_MINUTES
        return event_type.uri, timezone, duration

    async def _user_timezone(self, user_uri: str) -> str | None:
        path = user_uri.split(self._settings.api_base_url, 1)[-1]
        if not path.startswith("/users/"):
            return None
        try:
            payload = await self._get(path, {})
            user = _UserResponse.model_validate(payload)
        except (AvailabilityError, ValidationError):
            return None
        timezone = user.resource.timezone if user.resource else None
        if timezone:
            try:
                ZoneInfo(timezone)
            except ZoneInfoNotFoundError:
                return None
        return timezone

    async def _available_times(
        self, event_type_uri: str, start: datetime, end: datetime
    ) -> tuple[_AvailableTime, ...]:
        entries: list[_AvailableTime] = []
        page_token: str | None = None
        while True:
            params: dict[str, str | int] = {
                "event_type": event_type_uri,
                "start_time": _iso_utc(start),
                "end_time": _iso_utc(end),
                "count": self._settings.page_size,
            }
            if page_token is not None:
                params["page_token"] = page_token
            payload = await self._get("/event_type_available_times", params)
            try:
                page = _AvailableTimesResponse.model_validate(payload)
            except ValidationError as error:
                raise _unreadable(error) from error
            entries.extend(page.collection)
            page_token = page.pagination.next_page_token
            if page_token is None:
                return tuple(entries)

    async def _create_invitee(
        self, event_type_uri: str, request: BookingRequest, timezone: str | None
    ) -> None:
        invitee: dict[str, object] = {
            "name": request.invitee_name,
            "email": request.invitee_email,
        }
        if timezone:
            invitee["timezone"] = timezone
        if request.invitee_phone:
            invitee["text_reminder_number"] = request.invitee_phone
        body: dict[str, object] = {
            "event_type": event_type_uri,
            "start_time": _iso_utc(request.slot_start),
            "invitee": invitee,
        }
        if request.address:
            body["location"] = {"kind": "physical", "location": request.address}
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url}/invitees",
                json=body,
                headers={"Authorization": f"Bearer {self._settings.token}"},
                timeout=self._settings.api_timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("calendly invitee request failed: %s", type(error).__name__)
            raise AvailabilityError("calendly was unreachable") from error
        if response.status_code >= 400:
            if response.status_code < 500:
                # The Scheduling API is a paid-plan feature; client errors mean
                # "use a scheduling link", not "this booking is impossible".
                raise _DirectBookingRejectedError()
            logger.warning("calendly invitee returned http %s", response.status_code)
            raise AvailabilityError(f"calendly returned http {response.status_code}")

    async def _scheduling_link(self, event_type_uri: str) -> str:
        payload = await self._post(
            "/scheduling_links",
            {
                "owner": event_type_uri,
                "owner_type": "EventType",
                "max_event_count": 1,
            },
        )
        try:
            link = _SchedulingLinkResponse.model_validate(payload)
        except ValidationError as error:
            raise _unreadable(error) from error
        url = link.resource.booking_url if link.resource else ""
        if not url:
            raise _unreadable(ValueError("empty scheduling link"))
        return url

    async def schedule_timezone(self, business_id: BusinessId) -> str | None:
        _uri, schedule = await self._schedule(business_id)
        return schedule.timezone

    async def day_hours(self, business_id: BusinessId, day: date) -> DayHours:
        _uri, schedule = await self._schedule(business_id)
        for rule in schedule.rules:
            if rule.type == "date" and rule.date == day.isoformat():
                return DayHours(intervals=_rule_intervals(rule), overridden=True)
        weekday = day.strftime("%A").lower()
        for rule in schedule.rules:
            if rule.type == "wday" and rule.wday == weekday:
                return DayHours(intervals=_rule_intervals(rule), overridden=False)
        return DayHours(intervals=(), overridden=False)

    async def set_day_hours(
        self, business_id: BusinessId, day: date, intervals: tuple[Interval, ...] | None
    ) -> None:
        """Rewrites the booking event type's rules with ``day`` pinned.

        Calendly replaces every rule on PATCH, so the current set is read and
        written back with only this date's rule changed."""

        event_type_uri, schedule = await self._schedule(business_id)
        rules: list[dict[str, object]] = [
            _rule_payload(rule)
            for rule in schedule.rules
            if not (rule.type == "date" and rule.date == day.isoformat())
        ]
        if intervals is not None:
            rules.append(
                {
                    "type": "date",
                    "date": day.isoformat(),
                    "intervals": [
                        {"from": _clock(low), "to": _clock(high)} for low, high in intervals
                    ],
                }
            )
        await self._patch(
            "/event_type_availability_schedules",
            {"event_type": event_type_uri},
            {"availability_rule": {"timezone": schedule.timezone, "rules": rules}},
        )

    async def _schedule(self, business_id: BusinessId) -> tuple[str, _ScheduleRule]:
        spec = await self._event_type(business_id)
        if spec is None:
            raise AvailabilityError("no calendly event type is configured")
        payload = await self._get("/event_type_availability_schedules", {"event_type": spec[0]})
        try:
            listed = _EventTypeSchedulesResponse.model_validate(payload)
        except ValidationError as error:
            raise _unreadable(error) from error
        if not listed.collection:
            raise _unreadable(ValueError("event type has no availability schedule"))
        return spec[0], listed.collection[0].availability_rule

    async def _patch(
        self, path: str, params: dict[str, str | int], body: dict[str, object]
    ) -> None:
        try:
            response = await self._client.patch(
                f"{self._settings.api_base_url}{path}",
                params=params,
                json=body,
                headers={"Authorization": f"Bearer {self._settings.token}"},
                timeout=self._settings.api_timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("calendly request failed: %s", type(error).__name__)
            raise AvailabilityError("calendly was unreachable") from error
        if response.status_code >= 400:
            logger.warning("calendly returned http %s for %s", response.status_code, path)
            raise AvailabilityError(f"calendly returned http {response.status_code}")

    async def _get(self, path: str, params: dict[str, str | int]) -> object:
        try:
            response = await self._client.get(
                f"{self._settings.api_base_url}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._settings.token}"},
                timeout=self._settings.api_timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("calendly request failed: %s", type(error).__name__)
            raise AvailabilityError("calendly was unreachable") from error
        if response.status_code >= 400:
            logger.warning("calendly returned http %s for %s", response.status_code, path)
            raise AvailabilityError(f"calendly returned http {response.status_code}")
        try:
            return response.json()
        except ValueError as error:
            raise _unreadable(error) from error

    async def _post(self, path: str, body: dict[str, object]) -> object:
        try:
            response = await self._client.post(
                f"{self._settings.api_base_url}{path}",
                json=body,
                headers={"Authorization": f"Bearer {self._settings.token}"},
                timeout=self._settings.api_timeout_seconds,
            )
        except httpx.HTTPError as error:
            logger.warning("calendly request failed: %s", type(error).__name__)
            raise AvailabilityError("calendly was unreachable") from error
        if response.status_code >= 400:
            logger.warning("calendly returned http %s for %s", response.status_code, path)
            raise AvailabilityError(f"calendly returned http {response.status_code}")
        try:
            return response.json()
        except ValueError as error:
            raise _unreadable(error) from error


EVENT_TYPE_TTL_SECONDS = 3600.0


class _DirectBookingRejectedError(Exception):
    """``POST /invitees`` was refused — fall back to a scheduling link."""


def _unreadable(error: Exception) -> AvailabilityError:
    logger.warning("calendly returned an unreadable response: %s", type(error).__name__)
    return AvailabilityError("calendly returned an unreadable response")


def _parse_calendly_time(value: str) -> datetime:
    # ``start_time`` arrives ISO-8601 with a ``Z`` suffix.
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _prefilled_link(booking_url: str, request: BookingRequest, timezone: str | None) -> str:
    """Prefill name/email and preselect the chosen day on the link."""

    start = _localize(request.slot_start, timezone)
    params = {
        "name": request.invitee_name,
        "email": request.invitee_email,
        "month": start.strftime("%Y-%m"),
        "date": start.strftime("%Y-%m-%d"),
    }
    if request.reference:
        # Calendly records the link's utm parameters on the invitee and echoes
        # them back in the webhook payload's tracking block — the reference
        # binds the confirmed event to this exact request.
        params["utm_content"] = request.reference
    separator = "&" if "?" in booking_url else "?"
    return f"{booking_url}{separator}{urlencode(params)}"


def _minutes(clock: str) -> int:
    hours, _, minutes = clock.partition(":")
    return int(hours) * 60 + int(minutes or 0)


def _clock(minute: int) -> str:
    """Calendly intervals end by ``23:59`` at the latest."""

    minute = min(minute, MINUTES_PER_DAY - 1)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _rule_intervals(rule: _Rule) -> tuple[Interval, ...]:
    return tuple(
        (_minutes(interval.start), min(_minutes(interval.end), MINUTES_PER_DAY))
        for interval in rule.intervals
    )


def _rule_payload(rule: _Rule) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": rule.type,
        "intervals": [{"from": item.start, "to": item.end} for item in rule.intervals],
    }
    if rule.wday is not None:
        payload["wday"] = rule.wday
    if rule.date is not None:
        payload["date"] = rule.date
    return payload
