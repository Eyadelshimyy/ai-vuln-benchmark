"""Calendly bookings for the owner dashboard's calendar.

Lists the configured user's active scheduled events in a window with their
invitees. Uses the same token and installations as the appointment lookup;
failures surface as one fixed ``OwnerCalendarError`` message.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from gvas.domain.identifiers import BusinessId
from gvas.domain.owner import CalendarEvent, CalendarEventSource, OwnerCalendarError
from gvas.infrastructure.calendly.api import (
    CalendlyEventLocation,
    CalendlyInviteesResponse,
    CalendlyPagination,
    _iso_utc,
    _location_address,
)
from gvas.infrastructure.calendly.config import (
    CalendlyInstallation,
    CalendlySettings,
    parse_calendly_installations,
)

logger = logging.getLogger(__name__)

UNAVAILABLE = "Calendly bookings couldn't be loaded right now."
MAX_EVENTS = 200
#: Calendly filters on start time; look back this far so a booking already
#: under way when the window opens is still returned.
BOOKING_LOOKBACK = timedelta(hours=24)
MAX_PAGES = 5
INVITEE_CONCURRENCY = 5


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _ScheduledEvent(_Model):
    uri: str
    name: str | None = None
    status: str
    start_time: datetime
    end_time: datetime | None = None
    location: CalendlyEventLocation | None = None


class _ScheduledEventsPage(_Model):
    collection: tuple[_ScheduledEvent, ...]
    pagination: CalendlyPagination = CalendlyPagination()


class CalendlyBookedEvents:
    """Implements ``BookedEventsPort`` for the businesses in ``installations``."""

    def __init__(
        self,
        settings: CalendlySettings,
        client: httpx.AsyncClient,
        installations: tuple[CalendlyInstallation, ...] | None = None,
    ) -> None:
        if not settings.token:
            raise OwnerCalendarError("calendly token is not configured")
        self._settings = settings
        self._client = client
        resolved = installations or parse_calendly_installations(settings.installations)
        self._users: dict[BusinessId, str] = {
            installation.business_id: installation.user_uri for installation in resolved
        }

    def serves(self, business_id: BusinessId) -> bool:
        return business_id in self._users

    async def upcoming(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[CalendarEvent, ...]:
        user_uri = self._users.get(business_id)
        if user_uri is None:
            return ()
        fetched = await self._events(user_uri, start - BOOKING_LOOKBACK, end)
        scheduled = [
            event
            for event in fetched
            if (event.end_time or event.start_time + timedelta(minutes=1)) > start
        ][:MAX_EVENTS]
        gate = asyncio.Semaphore(INVITEE_CONCURRENCY)

        async def project(event: _ScheduledEvent) -> CalendarEvent:
            async with gate:
                invitees = await self._invitees(event)
            active = [invitee for invitee in invitees.collection if invitee.status == "active"]
            first = active[0] if active else None
            return CalendarEvent(
                source=CalendarEventSource.BOOKING,
                title=event.name or "Appointment",
                start=event.start_time,
                end=event.end_time,
                location=_location_address(event.location),
                invitee_name=first.name if first is not None else None,
                invitee_email=first.email if first is not None else None,
            )

        projected = await asyncio.gather(
            *(project(event) for event in scheduled if event.status == "active")
        )
        return tuple(sorted(projected, key=lambda event: event.start))

    async def _events(self, user_uri: str, start: datetime, end: datetime) -> list[_ScheduledEvent]:
        events: list[_ScheduledEvent] = []
        page_token: str | None = None
        for _ in range(MAX_PAGES):
            params: dict[str, str | int] = {
                "user": user_uri,
                "status": "active",
                "min_start_time": _iso_utc(start),
                "max_start_time": _iso_utc(end),
                "sort": "start_time:asc",
                "count": self._settings.page_size,
            }
            if page_token is not None:
                params["page_token"] = page_token
            try:
                page = _ScheduledEventsPage.model_validate(
                    await self._get("/scheduled_events", params)
                )
            except ValidationError as error:
                raise _unreadable(error) from error
            events.extend(page.collection)
            page_token = page.pagination.next_page_token
            if page_token is None or len(events) >= MAX_EVENTS:
                break
        return events

    async def _invitees(self, event: _ScheduledEvent) -> CalendlyInviteesResponse:
        event_uuid = event.uri.rstrip("/").rsplit("/", 1)[-1]
        if not event_uuid:
            return CalendlyInviteesResponse(collection=())
        payload = await self._get(f"/scheduled_events/{event_uuid}/invitees", {"count": 10})
        try:
            return CalendlyInviteesResponse.model_validate(payload)
        except ValidationError as error:
            raise _unreadable(error) from error

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
            raise OwnerCalendarError(UNAVAILABLE) from error
        if response.status_code >= 400:
            logger.warning("calendly returned http %s for owner calendar", response.status_code)
            raise OwnerCalendarError(UNAVAILABLE)
        try:
            return response.json()
        except ValueError as error:
            raise _unreadable(error) from error


def _unreadable(error: Exception) -> OwnerCalendarError:
    logger.warning("calendly returned an unreadable response: %s", type(error).__name__)
    return OwnerCalendarError(UNAVAILABLE)


__all__ = ["CalendlyBookedEvents"]
