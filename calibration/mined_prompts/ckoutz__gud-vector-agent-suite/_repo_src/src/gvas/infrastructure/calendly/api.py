"""Calendly API v2 adapter for the appointment lookup port.

Only this module talks HTTP to Calendly. It lists the configured user's active
scheduled events in the window (``GET /scheduled_events``) and then each event's
invitees (``GET /scheduled_events/{uuid}/invitees``). Failures surface as
``AppointmentLookupError`` with a fixed message: the token, request URLs and
response bodies never leave this module.
"""

import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from gvas.domain.appointments import Appointment, AppointmentLookupError, AppointmentWindow
from gvas.domain.identifiers import BusinessId
from gvas.infrastructure.calendly.config import (
    CalendlyInstallation,
    CalendlySettings,
    parse_calendly_installations,
)

logger = logging.getLogger(__name__)

CALENDLY_SOURCE_LABEL = "Calendly"
ADDRESS_LOCATION_TYPES = frozenset({"physical", "custom"})
ADDRESS_QUESTION_MARKER = "address"
PHONE_QUESTION_MARKER = "phone"


class CalendlyResponseModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class CalendlyEventLocation(CalendlyResponseModel):
    type: str | None = None
    location: str | None = None


class CalendlyScheduledEvent(CalendlyResponseModel):
    uri: str
    name: str | None = None
    status: str
    start_time: datetime
    location: CalendlyEventLocation | None = None


class CalendlyPagination(CalendlyResponseModel):
    next_page_token: str | None = None


class CalendlyScheduledEventsResponse(CalendlyResponseModel):
    collection: tuple[CalendlyScheduledEvent, ...]
    pagination: CalendlyPagination = CalendlyPagination()


class CalendlyQuestionAnswer(CalendlyResponseModel):
    question: str
    answer: str
    position: int


class CalendlyInvitee(CalendlyResponseModel):
    name: str
    email: str
    status: str
    timezone: str | None = None
    text_reminder_number: str | None = None
    questions_and_answers: tuple[CalendlyQuestionAnswer, ...] = ()


class CalendlyInviteesResponse(CalendlyResponseModel):
    collection: tuple[CalendlyInvitee, ...]
    pagination: CalendlyPagination = CalendlyPagination()


class CalendlyAppointmentLookup:
    """Implements ``AppointmentLookupPort`` for the businesses in ``installations``."""

    def __init__(
        self,
        settings: CalendlySettings,
        client: httpx.AsyncClient,
        installations: tuple[CalendlyInstallation, ...] | None = None,
    ) -> None:
        if not settings.token:
            raise AppointmentLookupError("calendly token is not configured")
        self._settings = settings
        self._client = client
        resolved = installations or parse_calendly_installations(settings.installations)
        self._users: dict[BusinessId, str] = {
            installation.business_id: installation.user_uri for installation in resolved
        }

    def serves(self, business_id: BusinessId) -> bool:
        return business_id in self._users

    async def find(self, window: AppointmentWindow) -> tuple[Appointment, ...]:
        user_uri = self._users.get(window.business_id)
        if user_uri is None:
            return ()
        appointments: list[Appointment] = []
        for event in await self._scheduled_events(user_uri, window):
            if event.status != "active":
                continue
            event_uuid = event.uri.rstrip("/").rsplit("/", 1)[-1]
            if not event_uuid:
                continue
            location_address = _location_address(event.location)
            for invitee in await self._invitees(event_uuid):
                if invitee.status != "active":
                    continue
                appointments.append(
                    Appointment(
                        appointment_id=f"calendly:{event_uuid}:{invitee.email.casefold()}",
                        start_time=_localize(event.start_time, invitee.timezone),
                        invitee_name=invitee.name,
                        invitee_email=invitee.email,
                        event_name=event.name or "Appointment",
                        invitee_phone=_invitee_phone(invitee),
                        address=location_address or _answered_address(invitee),
                        notes=booking_notes(invitee),
                        source_label=CALENDLY_SOURCE_LABEL,
                    )
                )
        return tuple(appointments)

    async def _scheduled_events(
        self, user_uri: str, window: AppointmentWindow
    ) -> tuple[CalendlyScheduledEvent, ...]:
        events: list[CalendlyScheduledEvent] = []
        page_token: str | None = None
        while True:
            params: dict[str, str | int] = {
                "user": user_uri,
                "status": "active",
                "min_start_time": _iso_utc(window.start),
                "max_start_time": _iso_utc(window.end),
                "count": self._settings.page_size,
            }
            if page_token is not None:
                params["page_token"] = page_token
            payload = await self._get("/scheduled_events", params)
            try:
                page = CalendlyScheduledEventsResponse.model_validate(payload)
            except ValidationError as error:
                raise _unreadable(error) from error
            events.extend(page.collection)
            page_token = page.pagination.next_page_token
            if page_token is None:
                return tuple(events)

    async def _invitees(self, event_uuid: str) -> tuple[CalendlyInvitee, ...]:
        invitees: list[CalendlyInvitee] = []
        page_token: str | None = None
        while True:
            params: dict[str, str | int] = {"count": self._settings.page_size}
            if page_token is not None:
                params["page_token"] = page_token
            payload = await self._get(f"/scheduled_events/{event_uuid}/invitees", params)
            try:
                page = CalendlyInviteesResponse.model_validate(payload)
            except ValidationError as error:
                raise _unreadable(error) from error
            invitees.extend(page.collection)
            page_token = page.pagination.next_page_token
            if page_token is None:
                return tuple(invitees)

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
            raise AppointmentLookupError("calendly was unreachable") from error
        if response.status_code >= 400:
            logger.warning("calendly returned http %s for %s", response.status_code, path)
            raise AppointmentLookupError(f"calendly returned http {response.status_code}")
        try:
            return response.json()
        except ValueError as error:
            raise _unreadable(error) from error


def _unreadable(error: Exception) -> AppointmentLookupError:
    logger.warning("calendly returned an unreadable response: %s", type(error).__name__)
    return AppointmentLookupError("calendly returned an unreadable response")


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def _location_address(location: CalendlyEventLocation | None) -> str | None:
    if location is None or location.type not in ADDRESS_LOCATION_TYPES:
        return None
    address = (location.location or "").strip()
    return address or None


def _answered_address(invitee: CalendlyInvitee) -> str | None:
    for entry in sorted(invitee.questions_and_answers, key=lambda item: item.position):
        if ADDRESS_QUESTION_MARKER in entry.question.casefold() and entry.answer.strip():
            return entry.answer.strip()
    return None


def _invitee_phone(invitee: CalendlyInvitee) -> str | None:
    """The SMS reminder number when the invitee opted in, else the answer to a
    question mentioning a phone; either way normalized to E.164 when possible."""

    candidates = [invitee.text_reminder_number or ""]
    candidates.extend(
        entry.answer
        for entry in sorted(invitee.questions_and_answers, key=lambda item: item.position)
        if PHONE_QUESTION_MARKER in entry.question.casefold()
    )
    for candidate in candidates:
        normalized = _e164(candidate)
        if normalized is not None:
            return normalized
    return None


def _e164(value: str) -> str | None:
    digits = "".join(character for character in value if character.isdigit())
    if not digits:
        return None
    if value.strip().startswith("+"):
        return f"+{digits}" if 7 <= len(digits) <= 15 else None
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return None


def booking_notes(invitee: CalendlyInvitee) -> tuple[str, ...]:
    return tuple(
        f"{entry.question.strip()}: {entry.answer.strip()}"
        for entry in sorted(invitee.questions_and_answers, key=lambda item: item.position)
        if entry.question.strip() and entry.answer.strip()
    )


def _localize(start_time: datetime, timezone_name: str | None) -> datetime:
    if timezone_name:
        try:
            return start_time.astimezone(ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError:
            pass
    return start_time
