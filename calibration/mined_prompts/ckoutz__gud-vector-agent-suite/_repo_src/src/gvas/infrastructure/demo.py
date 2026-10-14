"""Demo-deployment adapters: nothing leaves the process.

``GVAS_DEMO_MODE`` swaps every outbound channel (customer and owner e-mail,
texts, owner channel messages, sign-in links, reports) for one that writes to the log, and the
booking calendar for generated openings. A fictional business then runs the
real workflows, owner approval included, with no one contacted. Links are
withheld from the log: they carry bearer tokens (quote claims, sign-in and
decision links), and the demo is driven from the dashboard instead. The
production composition decides when these are used, and refuses to start a
demo that still holds a real provider credential.
"""

import hashlib
import logging
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.config import DemoSettings
from gvas.domain.customers import PortalLoginEmailRequest
from gvas.domain.enums import DeliveryStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import (
    AvailabilityError,
    AvailableSlot,
    BookingKind,
    BookingRequest,
    BookingResult,
    IntakeState,
    SupersededBooking,
)
from gvas.domain.messages import (
    AttachmentPayload,
    AttachmentReference,
    ConversationRef,
    CustomerDeliveryRequest,
    CustomerTextRequest,
    DeliveryReceipt,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.owner import CalendarEvent, CalendarEventSource
from gvas.domain.owner_email import OwnerEmailRequest
from gvas.domain.reporting import ReportEmailRequest
from gvas.infrastructure.hosted_links import PORTAL_LOGIN_LINK_REFERENCE
from gvas.infrastructure.intake_models import IntakeConversation as IntakeRow
from gvas.infrastructure.models import Business

logger = logging.getLogger(__name__)

DEMO_EVENT_TYPE_URI = "demo:estimate"
DEMO_DETAIL = "demo mode: logged, not sent"
# About one generated hour in this many is shown as already taken, so the
# openings Gus offers look like a real week rather than an empty calendar.
BUSY_ONE_IN = 3
# Sunday is closed.
CLOSED_WEEKDAYS = frozenset({6})
# Requests waiting for the owner and approved bookings hold their slot.
HOLDING_STATES = (IntakeState.AWAITING_OWNER.value, IntakeState.APPROVED.value)
TITLE_MAX_CHARS = 80

Span = tuple[datetime, datetime]


class DemoModeError(RuntimeError):
    """Something a demo deployment cannot do because it has no provider."""


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _receipt(
    key: str, *, customer_link: str | None = None, emailed: bool | None = None
) -> DeliveryReceipt:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return DeliveryReceipt(
        status=DeliveryStatus.DELIVERED,
        provider_message_id=f"demo-{digest}",
        occurred_at=datetime.now(UTC),
        detail=DEMO_DETAIL,
        customer_link=customer_link,
        emailed=emailed,
    )


def _log(kind: str, business_id: BusinessId, to: str, text: str) -> None:
    """Only that something was not sent: demo visitors may type real names,
    addresses and numbers, and logs outlive the sandbox that held them."""

    del to, text
    logger.info("demo mode, not sent: %s (business %s)", kind, business_id)


class LoggedCustomerEmail:
    """Customer e-mail (quotes, booking notices), logged instead of sent.

    Hosted link references resolve as the Resend adapter resolves them.
    """

    def __init__(self, portal_url: str) -> None:
        self._portal_url = portal_url

    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
        lines = [f"Subject: {request.subject or ''}", request.body_text]
        if request.quote_url:
            lines.append(f"Quote: {request.quote_url}")
        lines.extend(self._resolve_link(reference) for reference in request.links)
        _log("customer e-mail", request.business_id, request.recipient.address, "\n".join(lines))
        return _receipt(
            request.idempotency_key,
            customer_link=request.quote_url,
            emailed=request.recipient.email_address is not None,
        )

    def _resolve_link(self, reference: str) -> str:
        if reference == PORTAL_LOGIN_LINK_REFERENCE:
            return self._portal_url
        raise DemoModeError("quote carries an unknown hosted link reference")


class LoggedCustomerText:
    async def send_text(self, request: CustomerTextRequest) -> DeliveryReceipt:
        _log("text", request.business_id, request.phone_number, request.text)
        return _receipt(request.idempotency_key)


class LoggedOwnerEmail:
    async def send(self, request: OwnerEmailRequest) -> DeliveryReceipt:
        _log(
            "owner e-mail",
            request.business_id,
            request.to,
            f"Subject: {request.subject}\n{request.text}",
        )
        return _receipt(request.idempotency_key)


class LoggedOwnerReply:
    """Owner channel messages (chat or text threads), logged instead of posted."""

    async def send(
        self, conversation_ref: ConversationRef, message: OutboundOwnerMessage
    ) -> DeliveryReceipt:
        text = "\n".join(part.text for part in message.parts if isinstance(part, TextPart))
        _log(
            "owner message",
            message.business_id,
            conversation_ref.external_conversation_id,
            text or "(attachment)",
        )
        return _receipt(message.correlation_id)


class LoggedPortalLoginEmail:
    """The sign-in request is logged; its link (a bearer token) never is."""

    async def send_login_link(self, request: PortalLoginEmailRequest) -> DeliveryReceipt:
        _log("sign-in e-mail", request.business_id, request.to, f"Subject: {request.subject}")
        return _receipt(request.idempotency_key)


class LoggedReportEmail:
    async def deliver(self, request: ReportEmailRequest) -> DeliveryReceipt:
        _log(
            "report e-mail",
            request.business_id,
            request.recipient_address,
            f"Subject: {request.subject}\n{request.body_text}",
        )
        return _receipt(request.idempotency_key)


class NoAttachments:
    """A demo has no owner chat workspace, so there are no shared files to fetch."""

    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
        raise DemoModeError("demo mode has no shared files to fetch")


class DemoAvailability:
    """Generated openings in the business's zone, and bookings kept in GVAS.

    Working days (Monday to Saturday) offer one slot per ``slot_minutes``
    inside the configured hours. About one in three is shown as taken, picked
    by a stable hash so the same hour stays taken across calls, and any slot a
    waiting request, an approved booking or a booking a pending reschedule
    would replace holds is never offered again. ``book`` touches no calendar:
    the approved request is the booking. When another booking already took
    the slot it answers as Calendly does, with a link to pick again.
    """

    def __init__(
        self, settings: DemoSettings, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory

    def serves(self, business_id: BusinessId) -> bool:
        return True

    async def available_slots(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[AvailableSlot, ...]:
        async with self._session_factory() as session:
            name = await session.scalar(select(Business.timezone).where(Business.id == business_id))
            held = [hold.span for hold in await _holds(session, business_id, start, end)]
        zone = _zone(name or self._settings.timezone, self._settings.timezone)
        length = timedelta(minutes=self._settings.slot_minutes)
        first = self._settings.day_start_hour * 60
        last = self._settings.day_end_hour * 60 - self._settings.slot_minutes
        openings: list[AvailableSlot] = []
        day = start.astimezone(zone).date()
        while day <= end.astimezone(zone).date():
            if day.weekday() not in CLOSED_WEEKDAYS:
                for minute in range(first, last + 1, self._settings.slot_minutes):
                    slot_start = datetime.combine(day, time(minute // 60, minute % 60), tzinfo=zone)
                    slot_end = slot_start + length
                    if not start <= slot_start < end or _busy(business_id, slot_start):
                        continue
                    if any(
                        _utc(slot_start) < h_end and h_start < _utc(slot_end)
                        for h_start, h_end in held
                    ):
                        continue
                    openings.append(AvailableSlot(start=slot_start, end=slot_end))
            day += timedelta(days=1)
        return tuple(openings)

    async def book(self, request: BookingRequest) -> BookingResult:
        async with self._session_factory() as session:
            site_url = await session.scalar(
                select(Business.site_url).where(Business.id == request.business_id)
            )
            holds = await _holds(session, request.business_id, request.slot_start, request.slot_end)
        if any(hold.booked and hold.reference != request.reference for hold in holds):
            if not site_url:
                raise AvailabilityError("the demo slot is already booked")
            logger.info(
                "demo mode: %s already booked (business %s); sending a link to pick again",
                request.slot_start.isoformat(),
                request.business_id,
            )
            return BookingResult(
                kind=BookingKind.LINK, link=site_url, event_type_uri=DEMO_EVENT_TYPE_URI
            )
        logger.info(
            "demo mode: booked %s (business %s), no calendar touched",
            request.slot_start.isoformat(),
            request.business_id,
        )
        return BookingResult(kind=BookingKind.BOOKED, event_type_uri=DEMO_EVENT_TYPE_URI)

    async def find_booking(self, request: BookingRequest) -> BookingResult | None:
        # Nothing external was ever written, so a retried attempt books again.
        return None

    async def booking_event_type_uri(self, business_id: BusinessId) -> str | None:
        return DEMO_EVENT_TYPE_URI

    async def cancel_booking(self, business_id: BusinessId, event_uri: str) -> None:
        logger.info("demo mode: canceled %s (business %s)", event_uri, business_id)


class DemoBookedEvents:
    """The dashboard's bookings: approved website requests booked in GVAS,
    and bookings still in force while their reschedule waits for the owner."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def serves(self, business_id: BusinessId) -> bool:
        return True

    async def upcoming(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[CalendarEvent, ...]:
        async with self._session_factory() as session:
            rows = await _holding_rows(session, business_id, end)
        events = [
            _booking_event(row, span)
            for row in rows
            for span in _booked_spans(row)
            if span[0] < _utc(end) and span[1] > _utc(start)
        ]
        return tuple(sorted(events, key=lambda event: event.start))


class _Hold:
    __slots__ = ("booked", "reference", "span")

    def __init__(self, span: Span, reference: str, *, booked: bool) -> None:
        self.span = span
        self.reference = reference
        self.booked = booked


async def _holding_rows(
    session: AsyncSession, business_id: BusinessId, before: datetime
) -> list[IntakeRow]:
    rows = await session.scalars(
        select(IntakeRow).where(
            IntakeRow.business_id == business_id,
            IntakeRow.state.in_(HOLDING_STATES),
        )
    )
    return [row for row in rows if any(span[0] < _utc(before) for span in _spans(row))]


async def _holds(
    session: AsyncSession, business_id: BusinessId, start: datetime, end: datetime
) -> list[_Hold]:
    """Every span the business's calendar holds that overlaps ``start``–``end``.

    ``booked`` marks a confirmed booking (as opposed to a request still
    waiting for the owner), which is what a second booking must not overlap.
    """

    holds: list[_Hold] = []
    for row in await _holding_rows(session, business_id, end):
        booked = set(_booked_spans(row))
        for span in _spans(row):
            if span[0] < _utc(end) and span[1] > _utc(start):
                holds.append(_Hold(span, row.reference, booked=span in booked))
    return holds


def _span(slot_start: datetime, slot_end: datetime | None) -> Span:
    return _utc(slot_start), _utc(slot_end or slot_start + timedelta(minutes=1))


def _superseded(row: IntakeRow) -> Span | None:
    if not row.superseded_booking:
        return None
    old = SupersededBooking.model_validate(row.superseded_booking)
    return _span(old.slot_start, old.slot_end)


def _booked_spans(row: IntakeRow) -> list[Span]:
    """The confirmed bookings on a row: its approved booking, and the one a
    pending reschedule would replace, which stays in force until approval."""

    spans: list[Span] = []
    if (
        row.state == IntakeState.APPROVED.value
        and row.booking_kind == BookingKind.BOOKED.value
        and row.requested_slot_start is not None
    ):
        spans.append(_span(row.requested_slot_start, row.requested_slot_end))
    superseded = _superseded(row)
    if superseded is not None:
        spans.append(superseded)
    return spans


def _spans(row: IntakeRow) -> list[Span]:
    spans = _booked_spans(row)
    if row.requested_slot_start is not None:
        requested = _span(row.requested_slot_start, row.requested_slot_end)
        if requested not in spans:
            spans.append(requested)
    return spans


def _booking_event(row: IntakeRow, span: Span) -> CalendarEvent:
    collected = row.collected or {}

    def text(key: str) -> str | None:
        value = collected.get(key)
        return value.strip() or None if isinstance(value, str) else None

    details = text("details")
    title = "Estimate"
    if details:
        title = details if len(details) <= TITLE_MAX_CHARS else details[: TITLE_MAX_CHARS - 1] + "…"
    return CalendarEvent(
        source=CalendarEventSource.BOOKING,
        title=title,
        start=span[0],
        end=span[1],
        location=text("address"),
        invitee_name=text("name"),
        invitee_email=text("email"),
        reference=row.reference,
    )


def _busy(business_id: BusinessId, slot_start: datetime) -> bool:
    seed = f"{business_id}:{_utc(slot_start).isoformat()}".encode()
    return hashlib.sha256(seed).digest()[0] % BUSY_ONE_IN == 0


def _zone(name: str, fallback: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(fallback)
