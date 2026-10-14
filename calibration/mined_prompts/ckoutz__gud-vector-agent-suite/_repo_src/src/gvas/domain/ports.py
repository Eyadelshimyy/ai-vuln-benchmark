from datetime import date, datetime
from typing import Protocol

from gvas.domain.appointments import AppointmentLookupPort as AppointmentLookupPort
from gvas.domain.calendar_blocks import DayHours, Interval
from gvas.domain.customers import PortalLoginEmailRequest
from gvas.domain.identifiers import BusinessId
from gvas.domain.intake import (
    AvailableSlot,
    BookingRequest,
    BookingResult,
    IntakeTurn,
    IntakeTurnRequest,
)
from gvas.domain.intents import IntentResolution
from gvas.domain.messages import (
    AttachmentPayload,
    AttachmentReference,
    AudioReference,
    ConversationRef,
    CustomerDeliveryRequest,
    CustomerTextRequest,
    DeliveryReceipt,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TranscriptResult,
)
from gvas.domain.object_storage import ObjectCustodyRequest, StoredObject
from gvas.domain.owner import CalendarEvent
from gvas.domain.owner_email import OwnerEmailRequest
from gvas.domain.owner_understanding import (
    OwnerIntentInterpretation,
    OwnerIntentInterpretationRequest,
)
from gvas.domain.payments import (
    BillingCustomerRequest,
    BillingCustomerResult,
    BillingPortalRequest,
    BillingPortalResult,
    PaymentCheckoutRequest,
    PaymentCheckoutResult,
)
from gvas.domain.quotes import QuoteDraftProposal, QuoteDraftRequest
from gvas.domain.reporting import (
    ChecklistEvidence,
    ChecklistEvidenceAnnotation,
    ChecklistEvidenceRequest,
)


class IntentResolutionPort(Protocol):
    async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution: ...


class OwnerReplyPort(Protocol):
    async def send(
        self, conversation_ref: ConversationRef, message: OutboundOwnerMessage
    ) -> DeliveryReceipt: ...


class AttachmentAccessPort(Protocol):
    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload: ...


class ObjectStoragePort(Protocol):
    """Managed custody of bytes the system must retain (D7b).

    ``put`` is idempotent for a given request: storing the same content under
    the same scope and name twice yields one object and the same reference.
    ``fetch`` is tenant-checked by the adapter: a reference belonging to
    another business is refused even when the caller holds it.
    """

    async def put(self, request: ObjectCustodyRequest) -> StoredObject: ...

    async def fetch(
        self, business_id: BusinessId, artifact: AttachmentReference
    ) -> AttachmentPayload: ...


class TranscriptionPort(Protocol):
    async def transcribe(self, audio: AudioReference) -> TranscriptResult: ...


class CustomerQuoteDeliveryPort(Protocol):
    async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt: ...


class PaymentCheckoutPort(Protocol):
    """Opens one hosted checkout session for an accepted quote.

    Implementations must honour ``idempotency_key`` so a retried call returns
    the session already opened. Errors raise ``PaymentCheckoutError`` with
    provider credentials and raw responses stripped out.
    """

    async def create_checkout(self, request: PaymentCheckoutRequest) -> PaymentCheckoutResult: ...

    async def expire_checkout(self, session_id: str) -> None:
        """Close an open session so it can no longer be paid."""
        ...


class BillingAccountPort(Protocol):
    """Provider-side customers and the self-service billing portal.

    Same rules as :class:`PaymentCheckoutPort`: idempotent on the key,
    errors raise ``PaymentCheckoutError`` with nothing sensitive inside.
    """

    async def create_customer(self, request: BillingCustomerRequest) -> BillingCustomerResult: ...

    async def create_billing_portal_session(
        self, request: BillingPortalRequest
    ) -> BillingPortalResult: ...


class PortalLoginEmailPort(Protocol):
    async def send_login_link(self, request: PortalLoginEmailRequest) -> DeliveryReceipt: ...


class CustomerTextDeliveryPort(Protocol):
    """Texts a customer. A returned ``FAILED`` receipt or a raised error both
    mean the text did not go; the caller decides whether to retry."""

    async def send_text(self, request: CustomerTextRequest) -> DeliveryReceipt: ...


class ChecklistEvidencePort(Protocol):
    """Attributes a completed review's checklist items to transcript evidence.

    Completeness review reports only the items still missing, so evidence for
    the satisfied items is attributed through this port before a report snapshot
    is assembled.
    """

    async def attribute(
        self, request: ChecklistEvidenceRequest
    ) -> tuple[ChecklistEvidence, ...]: ...


class ChecklistEvidenceAnnotatorPort(Protocol):
    """Attaches supporting excerpts to items a primary attributor already satisfied.

    The annotator never decides whether an item is satisfied; it receives the
    attributed evidence and may only return verbatim transcript excerpts for the
    observed items.
    """

    async def annotate(
        self, request: ChecklistEvidenceRequest, attributed: tuple[ChecklistEvidence, ...]
    ) -> tuple[ChecklistEvidenceAnnotation, ...]: ...


class QuoteDraftingPort(Protocol):
    async def draft(self, request: QuoteDraftRequest) -> QuoteDraftProposal: ...


class IntakeAgentPort(Protocol):
    """One turn of the website booking agent.

    The implementation is the review model behind a strict JSON schema; the
    service layer still guards the reply (no prices, no invented availability)
    so a misbehaving model cannot leak through.
    """

    async def turn(self, request: IntakeTurnRequest) -> IntakeTurn: ...


class AvailabilityPort(Protocol):
    """The business's calendar: what is free, and how an approved slot is booked.

    ``available_slots`` returns real openings only — the agent never invents
    times. ``book`` runs only after the owner approved; implementations either
    book the invitee directly (``BOOKED``) or mint a one-time scheduling link
    (``LINK``) the customer confirms themselves. Errors raise a sanitized
    provider error, never credentials or raw responses.
    """

    async def available_slots(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[AvailableSlot, ...]: ...

    async def book(self, request: BookingRequest) -> BookingResult: ...

    async def find_booking(self, request: BookingRequest) -> BookingResult | None:
        """Whether the booking described by ``request`` already exists.

        Retried arrange commands reconcile through this before calling
        ``book`` again so a crashed earlier attempt cannot double-book.
        """
        ...

    async def booking_event_type_uri(self, business_id: BusinessId) -> str | None:
        """The event type ``book`` would use for this business right now.

        The arrange service resolves and persists this before the provider
        call so a retried attempt still targets the type the original call
        used — otherwise a changed first-active type could double-book.
        """
        ...

    async def cancel_booking(self, business_id: BusinessId, event_uri: str) -> None:
        """Cancel an existing calendar event (a declined customer pick).

        An event the provider no longer knows is already canceled — only
        provider failures raise ``AvailabilityError``.
        """
        ...


class BookedEventsPort(Protocol):
    """The bookings customers scheduled (Calendly), for the owner dashboard.

    Errors raise ``OwnerCalendarError`` with an owner-safe message.
    """

    def serves(self, business_id: BusinessId) -> bool: ...

    async def upcoming(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[CalendarEvent, ...]: ...


class CalendarFeedPort(Protocol):
    """The owner's own calendar (Google, Apple, Outlook, ...) read through its
    private subscription link. Errors raise ``OwnerCalendarError``."""

    async def events(
        self, feed_url: str, start: datetime, end: datetime
    ) -> tuple[CalendarEvent, ...]: ...


class ScheduleBlockPort(Protocol):
    """Hours on the booking calendar itself, one date at a time.

    Changing a date here changes what every booking page offers, the website
    chat and the business's own booking link alike. Errors raise
    ``AvailabilityError`` with a sanitized message.
    """

    def serves(self, business_id: BusinessId) -> bool: ...

    async def schedule_timezone(self, business_id: BusinessId) -> str | None:
        """The IANA zone the calendar's hours are written in."""
        ...

    async def day_hours(self, business_id: BusinessId, day: date) -> DayHours: ...

    async def set_day_hours(
        self, business_id: BusinessId, day: date, intervals: tuple[Interval, ...] | None
    ) -> None:
        """Pin ``day`` to ``intervals``; ``None`` drops the date's own rule so
        the weekly hours apply again."""
        ...


class OwnerEmailPort(Protocol):
    """Sends one owner e-mail (notice or reply): text + HTML, reply routing
    and thread headers. Idempotent on the request's ``idempotency_key``."""

    async def send(self, request: OwnerEmailRequest) -> DeliveryReceipt: ...


class OwnerIntentInterpreterPort(Protocol):
    """The model fallback for owner messages the trigger grammar does not
    match: answers one of the offered candidates or ``unclear``."""

    async def interpret(
        self, request: OwnerIntentInterpretationRequest
    ) -> OwnerIntentInterpretation: ...
