"""Website intake conversations: the AI booking agent's domain model.

A customer chats on the business's site; the agent collects the request,
offers real calendar availability and — only when the owner approves — the
booking is arranged. Conversations carry their own bearer credential (the
``conversationToken``), stored hashed like portal sessions, and are always
scoped to a single business: ``(business_id, id)`` is the tenant boundary
every repository and every message row honours.

Nothing here books anything: state moves to ``awaiting_owner`` when the
customer picks a slot and only owner decisions change it from there.
"""

import base64
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from datetime import date, datetime, timedelta, tzinfo
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gvas.domain.customers import hash_portal_token, new_portal_token, portal_token_matches
from gvas.domain.identifiers import (
    BusinessId,
    CustomerId,
    IntakeConversationId,
    IntakeMessageId,
    JsonValue,
    OutboxCommandId,
    WorkflowIntent,
)
from gvas.domain.outbox import OutboxCommand
from gvas.domain.owner_email import OwnerEmailContent, OwnerEmailDetail, OwnerEmailRequest
from gvas.domain.time_zones import zone_label

BOOKING_INTENT = WorkflowIntent("booking_decision")
INTAKE_CHANNEL_WEB = "web"
# Opened from a verified customer-portal session: the customer is known.
INTAKE_CHANNEL_PORTAL = "portal"
INTAKE_CONVERSATION_TTL = timedelta(hours=24)
INTAKE_MESSAGE_MAX_CHARS = 2000
INTAKE_MAX_USER_MESSAGES = 30
SERVICE_REQUEST_SOURCE_INTAKE = "intake"
SLOT_OFFER_LIMIT = 5
SLOT_OFFER_BUSINESS_DAYS = 7
DEFAULT_SLOT_MINUTES = 60

INTAKE_BOOKING_ARRANGE_COMMAND_TYPE = "intake_booking.arrange"
INTAKE_BOOKING_ARRANGE_COMMAND_NAMESPACE = UUID("8c1f7a2e-4b63-4d59-a6e2-9f0c1d2b3a47")
INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE = "intake_customer.email"
INTAKE_CUSTOMER_EMAIL_COMMAND_NAMESPACE = UUID("3a9d1c5f-7e24-4b18-9c36-2e5f8a1d4b60")
INTAKE_CUSTOMER_TEXT_COMMAND_TYPE = "intake_customer.text"
INTAKE_CUSTOMER_TEXT_COMMAND_NAMESPACE = UUID("5f2b8d1a-9c47-4e63-b1d5-8a3f6c2e9d15")
OWNER_NOTICE_EMAIL_COMMAND_TYPE = "owner_notice.email"
OWNER_NOTICE_EMAIL_COMMAND_NAMESPACE = UUID("c6e2f4a8-1d3b-4f7e-9a5c-0b8d2e6f4a13")
INTAKE_BOOKING_CANCEL_COMMAND_TYPE = "intake_booking.cancel"
INTAKE_BOOKING_CANCEL_COMMAND_NAMESPACE = UUID("7e4b2a91-3c58-4d1e-b6f9-0a2d5c8e4f17")
# Public one-click owner decision links hang off this path: the signed token
# names the request and the action. Must match the route mounted in
# interfaces/http/public.py.
BOOKING_DECISION_PATH = "/v1/intake/booking-decisions/"

DECLINE_REASON_MAX_CHARS = 200
INTAKE_BRIEF_MAX_CHARS = 1000
INTAKE_QUESTIONS_MAX_CHARS = 2000
INTAKE_OPENING_MAX_CHARS = 600
INTAKE_OFFER_LINE_MAX_CHARS = 300
INTAKE_VISIT_QUESTION_MAX_CHARS = 300
VISIT_NOTE_LABEL = "Before the visit:"
VISIT_ANSWER_MAX_CHARS = 500
INTAKE_DETAILS_MAX_CHARS = 2000
INTAKE_NOTES_MAX_CHARS = 2000

PRICE_GUARD_REPLY = (
    "I can't discuss pricing — the owner will confirm pricing when they review your request."
)

_CURRENCY_AMOUNT = re.compile(
    r"(?:[$€£]\s*\d|\b\d+(?:\.\d{1,2})?\s*(?:usd|dollars?|euros?|bucks)\b)", re.IGNORECASE
)
_SLOT_PREFIX = re.compile(r"^slot:(?P<start>\S+)\s*$", re.IGNORECASE)
_BOOKING_COMMAND = re.compile(
    r"^\s*(?P<action>approve|decline|cancel)\s+booking\s+(?P<reference>[0-9a-zA-Z-]{4,32})"
    r"(?:\s+(?P<reason>.+?))?\s*$",
    re.IGNORECASE | re.DOTALL,
)


class IntakeModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class IntakeState(StrEnum):
    COLLECTING = "collecting"
    PROPOSING_SLOTS = "proposing_slots"
    AWAITING_OWNER = "awaiting_owner"
    APPROVED = "approved"
    DECLINED = "declined"
    CLOSED = "closed"


class IntakeMessageRole(StrEnum):
    USER = "user"
    AGENT = "agent"
    OWNER = "owner"


# ``approved`` is deliberately not terminal: the chat stays open so the
# customer can ask questions or move their call; declined and closed end it.
TERMINAL_INTAKE_STATES = frozenset({IntakeState.DECLINED, IntakeState.CLOSED})
# States in which the visitor has a call they can ask about, move or cancel.
LIVE_BOOKING_STATES = frozenset({IntakeState.AWAITING_OWNER, IntakeState.APPROVED})


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("intake timestamps must be timezone-aware")
    return value


class IntakeProfile(IntakeModel):
    """Per-business agent copy: what the business is and books (``brief``),
    what to find out beyond contact details (``questions``) and the first
    message a visitor reads (``opening``). ``offer_line`` replaces the line
    above the offered times and ``visit_question`` is asked once a time is
    picked. ``None`` means the generic default (no visit question)."""

    brief: str | None = Field(default=None, max_length=INTAKE_BRIEF_MAX_CHARS)
    questions: str | None = Field(default=None, max_length=INTAKE_QUESTIONS_MAX_CHARS)
    opening: str | None = Field(default=None, max_length=INTAKE_OPENING_MAX_CHARS)
    offer_line: str | None = Field(default=None, max_length=INTAKE_OFFER_LINE_MAX_CHARS)
    visit_question: str | None = Field(default=None, max_length=INTAKE_VISIT_QUESTION_MAX_CHARS)

    @property
    def is_configured(self) -> bool:
        return any(value for value in (self.brief, self.questions, self.opening))

    @property
    def requires_address(self) -> bool:
        """Only the generic questions ask for a service address; a business
        that lists its own questions decides what it needs."""

        return not (self.questions and self.questions.strip())


class IntakeCollected(IntakeModel):
    """What the agent has gathered so far; ``None`` means not yet asked.

    ``details`` is what the customer needs in their words; ``notes`` holds
    answers to the business's own intake questions. ``address``,
    ``property_type`` and ``urgency`` are optional, for businesses whose
    questions ask for them.
    """

    name: str | None = Field(default=None, max_length=200)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=64)
    details: str | None = Field(default=None, max_length=INTAKE_DETAILS_MAX_CHARS)
    notes: str | None = Field(default=None, max_length=INTAKE_NOTES_MAX_CHARS)
    address: str | None = Field(default=None, max_length=500)
    property_type: str | None = Field(default=None, alias="propertyType", max_length=200)
    urgency: str | None = Field(default=None, max_length=200)

    @model_validator(mode="before")
    @classmethod
    def legacy_problem_is_details(cls, data: object) -> object:
        """Snapshots written before ``details`` carried it as ``problem``."""

        if isinstance(data, Mapping) and "problem" in data:
            values = dict(data)
            problem = values.pop("problem")
            if values.get("details") is None:
                values["details"] = problem
            return values
        return data

    def merge(self, reported: "IntakeCollected") -> "IntakeCollected":
        """Fills blanks with newly reported values; never overwrites.

        ``notes`` accumulates instead: a new note is appended unless it
        already repeats (or extends) what is stored.
        """

        updates: dict[str, object] = {}
        for key in (
            "name",
            "email",
            "phone",
            "details",
            "address",
            "property_type",
            "urgency",
        ):
            value = getattr(reported, key)
            if getattr(self, key) is None and value is not None and value.strip():
                updates[key] = value.strip()
        notes = _merged_notes(self.notes, reported.notes)
        if notes != self.notes:
            updates["notes"] = notes
        return self.model_copy(update=updates)

    @property
    def ready_for_slots(self) -> bool:
        """Contact details, including a phone number, and what the customer
        needs — before scheduling."""

        return self._has(self.name, self.email, self.phone, self.details)

    def is_complete(self, *, address_required: bool, phone_required: bool = True) -> bool:
        """``phone_required`` is waived for known customers, whose record may
        legitimately hold no phone number."""

        if not self._has(self.name, self.email, self.details):
            return False
        if phone_required and not self._has(self.phone):
            return False
        return not address_required or self._has(self.address)

    @staticmethod
    def _has(*values: str | None) -> bool:
        return all(value is not None and value.strip() for value in values)

    def as_stored(self) -> dict[str, object]:
        return self.model_dump(mode="json", by_alias=True)

    def summary(
        self, *, address_required: bool = True, phone_required: bool = True
    ) -> dict[str, object] | None:
        """The public summary projection: only once the request is complete.

        ``problem`` mirrors ``details`` for widgets built against the
        original shape.
        """

        if not self.is_complete(address_required=address_required, phone_required=phone_required):
            return None
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "problem": self.details,
            "details": self.details,
            "notes": self.notes,
        }


_NOTE_BREAK = re.compile(r"(?<=[.!?])\s+|\s*;\s*")
_NOTE_LABEL_MAX_WORDS = 6


def _note_facts(text: str) -> list[str]:
    return [fact for raw in _NOTE_BREAK.split(text) if (fact := raw.strip(" ;"))]


def _note_key(fact: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", fact.casefold()).split())


def _note_label(fact: str) -> str | None:
    """``Trade`` for ``Trade: website building`` — a re-reported answer to the
    same question replaces the earlier one instead of stacking beside it."""

    label, colon, _ = fact.partition(":")
    if not colon or len(label.split()) > _NOTE_LABEL_MAX_WORDS:
        return None
    return _note_key(label) or None


def _merged_notes(stored: str | None, reported: str | None) -> str | None:
    """Fact-level merge: the agent re-reports its whole running summary on
    every turn (and again on reschedules), so each sentence/``;`` clause is
    kept once — repeats and clauses that begin an existing one are dropped, extensions replace
    what they extend, and a newer answer to the same ``Label:`` wins."""

    new = " ".join((reported or "").split())
    if not new:
        return stored
    merged: list[str] = []
    for fact in (*_note_facts(stored or ""), *_note_facts(new)):
        key = _note_key(fact)
        if not key:
            continue
        label = _note_label(fact)
        for index, existing in enumerate(merged):
            existing_key = _note_key(existing)
            # Prefix (not substring) containment, so "No leak" never swallows "Leak".
            if existing_key == key or existing_key.startswith(f"{key} "):
                break
            if key.startswith(f"{existing_key} ") or (
                label is not None and label == _note_label(existing)
            ):
                merged[index] = fact
                break
        else:
            merged.append(fact)
    text = ""
    for fact in merged:
        if text:
            text += " " if text[-1] in ".!?" else "; "
        text += fact
    return text[:INTAKE_NOTES_MAX_CHARS] or stored


class AvailableSlot(IntakeModel):
    start: datetime
    end: datetime

    _aware_start = field_validator("start")(_aware)
    _aware_end = field_validator("end")(_aware)


class IntakeMessage(IntakeModel):
    message_id: IntakeMessageId
    conversation_id: IntakeConversationId
    business_id: BusinessId
    role: IntakeMessageRole
    content: str = Field(min_length=1, max_length=INTAKE_MESSAGE_MAX_CHARS * 2)
    created_at: datetime

    _aware_created = field_validator("created_at")(_aware)


class SupersededBooking(IntakeModel):
    """The booking a pending reschedule request replaces.

    It stays on the provider's calendar while the owner decides: approving
    the new time cancels it, declining puts it back in force. Only one can
    exist — a conversation holds at most one booking at a time.
    """

    slot_start: datetime
    slot_end: datetime | None = None
    event_uri: str | None = None
    booking_kind: str | None = None
    booking_link: str | None = None
    booking_attempted_at: datetime | None = None
    booking_event_type_uri: str | None = None

    _aware_superseded_start = field_validator("slot_start")(_aware)

    def slot_label(self, zone: tzinfo | None) -> str:
        return format_slot_label(self.slot_start, zone)


class IntakeConversation(IntakeModel):
    conversation_id: IntakeConversationId
    business_id: BusinessId
    customer_id: CustomerId | None = None
    reference: str = Field(min_length=4, max_length=16)
    token_hash: str = Field(min_length=64, max_length=64)
    channel: str = INTAKE_CHANNEL_WEB
    state: IntakeState = IntakeState.COLLECTING
    collected: IntakeCollected = Field(default_factory=IntakeCollected)
    proposed_slots: tuple[AvailableSlot, ...] = ()
    requested_slot_start: datetime | None = None
    requested_slot_end: datetime | None = None
    booking_kind: str | None = None
    booking_link: str | None = None
    # Set before the provider call runs: a retried arrange command that sees
    # this reconciles the booking instead of booking twice.
    booking_attempted_at: datetime | None = None
    # The provider event URI a webhook confirmed exists for this request.
    # Doubles as the processed-event marker so redelivered webhooks no-op.
    booked_event_uri: str | None = None
    # The booking a pending reschedule request replaces: kept until the owner
    # decides, then cancelled (approve) or restored (decline).
    superseded_booking: SupersededBooking | None = None
    # Set while the customer is choosing a new time for an existing booking:
    # ``proposed_slots`` are on the table without the conversation leaving
    # its booked state, so the booking stands if they never pick.
    reschedule_offered_at: datetime | None = None
    # The provider event type the arrange step booked on; webhooks for other
    # event types are not this request and are ignored.
    booking_event_type_uri: str | None = None
    decision_reason: str | None = None
    decision_at: datetime | None = None
    owner_notified_at: datetime | None = None
    escalation_notified_at: datetime | None = None
    # The visitor's answer to the site's SMS consent checkbox, if it sent one.
    sms_consent: bool | None = None
    sms_consent_at: datetime | None = None
    expires_at: datetime
    created_at: datetime
    updated_at: datetime

    _aware_expires = field_validator("expires_at")(_aware)
    _aware_created = field_validator("created_at")(_aware)
    _aware_updated = field_validator("updated_at")(_aware)

    def token_matches(self, raw_token: str) -> bool:
        return portal_token_matches(raw_token, self.token_hash)

    def is_live(self, now: datetime) -> bool:
        return now < self.expires_at and self.state not in TERMINAL_INTAKE_STATES

    @property
    def has_live_booking(self) -> bool:
        """A booking request is with the owner, or an approved one stands."""

        return self.state in LIVE_BOOKING_STATES

    @property
    def is_choosing_new_time(self) -> bool:
        return self.reschedule_offered_at is not None and self.has_live_booking

    def booking_snapshot(self) -> SupersededBooking | None:
        """What an approved booking would need restored if a reschedule of it
        is declined; ``None`` when nothing is on the calendar yet."""

        if self.requested_slot_start is None or self.state is not IntakeState.APPROVED:
            return None
        return SupersededBooking(
            slot_start=self.requested_slot_start,
            slot_end=self.requested_slot_end,
            event_uri=self.booked_event_uri,
            booking_kind=self.booking_kind,
            booking_link=self.booking_link,
            booking_attempted_at=self.booking_attempted_at,
            booking_event_type_uri=self.booking_event_type_uri,
        )

    def find_proposed_slot(self, start: datetime) -> AvailableSlot | None:
        for slot in self.proposed_slots:
            if slot.start == start:
                return slot
        return None

    def with_updates(self, now: datetime, **changes: object) -> "IntakeConversation":
        changes["updated_at"] = now
        return self.model_copy(update=changes)


class BookingDecisionAction(StrEnum):
    APPROVE = "approve"
    DECLINE = "decline"
    # Owner-channel only: never minted into an e-mail link.
    CANCEL = "cancel"


# --- One-click owner decision links -------------------------------------------
#
# The owner notification e-mail carries a signed ``Approve`` / ``Decline`` URL
# per booking request. The token is HMAC-signed, names one conversation and one
# action, expires after a week, and is stamped with the request's
# ``owner_notified_at`` so a reschedule (which issues a fresh notice and stamp)
# invalidates every older link — single-use is therefore implicit: a decided
# request is no longer ``awaiting_owner`` and a superseded request no longer
# matches the stamp.

DECISION_LINK_TTL = timedelta(days=7)
_DECISION_LINK_SIG_CHARS = 43  # ~172 bits of the hex digest — plenty


class BookingDecisionLinkError(ValueError):
    """A decision link could not be honoured; the message is page-safe."""


class InvalidDecisionLinkError(BookingDecisionLinkError):
    """Malformed or wrongly signed token."""


class ExpiredDecisionLinkError(BookingDecisionLinkError):
    """Past ``DECISION_LINK_TTL``."""


class BookingDecisionLink(IntakeModel):
    business_id: BusinessId
    conversation_id: IntakeConversationId
    reference: str
    action: BookingDecisionAction
    # Epoch seconds of the request's ``owner_notified_at`` this link belongs to.
    request_epoch: int
    expires_epoch: int

    def matches_request(self, conversation: IntakeConversation) -> bool:
        return (
            conversation.conversation_id == self.conversation_id
            and conversation.owner_notified_at is not None
            and int(conversation.owner_notified_at.timestamp()) == self.request_epoch
        )


def booking_decision_link_token(
    secret: str,
    *,
    conversation: IntakeConversation,
    action: BookingDecisionAction,
    now: datetime,
) -> str:
    """The signed token embedded in an owner e-mail decision link."""

    if conversation.owner_notified_at is None:
        raise ValueError("a decision link needs a notified request")
    payload = {
        "a": action.value,
        "b": str(conversation.business_id),
        "c": str(conversation.conversation_id),
        "r": conversation.reference,
        "n": int(conversation.owner_notified_at.timestamp()),
        "e": int((now + DECISION_LINK_TTL).timestamp()),
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    signature = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256)
    return f"{body}.{signature.hexdigest()[:_DECISION_LINK_SIG_CHARS]}"


def parse_booking_decision_link_token(
    secret: str, token: str, *, now: datetime
) -> BookingDecisionLink:
    body, sep, signature = token.rpartition(".")
    if not sep or not body:
        raise InvalidDecisionLinkError("malformed link")
    expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256)
    if not hmac.compare_digest(signature, expected.hexdigest()[:_DECISION_LINK_SIG_CHARS]):
        raise InvalidDecisionLinkError("bad link signature")
    try:
        payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
        link = BookingDecisionLink(
            business_id=BusinessId(UUID(str(payload["b"]))),
            conversation_id=IntakeConversationId(UUID(str(payload["c"]))),
            reference=str(payload["r"]),
            action=BookingDecisionAction(payload["a"]),
            request_epoch=int(payload["n"]),
            expires_epoch=int(payload["e"]),
        )
        if link.action is BookingDecisionAction.CANCEL:
            raise ValueError("cancel is not a link action")
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidDecisionLinkError("malformed link") from error
    if now.timestamp() >= link.expires_epoch:
        raise ExpiredDecisionLinkError("the link has expired")
    return link


class BookingDecision(IntakeModel):
    action: BookingDecisionAction
    reference: str = Field(min_length=4, max_length=32)
    reason: str | None = None


def booking_decision(text: str) -> BookingDecision | None:
    """``approve booking <ref>`` / ``decline booking <ref> [reason]`` /
    ``cancel booking <ref>``."""

    match = _BOOKING_COMMAND.match(text)
    if match is None:
        return None
    reason = match.group("reason")
    return BookingDecision(
        action=BookingDecisionAction(match.group("action").lower()),
        reference=match.group("reference").lower(),
        reason=sanitize_owner_reason(reason) if reason else None,
    )


def sanitize_owner_reason(reason: str) -> str:
    """The owner's decline note is shown to the customer: one line, no control
    characters, capped."""

    text = " ".join("".join(c if c.isprintable() else " " for c in reason).split())
    return text[:DECLINE_REASON_MAX_CHARS]


def slot_message_start(text: str) -> datetime | None:
    """``slot:<start>`` — the widget's deterministic pick."""

    match = _SLOT_PREFIX.match(text.strip())
    if match is None:
        return None
    try:
        value = datetime.fromisoformat(match.group("start"))
    except ValueError:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value


def contains_currency_amount(text: str) -> bool:
    return bool(_CURRENCY_AMOUNT.search(text))


def scrub_agent_reply(reply: str) -> str:
    """A reply that names a price is replaced wholesale: the owner confirms
    pricing, never the agent."""

    cleaned = " ".join(reply.split()).strip()
    if not cleaned:
        return "Could you tell me a bit more about what you need?"
    if contains_currency_amount(cleaned):
        return PRICE_GUARD_REPLY
    return cleaned


def new_conversation_token() -> str:
    return new_portal_token()


def conversation_token_hash(token: str) -> str:
    return hash_portal_token(token)


def new_reference() -> str:
    return uuid4().hex[:8]


def next_business_days(anchor: datetime, count: int) -> set[date]:
    """The dates of the next ``count`` Mon–Fri days in the anchor's timezone.

    Today's date is excluded so same-day slots are never offered.
    """

    days: set[date] = set()
    day = anchor.date()
    while len(days) < count:
        day += timedelta(days=1)
        if day.weekday() < 5:
            days.add(day)
    return days


def pick_offer_slots(
    slots: tuple[AvailableSlot, ...], *, now: datetime, limit: int = SLOT_OFFER_LIMIT
) -> tuple[AvailableSlot, ...]:
    """Up to ``limit`` slots spread over the next business days.

    Only slots inside the following ``SLOT_OFFER_BUSINESS_DAYS`` weekdays are
    considered; one slot per day is taken first, then days are revisited so a
    sparse week still yields up to ``limit`` options.
    """

    if not slots:
        return ()
    # Slot dates are business-local; the day window must be anchored in the
    # same zone or a business a day behind the server's clock loses the first
    # day entirely.
    anchor = now.astimezone(slots[0].start.tzinfo)
    allowed_days = next_business_days(anchor, SLOT_OFFER_BUSINESS_DAYS)
    by_day: dict[date, list[AvailableSlot]] = {}
    for slot in sorted(slots, key=lambda slot: slot.start):
        if slot.start <= now:
            continue
        if slot.start.date() not in allowed_days:
            continue
        by_day.setdefault(slot.start.date(), []).append(slot)
    picked: list[AvailableSlot] = []
    while len(picked) < limit:
        progressed = False
        for day in sorted(by_day):
            if by_day[day]:
                picked.append(by_day[day].pop(0))
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
    return tuple(picked)


def format_slot_label(start: datetime, zone: tzinfo | None = None) -> str:
    """``Tue Sep 15, 9:00 AM Pacific time`` in the business's zone.

    Stored times come back in UTC, so callers pass the business zone; without
    one the label keeps the time's own zone.
    """

    local = start.astimezone(zone) if zone is not None else start
    hour = local.hour % 12 or 12
    meridiem = "AM" if local.hour < 12 else "PM"
    label = f"{local:%a} {local:%b} {local.day}, {hour}:{local:%M} {meridiem}"
    return f"{label} {zone_label(local)}".rstrip()


def booking_request_notice(
    conversation: IntakeConversation,
    *,
    business_name: str | None = None,
    profile: IntakeProfile | None = None,
    previous_label: str | None = None,
    previous_booked: bool = True,
    unverified_email: bool = False,
    zone: tzinfo | None = None,
) -> str:
    """The owner notice posted when a customer picks a slot.

    Businesses that list their own intake questions may not book at an
    address, so a missing one is left out instead of reported as unknown.
    ``previous_label`` marks a reschedule: the notice reads as an update and
    names the time the new request replaces.
    """

    collected = conversation.collected
    who = [collected.name or "A customer"]
    if collected.address:
        who.append(collected.address)
    elif profile is None or profile.requires_address:
        who.append("address unknown")
    details = collected.details or "New request"
    requested = (
        format_slot_label(conversation.requested_slot_start, zone)
        if conversation.requested_slot_start is not None
        else "no time picked"
    )
    ref = conversation.reference
    kind = "Updated booking request" if previous_label else "Booking request"
    lines = [
        f"{kind} #{ref} — {', '.join(who)}. {details}. Requested {requested}.",
    ]
    if previous_label:
        if previous_booked:
            lines.append(f"Replaces {previous_label} — cancel the old booking only on approve.")
        else:
            lines.append(f"Replaces the earlier request for {previous_label}.")
    if collected.notes:
        lines.append(f"Notes: {collected.notes}")
    if unverified_email:
        lines.append(UNVERIFIED_EMAIL_NOTE)
    lines.append(f"Reply `approve booking {ref}` or `decline booking {ref} <reason>`.")
    lines.append(
        "Busy then? Reply like `unavailable 8-12` to block that time and send the "
        "customer a link to pick another."
    )
    if business_name:
        lines.append(f"Business: {business_name}.")
    return "\n".join(lines)


def booking_request_email(
    conversation: IntakeConversation,
    *,
    business_name: str | None = None,
    profile: IntakeProfile | None = None,
    previous_label: str | None = None,
    previous_booked: bool = True,
    unverified_email: bool = False,
    zone: tzinfo | None = None,
) -> OwnerEmailContent:
    """The e-mail form of ``booking_request_notice``: same facts, laid out
    as a details list with the owner-channel commands in the footer."""

    collected = conversation.collected
    ref = conversation.reference
    kind = "Updated booking request" if previous_label else "New booking request"
    name = collected.name or "A customer"
    details = [OwnerEmailDetail(label="Customer", value=name)]
    if collected.phone:
        details.append(OwnerEmailDetail(label="Phone", value=collected.phone))
    if collected.email:
        details.append(OwnerEmailDetail(label="E-mail", value=collected.email))
    if collected.address:
        details.append(OwnerEmailDetail(label="Address", value=collected.address))
    elif profile is None or profile.requires_address:
        details.append(OwnerEmailDetail(label="Address", value="unknown"))
    details.append(OwnerEmailDetail(label="Request", value=collected.details or "New request"))
    details.append(
        OwnerEmailDetail(
            label="Requested",
            value=(
                format_slot_label(conversation.requested_slot_start, zone)
                if conversation.requested_slot_start is not None
                else "no time picked"
            ),
        )
    )
    if previous_label:
        details.append(
            OwnerEmailDetail(
                label="Replaces",
                value=(
                    f"{previous_label} — the old booking is canceled only if you approve"
                    if previous_booked
                    else f"the earlier request for {previous_label}"
                ),
            )
        )
    if collected.notes:
        details.append(OwnerEmailDetail(label="Notes", value=collected.notes))
    if unverified_email:
        details.append(OwnerEmailDetail(label="Unverified", value=UNVERIFIED_EMAIL_NOTE))
    return OwnerEmailContent(
        heading=f"{kind} #{ref}",
        details=tuple(details),
        commands=(
            f"In your owner channel: `approve booking {ref}` or `decline booking {ref} <reason>`.",
            "Busy then? `unavailable 8-12` blocks that time and sends the customer "
            "a link to pick another.",
        ),
        business_name=business_name,
        subject=f"{kind} #{ref} — {name}",
    )


UNVERIFIED_EMAIL_NOTE = (
    "This e-mail matches an existing customer, but the chat didn't prove it's "
    "them, so the request isn't linked to that customer's record."
)


def unverified_booking_change_notice(
    holder: IntakeConversation,
    requester: IntakeConversation,
    *,
    cancel: bool,
    business_name: str | None = None,
    zone: tzinfo | None = None,
) -> str:
    """A chat other than the booking's own asked to cancel or move it.

    The only link between them is an e-mail typed into the chat, which
    proves nothing, so nothing changes until the owner acts.
    """

    ref = holder.reference
    who = holder.collected.name or "a customer"
    when = (
        format_slot_label(holder.requested_slot_start, zone)
        if holder.requested_slot_start is not None
        else "their requested time"
    )
    ask = "cancel" if cancel else "move"
    lines = [
        f"Booking #{ref} — a different website chat asked to {ask} {who}'s {when}. "
        "It only typed the booking's e-mail, which doesn't prove who it is, "
        "so nothing has changed."
    ]
    typed = [value for value in (requester.collected.name, requester.collected.phone) if value]
    if typed:
        lines.append(f"That chat gave: {', '.join(typed)}")
    if holder.collected.email:
        lines.append(f"Booking contact: {holder.collected.email}")
    if cancel:
        lines.append(
            f"If you've checked with the customer, reply `cancel booking {ref}`. "
            "Otherwise no action is needed."
        )
    else:
        lines.append(
            "If you've checked with the customer, arrange the new time with them; "
            "the booking stays as it is."
        )
    if business_name:
        lines.append(f"Business: {business_name}.")
    return "\n".join(lines)


def cancel_request_notice(
    conversation: IntakeConversation,
    *,
    business_name: str | None = None,
    zone: tzinfo | None = None,
) -> str:
    """The owner notice when the customer cancels their own call.

    Their own booking is theirs to drop, so this reports a fact rather than
    asking for a decision.
    """

    collected = conversation.collected
    who = collected.name or "A customer"
    when = (
        format_slot_label(conversation.requested_slot_start, zone)
        if conversation.requested_slot_start is not None
        else "their requested time"
    )
    lines = [f"Booking #{conversation.reference} — {who} canceled {when}."]
    if collected.email:
        lines.append(f"Contact: {collected.email}")
    if business_name:
        lines.append(f"Business: {business_name}.")
    return "\n".join(lines)


def visit_note(answer: str) -> str:
    """The answer to the visit question as it is saved: ``Before the visit: …``."""

    return f"{VISIT_NOTE_LABEL} {' '.join(answer.split())[:VISIT_ANSWER_MAX_CHARS]}"


def with_visit_note(collected: IntakeCollected, note: str) -> IntakeCollected:
    """Adds ``note`` to the notes. It always lands in full: when the notes
    are full, older notes give way."""

    merged = collected.merge(IntakeCollected(notes=note))
    if merged.notes is not None and note in merged.notes:
        return merged
    room = INTAKE_NOTES_MAX_CHARS - len(note) - 2
    kept = (collected.notes or "")[: max(room, 0)].rstrip()
    return collected.model_copy(update={"notes": f"{kept}; {note}" if kept else note})


def visit_note_notice(conversation: IntakeConversation, note: str) -> str:
    """The owner notice for an answer that came after the request notice."""

    who = conversation.collected.name or "The customer"
    return f"{who} added a note to request {conversation.reference}: {note}"


def escalation_notice(conversation: IntakeConversation, summary: str) -> str:
    """The owner notice for ``needs_human`` turns: transcript summary only."""

    collected = conversation.collected
    details = ", ".join(
        part
        for part in (
            collected.name or "",
            collected.email or "",
            collected.details or "",
        )
        if part
    )
    text = f"Chat {conversation.reference} needs you — {details or 'a customer asked for help'}."
    if summary.strip():
        text += f"\n{summary.strip()[:800]}"
    return text


def _confirm_channels(texts: bool) -> str:
    """Only a customer who said yes to texts is told a text is coming."""
    return "email and text" if texts else "email"


def slot_confirmed_reply(
    slot: AvailableSlot, zone: tzinfo | None = None, *, texts: bool = False
) -> str:
    return (
        f"Great — I've requested {format_slot_label(slot.start, zone)}. "
        f"We'll confirm by {_confirm_channels(texts)} once the owner approves."
    )


def slot_held_reply(*, texts: bool = False) -> str:
    return (
        "That time is already with the team for approval — we'll confirm by "
        f"{_confirm_channels(texts)}."
    )


def intake_booking_arrange_command(
    conversation: IntakeConversation,
) -> OutboxCommand:
    """Worker command that turns an approval into a calendar booking.

    Deduped per decision: a webhook re-route puts the request back in front of
    the owner, and the second approval must enqueue its own arrange command
    instead of deduping against the first.
    """

    decision_marker = conversation.decision_at.isoformat() if conversation.decision_at else "first"
    command_id = OutboxCommandId(
        uuid5(
            INTAKE_BOOKING_ARRANGE_COMMAND_NAMESPACE,
            f"{conversation.business_id}:{conversation.conversation_id}:{decision_marker}",
        )
    )
    return OutboxCommand(
        command_id=command_id,
        business_id=conversation.business_id,
        command_type=INTAKE_BOOKING_ARRANGE_COMMAND_TYPE,
        payload={"conversation_id": str(conversation.conversation_id)},
        dedup_key=f"intake_booking:{conversation.conversation_id}:{decision_marker}",
    )


class IntakeCustomerEmail(IntakeModel):
    """A plain customer-facing email the worker can send verbatim.

    Owner notices also carry an ``html`` part; customer e-mails leave it
    unset.
    """

    business_id: BusinessId
    to: str = Field(min_length=3)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    html: str | None = None


def intake_customer_email_command(email: IntakeCustomerEmail) -> OutboxCommand:
    command_id = OutboxCommandId(
        uuid5(INTAKE_CUSTOMER_EMAIL_COMMAND_NAMESPACE, email.idempotency_key)
    )
    return OutboxCommand(
        command_id=command_id,
        business_id=email.business_id,
        command_type=INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE,
        payload={
            "to": email.to,
            "subject": email.subject,
            "body": email.body,
            "idempotency_key": email.idempotency_key,
        },
        dedup_key=f"intake_email:{email.idempotency_key}",
    )


def owner_notice_email_command(email: IntakeCustomerEmail) -> OutboxCommand:
    """The same verbatim e-mail send, typed so a failure reads as a missed
    owner copy rather than a missed customer e-mail."""

    return OutboxCommand(
        command_id=OutboxCommandId(
            uuid5(OWNER_NOTICE_EMAIL_COMMAND_NAMESPACE, email.idempotency_key)
        ),
        business_id=email.business_id,
        command_type=OWNER_NOTICE_EMAIL_COMMAND_TYPE,
        payload=_email_payload(email),
        dedup_key=f"owner_notice_email:{email.idempotency_key}",
    )


def intake_customer_email_request(
    business_id: BusinessId, payload: Mapping[str, object]
) -> IntakeCustomerEmail:
    fields = {key: payload.get(key) for key in ("to", "subject", "body", "idempotency_key")}
    if not all(isinstance(value, str) and value for value in fields.values()):
        raise ValueError("intake email command payload is incomplete")
    html = payload.get("html")
    return IntakeCustomerEmail(
        business_id=business_id,
        to=str(fields["to"]),
        subject=str(fields["subject"]),
        body=str(fields["body"]),
        idempotency_key=str(fields["idempotency_key"]),
        html=html if isinstance(html, str) and html else None,
    )


def intake_owner_email_request(
    business_id: BusinessId, payload: Mapping[str, object]
) -> OwnerEmailRequest:
    """An ``owner_notice.email`` payload as a multipart owner e-mail send."""

    email = intake_customer_email_request(business_id, payload)
    return OwnerEmailRequest(
        business_id=business_id,
        to=email.to,
        subject=email.subject,
        text=email.body,
        html=email.html,
        idempotency_key=email.idempotency_key,
    )


def _email_payload(email: IntakeCustomerEmail) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "to": email.to,
        "subject": email.subject,
        "body": email.body,
        "idempotency_key": email.idempotency_key,
    }
    if email.html:
        payload["html"] = email.html
    return payload


def intake_booking_cancel_command(
    conversation: IntakeConversation,
    event_uri: str | None = None,
) -> OutboxCommand:
    """Worker command that cancels the calendar event recorded on the request.

    Deduped per event URI: a declined re-routed booking cancels exactly the
    event the customer created, and a later different event cancels separately.
    ``event_uri`` lets a reschedule cancel a superseded event, which is no
    longer the conversation's recorded one.
    """

    event_uri = event_uri if event_uri is not None else (conversation.booked_event_uri or "")
    command_id = OutboxCommandId(
        uuid5(
            INTAKE_BOOKING_CANCEL_COMMAND_NAMESPACE,
            f"{conversation.business_id}:{event_uri}",
        )
    )
    return OutboxCommand(
        command_id=command_id,
        business_id=conversation.business_id,
        command_type=INTAKE_BOOKING_CANCEL_COMMAND_TYPE,
        payload={"event_uri": event_uri},
        dedup_key=f"intake_cancel:{conversation.business_id}:{event_uri}",
    )


def intake_customer_text_command(
    business_id: BusinessId,
    *,
    customer_id: CustomerId,
    phone: str,
    text: str,
    idempotency_key: str,
) -> OutboxCommand:
    command_id = OutboxCommandId(uuid5(INTAKE_CUSTOMER_TEXT_COMMAND_NAMESPACE, idempotency_key))
    return OutboxCommand(
        command_id=command_id,
        business_id=business_id,
        command_type=INTAKE_CUSTOMER_TEXT_COMMAND_TYPE,
        payload={
            "customer_id": str(customer_id),
            "phone": phone,
            "text": text,
            "idempotency_key": idempotency_key,
        },
        dedup_key=f"intake_text:{idempotency_key}",
    )


class IntakeConversationRepository(Protocol):
    async def add(self, conversation: IntakeConversation) -> None: ...

    async def get(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> IntakeConversation | None: ...

    async def find_by_reference(
        self, business_id: BusinessId, reference: str
    ) -> IntakeConversation | None: ...

    async def lock(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> IntakeConversation | None:
        """Row-level lock for the turn being written: concurrent customer
        posts serialize instead of overwriting each other's snapshot."""
        ...

    async def lock_by_reference(
        self, business_id: BusinessId, reference: str
    ) -> IntakeConversation | None:
        """Row-level lock by owner-facing reference: concurrent approve and
        decline decisions serialize so only one reaches a terminal state."""
        ...

    async def lock_latest_by_invitee_email(
        self, business_id: BusinessId, email: str
    ) -> IntakeConversation | None:
        """Row-level lock on the newest undecided conversation for ``email``.

        Booking webhooks arrive per invitee; the lock serializes the update
        against an owner decision that may land at the same moment."""
        ...

    async def find_by_token(
        self, conversation_id: IntakeConversationId, token_hash: str
    ) -> IntakeConversation | None:
        """Token-as-credential lookup: the bearer proves tenancy, so no
        business id is supplied or assumed here."""
        ...

    async def save(self, conversation: IntakeConversation) -> None: ...

    async def count_created_since(self, business_id: BusinessId, since: datetime) -> int: ...

    async def list_booking_requests(
        self, business_id: BusinessId, *, limit: int
    ) -> tuple[IntakeConversation, ...]:
        """Requests that reached the owner (waiting, approved or declined),
        most recently updated first."""
        ...

    async def list_awaiting_owner(
        self, business_id: BusinessId, *, limit: int
    ) -> tuple[IntakeConversation, ...]:
        """Requests still waiting for the owner, most recently updated first."""
        ...


class IntakeMessageRepository(Protocol):
    async def add(self, message: IntakeMessage) -> None: ...

    async def list_for(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> tuple[IntakeMessage, ...]: ...

    async def count_user(
        self, business_id: BusinessId, conversation_id: IntakeConversationId
    ) -> int: ...


class ExistingBookingStatus(StrEnum):
    """How far the customer's existing call has progressed."""

    REQUESTED = "requested"
    CONFIRMED = "confirmed"


class ExistingBooking(IntakeModel):
    """A call the customer already has — from this chat or an earlier one
    under the same e-mail — so the agent can discuss, move or cancel it
    instead of starting over.

    ``verified`` is false when only a typed e-mail links this chat to the
    booking: its status and time are withheld and a move or cancel goes to
    the owner.
    """

    status: ExistingBookingStatus | None = None
    slot_label: str | None = None
    verified: bool = True


class IntakeTurnRequest(IntakeModel):
    """One agent step: the transcript, collected state and (when proposing) the
    slots the customer may pick from."""

    business_id: BusinessId
    conversation_id: IntakeConversationId
    business_name: str = Field(min_length=1)
    transcript: tuple[IntakeMessage, ...] = ()
    collected: IntakeCollected = Field(default_factory=IntakeCollected)
    offered_slots: tuple[AvailableSlot, ...] = ()
    # Portal-linked customers already identified themselves; the agent skips
    # name/email/phone questions and only asks about the new service.
    known_customer: bool = False
    # Set once the customer has a requested or confirmed call: the agent then
    # answers questions, moves the booking or cancels it instead of collecting
    # a fresh request.
    existing_booking: ExistingBooking | None = None
    # The business's intake profile; ``None`` means the agent's generic
    # default.
    brief: str | None = Field(default=None, max_length=INTAKE_BRIEF_MAX_CHARS)
    questions: str | None = Field(default=None, max_length=INTAKE_QUESTIONS_MAX_CHARS)


class IntakeTurn(IntakeModel):
    reply: str = Field(min_length=1)
    collected: IntakeCollected = Field(default_factory=IntakeCollected)
    ready_for_slots: bool = False
    chosen_slot: datetime | None = None
    needs_human: bool = False
    # Booked-mode intents: the customer wants the existing call moved or
    # cancelled. Only honoured while an existing booking is in scope.
    wants_reschedule: bool = False
    wants_cancel: bool = False
    # Short transcript summary for the escalation notice.
    summary: str = ""


class IntakeAgentError(RuntimeError):
    """The model could not be reached or read; sanitized, like the other
    provider errors — no key, no raw response."""


class AvailabilityError(RuntimeError):
    """The calendar provider could not be reached or read; sanitized — no
    token, no raw response."""


class BookingKind(StrEnum):
    BOOKED = "booked"
    LINK = "link"


class BookingRequest(IntakeModel):
    business_id: BusinessId
    slot_start: datetime
    slot_end: datetime
    invitee_name: str = Field(min_length=1)
    invitee_email: str = Field(min_length=3)
    invitee_phone: str | None = None
    address: str | None = None
    details: str | None = None
    # The owner-facing request reference, carried into the scheduling link so
    # the resulting webhook can be bound back to this exact request.
    reference: str | None = None
    # The provider event type pinned when the attempt marker was persisted —
    # a retried attempt books and reconciles against the type the original
    # call used, not whatever is currently first active.
    event_type_uri: str | None = None

    _aware_start = field_validator("slot_start")(_aware)
    _aware_end = field_validator("slot_end")(_aware)


class BookingResult(IntakeModel):
    kind: BookingKind
    link: str | None = None
    # The provider event type used for the booking, when the adapter knows
    # it — lets webhooks confirm they belong to this request.
    event_type_uri: str | None = None


class BookingEventKind(StrEnum):
    CREATED = "created"
    CANCELED = "canceled"


class IntakeBookingEvent(IntakeModel):
    """A provider-confirmed calendar event, translated from a webhook.

    The vendor-specific shape is parsed in infrastructure; the application
    layer only sees the booking fact: whose invitee, which event, and when.
    """

    kind: BookingEventKind
    business_id: BusinessId
    invitee_email: str = Field(min_length=3)
    event_uri: str = Field(min_length=1)
    start: datetime
    end: datetime | None = None
    # The provider's event type and the request reference the booking link
    # carried (utm tracking), when the provider reports them.
    event_type_uri: str | None = None
    reference: str | None = None

    _aware_event_start = field_validator("start")(_aware)
