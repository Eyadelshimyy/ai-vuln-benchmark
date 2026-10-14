"""Hosted quote payments: checkout handoff and webhook bookkeeping.

Provider-neutral: the records name a ``provider`` string but never import or
know a vendor's API. The checkout port lives in ``gvas.domain.ports``; these
are the values it moves.
"""

import calendar
from collections.abc import Iterable
from datetime import date, datetime, timedelta, tzinfo
from enum import StrEnum
from typing import Literal, Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gvas.domain.enums import BillingInterval, QuotePaymentStatus
from gvas.domain.identifiers import (
    BusinessId,
    CustomerId,
    OutboxCommandId,
    QuoteId,
    SubscriptionId,
)
from gvas.domain.outbox import OutboxCommand

#: The only card-checkout provider wired today; kept as data, not code, so a
#: second provider is a new adapter plus a new literal here.
STRIPE_PROVIDER = "stripe"


class PaymentModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PaymentLineItem(PaymentModel):
    """One chargeable line handed to the checkout provider."""

    description: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    amount_minor: int = Field(ge=0)


class PaymentCheckoutRequest(PaymentModel):
    """Everything a checkout adapter needs to open one payment session."""

    business_id: BusinessId
    quote_id: QuoteId
    # The public-facing identifier the provider records; never the row id.
    client_reference: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    line_items: tuple[PaymentLineItem, ...] = Field(min_length=1)
    success_url: str = Field(min_length=1)
    cancel_url: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    # Set for a subscription checkout: every line item recurs on this
    # interval and the session is opened for ``customer_ref``, the provider's
    # handle for the paying customer. Unset means a one-time payment.
    recurring_interval: BillingInterval | None = None
    customer_ref: str | None = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        if not value.isalpha():
            raise ValueError("currency must contain three letters")
        return value.lower()

    @property
    def is_subscription(self) -> bool:
        return self.recurring_interval is not None


class PaymentCheckoutResult(PaymentModel):
    session_id: str = Field(min_length=1)
    checkout_url: str = Field(min_length=1)
    payment_intent_id: str | None = None
    # When the provider abandons an unpaid session; None means it does not
    # expire on its own.
    expires_at: datetime | None = None


class BillingCustomerRequest(PaymentModel):
    """Opens the provider-side customer a subscription is billed to."""

    business_id: BusinessId
    customer_id: CustomerId
    email: str = Field(min_length=3)
    name: str | None = None
    phone: str | None = None
    idempotency_key: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class BillingCustomerResult(PaymentModel):
    customer_ref: str = Field(min_length=1)


class BillingPortalRequest(PaymentModel):
    business_id: BusinessId
    customer_ref: str = Field(min_length=1)
    return_url: str = Field(min_length=1)


class BillingPortalResult(PaymentModel):
    url: str = Field(min_length=1)


class PaymentCheckoutError(RuntimeError):
    """Raised when a checkout provider could not open a session.

    Adapters sanitize the message: no credentials and no raw provider
    responses; the caller maps it to a neutral reply.
    """


class PaymentCheckoutClosedError(PaymentCheckoutError):
    """The provider refused to close a session that can no longer be paid
    (already expired or completed): there is nothing left to close."""


CHECKOUT_EXPIRE_COMMAND_TYPE = "payment.checkout_expire"
CHECKOUT_EXPIRE_COMMAND_NAMESPACE = UUID("6d0f4c1e-2b7a-4f53-9e18-3a5c7b9d1e24")
MANUAL_RECEIPT_COMMAND_TYPE = "payment.manual_receipt"
MANUAL_RECEIPT_COMMAND_NAMESPACE = UUID("a3e9b7d2-5c41-4e86-8f0a-7b2d4c6e9f13")
PLAN_NUDGE_COMMAND_TYPE = "payment.plan_nudge"
PLAN_NUDGE_COMMAND_NAMESPACE = UUID("c7f2a915-3d6b-4e08-9a41-5b8e2d7c0f36")
MANUAL_PROVIDER = "manual"
ENDED_SUBSCRIPTION_STATUSES = frozenset({"canceled", "incomplete_expired"})
#: Longest prepay one manual plan payment can cover.
PLAN_MONTHS_MAX = 24
#: How long before a manual plan's paid-through date the owner is nudged.
PLAN_NUDGE_LEAD = timedelta(days=7)


def checkout_expire_command(business_id: BusinessId, session_id: str) -> OutboxCommand:
    """Close an open hosted checkout from the worker, so a provider outage
    is retried instead of leaving the session payable."""

    return OutboxCommand(
        command_id=OutboxCommandId(uuid5(CHECKOUT_EXPIRE_COMMAND_NAMESPACE, session_id)),
        business_id=business_id,
        command_type=CHECKOUT_EXPIRE_COMMAND_TYPE,
        payload={"session_id": session_id},
        dedup_key=f"checkout_expire:{session_id}",
    )


def manual_receipt_command(
    *,
    business_id: BusinessId,
    quote_id: QuoteId,
    payment_id: UUID,
    to: str,
    subject: str,
    body: str,
    subscription_id: SubscriptionId | None = None,
) -> OutboxCommand:
    """The customer's receipt for a manual payment; the worker drops it when
    the payment was voided before delivery. For a plan payment it adds the
    plan's paid-through date as it stands when the receipt goes out."""

    key = f"manual-receipt:{payment_id}"
    return OutboxCommand(
        command_id=OutboxCommandId(uuid5(MANUAL_RECEIPT_COMMAND_NAMESPACE, key)),
        business_id=business_id,
        command_type=MANUAL_RECEIPT_COMMAND_TYPE,
        payload={
            "to": to,
            "subject": subject,
            "body": body,
            "idempotency_key": key,
            "quote_id": str(quote_id),
            "payment_id": str(payment_id),
            **({} if subscription_id is None else {"subscription_id": str(subscription_id)}),
        },
        dedup_key=key,
    )


class PaymentEventOutcome(StrEnum):
    """What one provider event means for the checkout attempt it names."""

    SUCCEEDED = "succeeded"
    PENDING = "pending"  # completed but the money has not settled yet
    FAILED = "failed"
    OTHER = "other"
    # Subscription lifecycle, keyed by ``subscription.subscription_ref``.
    SUBSCRIPTION_RENEWED = "subscription_renewed"
    SUBSCRIPTION_PAYMENT_FAILED = "subscription_payment_failed"
    SUBSCRIPTION_UPDATED = "subscription_updated"
    SUBSCRIPTION_CANCELLED = "subscription_cancelled"


class SubscriptionEventData(PaymentModel):
    """What a provider tells us about one subscription in one event; only
    the fields the event carried are set."""

    subscription_ref: str = Field(min_length=1)
    customer_ref: str | None = None
    status: str | None = None
    interval: BillingInterval | None = None
    #: The recurring price; only subscription objects carry it.
    amount_minor: int | None = Field(default=None, ge=0)
    #: What one invoice actually collected (credits, proration, tax included);
    #: reported to the owner, never written over the recurring price.
    paid_minor: int | None = Field(default=None, ge=0)
    currency: str | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool | None = None
    #: The provider's invoice behind a renewal; one ledger row per invoice.
    invoice_ref: str | None = None


class PaymentWebhookEvent(PaymentModel):
    """One verified provider notification, normalized for the application."""

    provider: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    outcome: PaymentEventOutcome = PaymentEventOutcome.OTHER
    checkout_session_id: str | None = None
    payment_intent_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    # Present on a subscription-mode checkout completion and on every
    # subscription lifecycle event.
    subscription: SubscriptionEventData | None = None
    #: When the provider says it happened (its settle time for a payment).
    occurred_at: datetime | None = None
    #: What a completed checkout actually collected, in minor units; zero
    #: when nothing was charged (e.g. a full discount).
    collected_minor: int | None = Field(default=None, ge=0)


class QuotePaymentRecord(PaymentModel):
    """One checkout attempt against one quote.

    ``checkout_session_id`` is unique: two accepts cannot create two payment
    rows for the same provider session.
    """

    payment_id: UUID
    business_id: BusinessId
    quote_id: QuoteId
    provider: str = Field(min_length=1)
    checkout_session_id: str = Field(min_length=1)
    checkout_url: str = Field(min_length=1)
    payment_intent_id: str | None = None
    expires_at: datetime | None = None
    amount_minor: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    status: QuotePaymentStatus = QuotePaymentStatus.OPEN
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at", "expires_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("payment timestamps must be timezone-aware")
        return value

    @property
    def is_open(self) -> bool:
        return self.status is QuotePaymentStatus.OPEN

    def is_expired(self, now: datetime) -> bool:
        """An open session the provider has already abandoned: unusable even
        though the provider never told us so directly."""

        return self.status is QuotePaymentStatus.OPEN and (
            self.expires_at is not None and self.expires_at <= now
        )

    def mark_paid(self, payment_intent_id: str | None, now: datetime) -> "QuotePaymentRecord":
        if self.status is QuotePaymentStatus.PAID:
            return self
        # An async attempt can fail first and still settle later, a pending
        # one settles once its delayed payment lands, and a session that
        # completed just before its deadline arrives at us already expired —
        # the provider's success is authoritative either way.
        if self.status not in (
            QuotePaymentStatus.OPEN,
            QuotePaymentStatus.PENDING,
            QuotePaymentStatus.FAILED,
            QuotePaymentStatus.EXPIRED,
        ):
            raise InvalidPaymentTransitionError(f"payment cannot be paid from {self.status}")
        return self.model_copy(
            update={
                "status": QuotePaymentStatus.PAID,
                "payment_intent_id": payment_intent_id or self.payment_intent_id,
                "updated_at": now,
            }
        )

    def mark_failed(self, now: datetime) -> "QuotePaymentRecord":
        if self.status is QuotePaymentStatus.PAID:
            return self
        return self.model_copy(update={"status": QuotePaymentStatus.FAILED, "updated_at": now})

    def mark_expired(self, now: datetime) -> "QuotePaymentRecord":
        """Retire an attempt the provider will no longer settle. Also closes a
        pending attempt superseded by a sibling that already collected."""

        if self.status not in (QuotePaymentStatus.OPEN, QuotePaymentStatus.PENDING):
            return self
        return self.model_copy(update={"status": QuotePaymentStatus.EXPIRED, "updated_at": now})

    def mark_pending(self, now: datetime) -> "QuotePaymentRecord":
        """Checkout completed but the payment is still settling: the attempt
        can no longer expire and resolves only as paid or failed."""

        if self.status is not QuotePaymentStatus.OPEN:
            return self
        return self.model_copy(update={"status": QuotePaymentStatus.PENDING, "updated_at": now})


class InvalidPaymentTransitionError(ValueError):
    pass


class QuotePaymentConflictError(RuntimeError):
    """A payment row for the same checkout session already exists."""


class QuotePaymentRepository(Protocol):
    async def find_open(
        self, business_id: BusinessId, quote_id: QuoteId
    ) -> QuotePaymentRecord | None:
        """The newest attempt still in play — open, or completed and awaiting
        asynchronous settlement; ``is_expired`` decides whether an open one is
        still usable."""
        ...

    async def find_by_checkout_session(
        self, checkout_session_id: str
    ) -> QuotePaymentRecord | None: ...

    async def count_for_quote(self, business_id: BusinessId, quote_id: QuoteId) -> int: ...

    async def create(self, record: QuotePaymentRecord) -> None:
        """Raises :class:`QuotePaymentConflictError` when the session id is taken."""
        ...

    async def save(
        self,
        record: QuotePaymentRecord,
        *,
        expected_from: QuotePaymentStatus,
    ) -> None:
        """Write the transition only if the row still holds ``expected_from``;
        a concurrent webhook that moved it first raises
        :class:`QuotePaymentConflictError` so the caller's event rolls back and
        is retried against fresh state."""
        ...


class QuoteSubscriptionRecord(PaymentModel):
    """The recurring agreement a paid recurring quote turned into."""

    subscription_id: SubscriptionId
    business_id: BusinessId
    quote_id: QuoteId
    # None only for a manual plan whose quote has no customer e-mail.
    customer_id: CustomerId | None
    provider: str = Field(min_length=1)
    subscription_ref: str = Field(min_length=1)
    status: str = Field(min_length=1)
    interval: BillingInterval
    amount_minor: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False
    created_at: datetime
    updated_at: datetime
    #: Manual plans only: the first day paid for, and the last day covered.
    paid_from: date | None = None
    paid_through: date | None = None

    @property
    def is_manual(self) -> bool:
        return self.provider == MANUAL_PROVIDER

    @property
    def is_live(self) -> bool:
        """Still billing or covering the customer: not ended."""

        return self.status not in ENDED_SUBSCRIPTION_STATUSES

    @field_validator("created_at", "updated_at", "current_period_end")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("subscription timestamps must be timezone-aware")
        return value

    def apply(self, data: SubscriptionEventData, now: datetime) -> "QuoteSubscriptionRecord":
        """Fold the fields an event carried into the row."""

        update: dict[str, object] = {"updated_at": now}
        if data.status is not None:
            update["status"] = data.status
        if data.interval is not None:
            update["interval"] = data.interval
        if data.amount_minor is not None:
            update["amount_minor"] = data.amount_minor
        if data.currency is not None:
            update["currency"] = data.currency.upper()
        if data.current_period_end is not None:
            update["current_period_end"] = data.current_period_end
        if data.cancel_at_period_end is not None:
            update["cancel_at_period_end"] = data.cancel_at_period_end
        return self.model_copy(update=update)


class QuoteSubscriptionRepository(Protocol):
    async def find_by_subscription_ref(
        self, subscription_ref: str
    ) -> QuoteSubscriptionRecord | None: ...

    async def list_for_customer(
        self, business_id: BusinessId, customer_id: CustomerId
    ) -> tuple[QuoteSubscriptionRecord, ...]: ...

    async def list_for_business(
        self, business_id: BusinessId
    ) -> tuple[QuoteSubscriptionRecord, ...]:
        """Newest first."""
        ...

    async def create(self, record: QuoteSubscriptionRecord) -> None:
        """Raises :class:`QuotePaymentConflictError` when the provider
        subscription is already recorded."""
        ...

    async def save(self, record: QuoteSubscriptionRecord) -> None: ...

    async def list_for_quote(
        self, business_id: BusinessId, quote_id: QuoteId
    ) -> tuple[QuoteSubscriptionRecord, ...]: ...

    async def get(
        self,
        business_id: BusinessId,
        subscription_id: SubscriptionId,
        *,
        for_update: bool = False,
    ) -> QuoteSubscriptionRecord | None: ...


class PaymentEventRepository(Protocol):
    """Replay ledger for provider webhooks: first record wins, so a retried
    event is answered from what the first delivery already did."""

    async def try_record(self, provider: str, event_id: str, now: datetime) -> bool:
        """True when this delivery recorded the event; False on a replay."""
        ...


class PaymentSource(StrEnum):
    STRIPE = "stripe"
    MANUAL = "manual"


class PaymentMethod(StrEnum):
    CARD = "card"
    CHECK = "check"
    CASH = "cash"
    OTHER = "other"


class PaymentKind(StrEnum):
    """A one-off quote is paid once; a plan is paid again every period."""

    ONE_OFF = "one_off"
    PLAN = "plan"


class LedgerPayment(PaymentModel):
    """One settled payment: money that arrived, on the day it arrived.

    ``reference`` is unique per ``source``: the provider's own id (checkout
    session, renewal invoice) for card payments, the form's one-time key for
    manual ones. A second active one-off payment for the same quote is kept
    but flagged ``duplicate`` and left out of totals.
    """

    payment_id: UUID
    business_id: BusinessId
    quote_id: QuoteId
    kind: PaymentKind
    source: PaymentSource
    method: PaymentMethod
    reference: str = Field(min_length=1, max_length=255)
    amount_minor: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    paid_at: datetime
    months_covered: int | None = Field(default=None, ge=1)
    recorded_by: str | None = None
    recorded_at: datetime
    note: str | None = Field(default=None, max_length=500)
    voided_at: datetime | None = None
    voided_by: str | None = None
    duplicate: bool = False
    # Where "Mark unpaid" puts the quote back: what the customer had done
    # before the owner marked it paid. None on rows that predate it.
    customer_status_before: Literal["sent", "viewed", "accepted"] | None = None

    @field_validator("currency")
    @classmethod
    def upper_currency(cls, value: str) -> str:
        return value.upper()

    @field_validator("paid_at", "recorded_at", "voided_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("payment timestamps must be timezone-aware")
        return value

    @property
    def counts(self) -> bool:
        return self.voided_at is None and not self.duplicate


class PaymentLedgerRepository(Protocol):
    async def record(self, payment: LedgerPayment) -> LedgerPayment | None:
        """Add the payment; ``None`` when its ``(source, reference)`` is
        already recorded. A second active one-off payment for the quote is
        stored with ``duplicate`` set and returned that way."""
        ...

    async def list_for_business(
        self, business_id: BusinessId, *, since: datetime | None = None
    ) -> tuple[LedgerPayment, ...]:
        """Newest first by ``paid_at``."""
        ...

    async def list_for_quote(
        self, business_id: BusinessId, quote_id: QuoteId
    ) -> tuple[LedgerPayment, ...]:
        """Oldest first by ``paid_at``."""
        ...

    async def void_manual(
        self, business_id: BusinessId, payment_id: UUID, *, by: str, at: datetime
    ) -> bool:
        """Void a manual payment that is not already voided; ``False`` when
        there is none (Stripe payments are never voided here)."""
        ...

    async def mark_duplicate(
        self, business_id: BusinessId, payment_id: UUID, duplicate: bool
    ) -> None: ...


def months_for(interval: BillingInterval | None) -> int | None:
    if interval is None:
        return None
    return 12 if interval is BillingInterval.YEAR else 1


def add_months(day: date, months: int) -> date:
    """The same day ``months`` later, clamped to the end of a short month."""

    index = day.month - 1 + months
    year, month = day.year + index // 12, index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def paid_through(paid_from: date, months: int) -> date:
    """The last day covered by ``months`` paid from ``paid_from``."""

    return add_months(paid_from, months) - timedelta(days=1)


def plan_nudge_command(record: QuoteSubscriptionRecord, *, at: datetime, key: str) -> OutboxCommand:
    """Remind the owner that a manual plan runs out soon; the worker drops
    it when the plan moved on (paid again, undone or ended) since."""

    assert record.paid_through is not None
    dedup = f"plan-nudge:{record.subscription_id}:{record.paid_through.isoformat()}:{key}"
    return OutboxCommand(
        command_id=OutboxCommandId(uuid5(PLAN_NUDGE_COMMAND_NAMESPACE, dedup)),
        business_id=record.business_id,
        command_type=PLAN_NUDGE_COMMAND_TYPE,
        payload={
            "subscription_id": str(record.subscription_id),
            "paid_through": record.paid_through.isoformat(),
        },
        dedup_key=dedup,
        not_before=at,
    )


def month_totals(payments: Iterable[LedgerPayment], zone: tzinfo, now: datetime) -> dict[str, int]:
    """Money that arrived in ``now``'s calendar month in ``zone``, per
    currency, before fees; voided and duplicate payments don't count."""

    current = now.astimezone(zone)
    totals: dict[str, int] = {}
    for payment in payments:
        local = payment.paid_at.astimezone(zone)
        if payment.counts and (local.year, local.month) == (current.year, current.month):
            totals[payment.currency] = totals.get(payment.currency, 0) + payment.amount_minor
    return totals


def first_counted(payments: Iterable[LedgerPayment]) -> dict[QuoteId, LedgerPayment]:
    """Each quote's earliest counted payment: when, and how, it was paid."""

    counted: dict[QuoteId, LedgerPayment] = {}
    for payment in payments:
        current = counted.get(payment.quote_id)
        if payment.counts and (current is None or payment.paid_at < current.paid_at):
            counted[payment.quote_id] = payment
    return counted


def first_paid_at(payments: Iterable[LedgerPayment]) -> dict[QuoteId, datetime]:
    """Each quote's earliest counted payment: when it was paid."""

    paid: dict[QuoteId, datetime] = {}
    for payment in payments:
        if payment.counts and (
            payment.quote_id not in paid or payment.paid_at < paid[payment.quote_id]
        ):
            paid[payment.quote_id] = payment.paid_at
    return paid
