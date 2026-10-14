"""Stripe webhook payloads: the checkout-session, invoice and subscription
events GVAS consumes.

Verification happens in ``signature.py`` before this parsing runs; this module
only normalizes the JSON into the provider-neutral ``PaymentWebhookEvent``.
Checkout events name the session; invoice and subscription events name the
subscription, and carry the metadata Stripe copied from the checkout so an
unknown subscription can still be recognised as (not) one of ours.
"""

import json
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from gvas.domain.enums import BillingInterval
from gvas.domain.payments import (
    STRIPE_PROVIDER,
    PaymentEventOutcome,
    PaymentWebhookEvent,
    SubscriptionEventData,
)

CHECKOUT_COMPLETED: Final = "checkout.session.completed"
CHECKOUT_ASYNC_SUCCEEDED: Final = "checkout.session.async_payment_succeeded"
CHECKOUT_ASYNC_FAILED: Final = "checkout.session.async_payment_failed"
INVOICE_PAID: Final = "invoice.paid"
INVOICE_PAYMENT_FAILED: Final = "invoice.payment_failed"
SUBSCRIPTION_UPDATED: Final = "customer.subscription.updated"
SUBSCRIPTION_DELETED: Final = "customer.subscription.deleted"

#: ``payment_status`` values meaning the session's money has settled.
SETTLED_PAYMENT_STATUSES: Final = frozenset({"paid", "no_payment_required"})
#: Events whose payment has settled without needing a status check.
PAYMENT_SUCCEEDED_EVENTS: Final = frozenset({CHECKOUT_COMPLETED, CHECKOUT_ASYNC_SUCCEEDED})
#: Subscription lifecycle events and the outcome each maps to.
SUBSCRIPTION_EVENTS: Final[dict[str, PaymentEventOutcome]] = {
    INVOICE_PAID: PaymentEventOutcome.SUBSCRIPTION_RENEWED,
    INVOICE_PAYMENT_FAILED: PaymentEventOutcome.SUBSCRIPTION_PAYMENT_FAILED,
    SUBSCRIPTION_UPDATED: PaymentEventOutcome.SUBSCRIPTION_UPDATED,
    SUBSCRIPTION_DELETED: PaymentEventOutcome.SUBSCRIPTION_CANCELLED,
}
#: Every event type the webhook endpoint subscribes to.
HANDLED_EVENTS: Final = frozenset(
    {CHECKOUT_COMPLETED, CHECKOUT_ASYNC_SUCCEEDED, CHECKOUT_ASYNC_FAILED, *SUBSCRIPTION_EVENTS}
)
#: The first invoice of a subscription is settled by the checkout itself; it is
#: not a renewal, so it updates the row without a renewal notice.
INITIAL_INVOICE_REASONS: Final = frozenset({"subscription_create"})


class StripeEventError(ValueError):
    pass


class _StripeRecurring(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    interval: str | None = None


class _StripePrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    unit_amount: int | None = None
    currency: str | None = None
    recurring: _StripeRecurring | None = None


class _StripeSubscriptionItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    price: _StripePrice | None = None
    quantity: int | None = None


class _StripeItemList(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_StripeSubscriptionItem, ...] = ()


class _StripeInvoiceLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    period: dict[str, int] = {}


class _StripeInvoiceLines(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_StripeInvoiceLine, ...] = ()


class _StripeSubscriptionDetails(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    metadata: dict[str, str] = {}


class _StripeParent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    subscription_details: _StripeSubscriptionDetails | None = None


class _StripeObject(BaseModel):
    """The union of the fields GVAS reads from a checkout session, an invoice
    and a subscription; every one is optional because each event type fills
    a different subset."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    object: str | None = None
    mode: str | None = None
    payment_intent: str | None = None
    payment_status: str | None = None
    metadata: dict[str, str] = {}
    customer: str | None = None
    # checkout session (mode=subscription) and invoice
    subscription: str | None = None
    subscription_details: _StripeSubscriptionDetails | None = None
    parent: _StripeParent | None = None
    billing_reason: str | None = None
    amount_paid: int | None = None
    amount_total: int | None = None
    amount_due: int | None = None
    currency: str | None = None
    lines: _StripeInvoiceLines | None = None
    # subscription
    status: str | None = None
    items: _StripeItemList | None = None
    current_period_end: int | None = None
    cancel_at_period_end: bool | None = None
    canceled_at: int | None = None


class _StripeEventData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    object: _StripeObject


class _StripeEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    type: str
    created: int | None = None
    data: _StripeEventData


def parse_checkout_event(body: bytes) -> PaymentWebhookEvent:
    """Decode one event body; payloads GVAS does not handle parse as OTHER."""

    try:
        parsed = _StripeEvent.model_validate(json.loads(body))
    except (ValueError, ValidationError) as error:
        raise StripeEventError("stripe event is not readable") from error
    record = parsed.data.object
    if parsed.type in SUBSCRIPTION_EVENTS:
        return _subscription_event(parsed.id, parsed.type, record).model_copy(
            update={"occurred_at": _timestamp(parsed.created)}
        )
    if parsed.type == CHECKOUT_COMPLETED and (
        record.payment_status not in SETTLED_PAYMENT_STATUSES
    ):
        # A completed session can still be unpaid when the customer chose a
        # delayed method; the async outcome event carries the real answer, so
        # the attempt is pinned rather than left expirable.
        outcome = PaymentEventOutcome.PENDING
    elif parsed.type in PAYMENT_SUCCEEDED_EVENTS:
        outcome = PaymentEventOutcome.SUCCEEDED
    elif parsed.type == CHECKOUT_ASYNC_FAILED:
        outcome = PaymentEventOutcome.FAILED
    else:
        outcome = PaymentEventOutcome.OTHER
    subscription = None
    if record.mode == "subscription" and record.subscription:
        subscription = SubscriptionEventData(
            subscription_ref=record.subscription, customer_ref=record.customer
        )
    return PaymentWebhookEvent(
        provider=STRIPE_PROVIDER,
        event_id=parsed.id,
        event_type=parsed.type,
        outcome=outcome,
        checkout_session_id=record.id or None,
        payment_intent_id=record.payment_intent,
        metadata=record.metadata,
        subscription=subscription,
        occurred_at=_timestamp(parsed.created),
        collected_minor=record.amount_total if parsed.type in PAYMENT_SUCCEEDED_EVENTS else None,
    )


def _subscription_event(
    event_id: str, event_type: str, record: _StripeObject
) -> PaymentWebhookEvent:
    outcome = SUBSCRIPTION_EVENTS[event_type]
    if record.object == "invoice" or event_type.startswith("invoice."):
        subscription_ref = record.subscription
        details = record.subscription_details or (
            record.parent.subscription_details if record.parent is not None else None
        )
        metadata = details.metadata if details is not None else {}
        if subscription_ref is None:
            # An invoice unrelated to any subscription (one-off invoicing in
            # the same account) is nothing GVAS tracks.
            return PaymentWebhookEvent(
                provider=STRIPE_PROVIDER,
                event_id=event_id,
                event_type=event_type,
                outcome=PaymentEventOutcome.OTHER,
                metadata=metadata,
            )
        if outcome is PaymentEventOutcome.SUBSCRIPTION_RENEWED and (
            record.billing_reason in INITIAL_INVOICE_REASONS
        ):
            outcome = PaymentEventOutcome.SUBSCRIPTION_UPDATED
        period_end = None
        if record.lines is not None:
            ends = [line.period.get("end") for line in record.lines.data if "end" in line.period]
            if ends:
                period_end = _timestamp(max(end for end in ends if end is not None))
        paid = record.amount_paid if outcome is PaymentEventOutcome.SUBSCRIPTION_RENEWED else None
        data = SubscriptionEventData(
            subscription_ref=subscription_ref,
            customer_ref=record.customer,
            paid_minor=paid,
            currency=record.currency,
            current_period_end=period_end,
            invoice_ref=record.id,
        )
    else:
        metadata = record.metadata
        interval = None
        amount = None
        currency = None
        if record.items is not None and record.items.data:
            total = 0
            for item in record.items.data:
                price = item.price
                if price is None:
                    continue
                if price.recurring is not None and price.recurring.interval in {
                    member.value for member in BillingInterval
                }:
                    interval = BillingInterval(price.recurring.interval)
                if price.unit_amount is not None:
                    total += price.unit_amount * (item.quantity or 1)
                currency = currency or price.currency
            amount = total if total else None
        status = record.status
        if outcome is PaymentEventOutcome.SUBSCRIPTION_CANCELLED:
            status = "canceled"
        data = SubscriptionEventData(
            subscription_ref=record.id,
            customer_ref=record.customer,
            status=status,
            interval=interval,
            amount_minor=amount,
            currency=currency,
            current_period_end=_timestamp(record.current_period_end),
            cancel_at_period_end=record.cancel_at_period_end,
        )
    return PaymentWebhookEvent(
        provider=STRIPE_PROVIDER,
        event_id=event_id,
        event_type=event_type,
        outcome=outcome,
        metadata=metadata,
        subscription=data,
    )


def _timestamp(value: int | None) -> datetime | None:
    return None if value is None else datetime.fromtimestamp(value, UTC)
