"""Public quote use cases: view, accept, decline and hosted checkout.

Everything here is scoped by claim token, never by business id: a token
resolves to exactly one quote and every record it touches stays inside that
quote's business. Tokens are looked up by their SHA-256 hash and compared in
constant time so the raw token is the only secret.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from gvas.domain.customer_linking import enqueue_quote_owner_notice, link_quote_customer
from gvas.domain.customers import CustomerRecord
from gvas.domain.enums import CustomerQuoteStatus, QuotePaymentStatus
from gvas.domain.identifiers import BusinessId, SubscriptionId
from gvas.domain.money import format_money
from gvas.domain.payments import (
    STRIPE_PROVIDER,
    BillingCustomerRequest,
    LedgerPayment,
    PaymentCheckoutRequest,
    PaymentEventOutcome,
    PaymentKind,
    PaymentLineItem,
    PaymentMethod,
    PaymentSource,
    PaymentWebhookEvent,
    QuotePaymentConflictError,
    QuotePaymentRecord,
    QuoteSubscriptionRecord,
    SubscriptionEventData,
    checkout_expire_command,
    months_for,
)
from gvas.domain.ports import BillingAccountPort, PaymentCheckoutPort
from gvas.domain.quotes import (
    InvalidQuoteTransitionError,
    Quote,
    QuoteConcurrencyError,
    QuoteDraftProposal,
    claim_token_matches,
    delivery_line_items,
    hash_claim_token,
    public_quote_id,
)
from gvas.domain.repositories import BusinessRecord, UnitOfWork

logger = logging.getLogger(__name__)


class QuoteNotFoundError(ValueError):
    """The token did not cleanly identify one claimable quote; shown as 404."""


class OpenCheckoutUnavailableError(RuntimeError):
    """Accepting is configured off: no checkout provider is wired."""


class UnknownPaymentSessionError(RuntimeError):
    """A webhook named a checkout session (or a subscription born from one
    of our checkouts) we have not recorded yet.

    The provider can deliver its event before ``accept_quote`` commits the
    session row; answering 2xx would consume the event forever, so the
    endpoint answers 503 and the provider's retry lands after the record
    exists.
    """


#: Metadata key stamped on every checkout and copied by the provider onto the
#: subscription and its invoices; its presence says a record is ours.
QUOTE_METADATA_KEY = "gvas_quote_id"
#: Metadata key naming the deployment that created a checkout, subscription or
#: customer. Deployments sharing one provider account each act only on their
#: own events; records from before the tag existed belong to production.
DEPLOYMENT_METADATA_KEY = "gvas_deployment"
DEFAULT_DEPLOYMENT = "production"


class PublicQuoteView:
    """The customer-facing projection of one hosted quote: only fields a
    customer may see — no internal ids beyond the public opaque id and no
    owner contact details."""

    def __init__(self, business: BusinessRecord, quote: Quote) -> None:
        draft = quote.draft
        if draft is None:
            raise QuoteNotFoundError("quote is not ready")
        self.business_display_name = business.display_name or business.name
        self.business_site_url = business.site_url or ""
        self.quote_public_id = public_quote_id(quote.quote_id)
        self.quote_status = (
            quote.customer_status.value
            if quote.customer_status is not None
            else CustomerQuoteStatus.VIEWED.value
        )
        self.customer_name = draft.recipient.display_name
        self.service_address = draft.recipient.service_address
        self.items = [
            {
                "description": item.description,
                "quantity": item.quantity,
                "amountCents": item.quantity * item.unit_price_minor,
            }
            for item in delivery_line_items(draft)
        ]
        self.subtotal_cents = draft.subtotal_minor
        self.total_cents = draft.total_minor
        self.currency = draft.currency
        self.note = draft.owner_note
        self.billing = draft.billing.value
        self.interval = None if draft.interval is None else draft.interval.value
        self.created_at = quote.created_at
        self.approved_at = quote.approved_at

    def as_payload(self) -> dict[str, object]:
        return {
            "business": {
                "displayName": self.business_display_name,
                "siteUrl": self.business_site_url,
            },
            "quote": {
                "id": self.quote_public_id,
                "status": self.quote_status,
                "customerName": self.customer_name,
                "serviceAddress": self.service_address,
                "items": self.items,
                "subtotalCents": self.subtotal_cents,
                "totalCents": self.total_cents,
                "currency": self.currency,
                "note": self.note,
                "billing": self.billing,
                "interval": self.interval,
                "createdAt": self.created_at.isoformat(),
                "approvedAt": (
                    self.approved_at.isoformat() if self.approved_at is not None else None
                ),
            },
        }


def checkout_line_items(draft: QuoteDraftProposal) -> tuple[PaymentLineItem, ...]:
    """The draft as checkout line items.

    Checkout providers cannot take negative-priced lines, so when a discount
    is in play the whole quote collapses into a single total line — the charge
    then still equals ``draft.total_minor``. Tax travels as its own positive
    line.
    """

    items = delivery_line_items(draft)
    if any(item.unit_price_minor < 0 for item in items):
        return (
            PaymentLineItem(description="Quote total", quantity=1, amount_minor=draft.total_minor),
        )
    return tuple(
        PaymentLineItem(
            description=item.description,
            quantity=item.quantity,
            amount_minor=item.unit_price_minor,
        )
        for item in items
    )


class PublicQuoteService:
    """Runs the customer-facing use cases inside the unit of work."""

    def __init__(
        self,
        unit_of_work_factory: Callable[[], UnitOfWork],
        checkout: PaymentCheckoutPort | None = None,
        billing_accounts: BillingAccountPort | None = None,
        deployment: str = DEFAULT_DEPLOYMENT,
        payments_off: Callable[[BusinessId], Awaitable[bool]] | None = None,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._checkout = checkout
        # Businesses that never take a card (demo sandboxes).
        self._payments_off = payments_off
        self._billing_accounts = billing_accounts
        self._deployment = deployment

    async def fetch_quote(self, claim_token: str) -> PublicQuoteView:
        """Return the projection; the first successful fetch marks the quote
        viewed and repeat fetches change nothing."""

        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_claimable(unit_of_work, claim_token)
            business = await self._hosted_business(unit_of_work, quote.business_id)
            viewed = quote.record_customer_view(_now())
            conflicted = False
            if viewed is not quote:
                try:
                    await unit_of_work.quotes.save(viewed, expected_version=quote.version)
                except QuoteConcurrencyError:
                    # A concurrent transition (accept, decline, payment) won;
                    # the committed state is the answer, not a stale "viewed".
                    conflicted = True
            await unit_of_work.commit()
            if not conflicted:
                return PublicQuoteView(business, viewed)
        async with self._unit_of_work_factory() as unit_of_work:
            fresh = await self._find_claimable(unit_of_work, claim_token)
            business = await self._hosted_business(unit_of_work, fresh.business_id)
            return PublicQuoteView(business, fresh)

    async def accept_quote(self, claim_token: str) -> str:
        """Return the hosted checkout URL, opening a session on first accept.

        While an open session stands, repeat accepts return its URL. The
        provider call happens between two transactions: the first records the
        customer status, the second records the session — a crash or a racing
        accept reuses the same provider idempotency key, so the same session
        comes back and the unique ``checkout_session_id`` settles it. A lost
        optimistic write answers 409 instead of continuing from stale state:
        a racing decline must not produce a payable session, and the retried
        accept simply returns the open one a winning accept already made.
        """

        if self._checkout is None:
            raise OpenCheckoutUnavailableError("checkout is not configured")
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_claimable(unit_of_work, claim_token)
            if self._payments_off is not None and await self._payments_off(quote.business_id):
                raise OpenCheckoutUnavailableError("payments are off for this business")
            accepted = quote.record_customer_accept(_now())
            business = await self._hosted_business(unit_of_work, quote.business_id)
            open_payment = await unit_of_work.quote_payments.find_open(
                quote.business_id, quote.quote_id
            )
            if open_payment is not None and open_payment.is_expired(_now()):
                # The provider abandons unpaid sessions without telling us;
                # retire the attempt and open a fresh one below.
                await unit_of_work.quote_payments.save(
                    open_payment.mark_expired(_now()),
                    expected_from=open_payment.status,
                )
                open_payment = None
            if open_payment is not None:
                if accepted is not quote:
                    await self._save_customer_move(unit_of_work, quote, accepted)
                await unit_of_work.commit()
                return open_payment.checkout_url
            draft = accepted.draft
            if draft is None or accepted.claim_token is None:
                raise QuoteNotFoundError("quote is not ready")
            customer: CustomerRecord | None = None
            if draft.is_recurring:
                if self._billing_accounts is None:
                    raise OpenCheckoutUnavailableError("subscription checkout is not configured")
                accepted, customer = await link_quote_customer(unit_of_work, accepted, _now())
                if customer is None:
                    raise QuoteNotFoundError("recurring quotes need an e-mail recipient")
            if accepted is not quote:
                await self._save_customer_move(unit_of_work, quote, accepted)
            quote_url = accepted.quote_url(business.site_url or "")
            items = checkout_line_items(draft)
            currency = draft.currency
            # One idempotency scope per attempt: the first accept reuses the
            # delivery key; a regenerated session (the previous one expired)
            # must not replay the provider's earlier answer.
            attempt = await unit_of_work.quote_payments.count_for_quote(
                quote.business_id, quote.quote_id
            )
            idempotency_key = (
                f"quote-delivery:{quote.quote_id}"
                if attempt == 0
                else f"quote-delivery:{quote.quote_id}:attempt-{attempt + 1}"
            )
            metadata = {
                QUOTE_METADATA_KEY: public_quote_id(quote.quote_id),
                "business_id": str(quote.business_id),
                DEPLOYMENT_METADATA_KEY: self._deployment,
            }
            await unit_of_work.commit()
        customer_ref: str | None = None
        if customer is not None:
            customer_ref = await self._ensure_billing_customer(customer, metadata)
        request = PaymentCheckoutRequest(
            business_id=quote.business_id,
            quote_id=quote.quote_id,
            client_reference=public_quote_id(quote.quote_id),
            currency=currency,
            line_items=items,
            success_url=f"{quote_url}?paid=1",
            cancel_url=quote_url,
            idempotency_key=idempotency_key,
            metadata=metadata,
            recurring_interval=draft.interval,
            customer_ref=customer_ref,
        )
        result = await self._checkout.create_checkout(request)
        async with self._unit_of_work_factory() as unit_of_work:
            record = QuotePaymentRecord(
                payment_id=uuid4(),
                business_id=quote.business_id,
                quote_id=quote.quote_id,
                provider=STRIPE_PROVIDER,
                checkout_session_id=result.session_id,
                checkout_url=result.checkout_url,
                payment_intent_id=result.payment_intent_id,
                expires_at=result.expires_at,
                amount_minor=draft.total_minor,
                currency=currency,
                created_at=_now(),
                updated_at=_now(),
            )
            try:
                await unit_of_work.quote_payments.create(record)
            except QuotePaymentConflictError:
                existing = await unit_of_work.quote_payments.find_by_checkout_session(
                    result.session_id
                )
                if existing is None:
                    raise
                # A concurrent accept recorded this session; it already
                # closed it if the quote stopped being payable.
                current = await self._find_claimable(unit_of_work, claim_token)
                if (
                    current.customer_status is not CustomerQuoteStatus.ACCEPTED
                    or existing.status is not QuotePaymentStatus.OPEN
                ):
                    raise InvalidQuoteTransitionError(
                        "the quote changed while checkout opened"
                    ) from None
                await unit_of_work.commit()
                return existing.checkout_url
            # The owner may have marked the quote paid, or the customer
            # declined, while the session opened. Bumping the version
            # serialises against those writes.
            current = await self._find_claimable(unit_of_work, claim_token)
            paid_meanwhile = current.customer_status is not CustomerQuoteStatus.ACCEPTED
            if not paid_meanwhile:
                try:
                    await unit_of_work.quotes.save(
                        current.model_copy(update={"version": current.version + 1}),
                        expected_version=current.version,
                    )
                except QuoteConcurrencyError:
                    paid_meanwhile = True
            if paid_meanwhile:
                await unit_of_work.quote_payments.save(
                    record.mark_expired(_now()), expected_from=record.status
                )
                await unit_of_work.outbox.enqueue(
                    checkout_expire_command(quote.business_id, result.session_id)
                )
                await unit_of_work.commit()
                raise InvalidQuoteTransitionError("the quote changed while checkout opened")
            await unit_of_work.commit()
            return record.checkout_url

    async def _ensure_billing_customer(
        self, customer: CustomerRecord, metadata: dict[str, str]
    ) -> str:
        """The provider-side customer for this (business, customer), created on
        first use. The idempotency key is the customer id, so a crash between
        the provider call and the write yields the same provider customer."""

        if customer.stripe_customer_id is not None:
            return customer.stripe_customer_id
        if self._billing_accounts is None:
            raise OpenCheckoutUnavailableError("subscription checkout is not configured")
        created = await self._billing_accounts.create_customer(
            BillingCustomerRequest(
                business_id=customer.business_id,
                customer_id=customer.customer_id,
                email=customer.email,
                name=customer.display_name,
                phone=customer.phone,
                idempotency_key=f"billing-customer:{customer.customer_id}",
                metadata={
                    "business_id": str(customer.business_id),
                    "gvas_customer_id": str(customer.customer_id),
                    DEPLOYMENT_METADATA_KEY: self._deployment,
                },
            )
        )
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.customers.set_stripe_customer_id(
                customer.business_id, customer.customer_id, created.customer_ref
            )
            fresh = await unit_of_work.customers.get(customer.business_id, customer.customer_id)
            await unit_of_work.commit()
        if fresh is not None and fresh.stripe_customer_id is not None:
            return fresh.stripe_customer_id
        return created.customer_ref

    async def decline_quote(self, claim_token: str) -> str:
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_claimable(unit_of_work, claim_token)
            await self._hosted_business(unit_of_work, quote.business_id)
            declined = quote.record_customer_decline(_now())
            if declined is not quote:
                try:
                    await unit_of_work.quotes.save(declined, expected_version=quote.version)
                except QuoteConcurrencyError as error:
                    # The quote moved underneath this request (a racing accept
                    # or a payment); reporting a decline that did not persist
                    # would lie, so the caller retries against fresh state.
                    raise InvalidQuoteTransitionError(
                        "the quote changed while declining; retry the request"
                    ) from error
            await unit_of_work.commit()
            return (
                declined.customer_status.value
                if declined.customer_status is not None
                else CustomerQuoteStatus.DECLINED.value
            )

    async def booking_link(self, public_key: str) -> dict[str, str]:
        async with self._unit_of_work_factory() as unit_of_work:
            business = await unit_of_work.businesses.get_by_public_key(public_key)
            await unit_of_work.commit()
        if business is None or business.calendly_url is None:
            raise QuoteNotFoundError("unknown business")
        return {
            "calendlyUrl": business.calendly_url,
            "displayName": business.display_name or business.name,
        }

    async def record_payment_event(self, event: PaymentWebhookEvent) -> bool:
        """Apply one verified payment event; False when the event id was
        already recorded so the caller answers without redoing work.

        The dedup row, the payment transition, the quote's customer status and
        the owner notice all land in one transaction, so a mid-failure rolls
        the recording back and the provider's retry replays cleanly.

        An event tagged for another deployment sharing the provider account
        is answered as ignored, never retried and never recorded.
        """

        owner = event.metadata.get(DEPLOYMENT_METADATA_KEY, DEFAULT_DEPLOYMENT)
        if owner != self._deployment:
            logger.info("ignoring %s for deployment %r", event.event_type, owner)
            return False
        if event.outcome is PaymentEventOutcome.OTHER:
            return False
        async with self._unit_of_work_factory() as unit_of_work:
            recorded = await unit_of_work.payment_events.try_record(
                event.provider, event.event_id, _now()
            )
            if not recorded:
                await unit_of_work.commit()
                return False
            if event.outcome in SUBSCRIPTION_OUTCOMES:
                await self._apply_subscription_event(unit_of_work, event)
                await unit_of_work.commit()
                return True
            if event.checkout_session_id is None:
                await unit_of_work.commit()
                return True
            payment = await unit_of_work.quote_payments.find_by_checkout_session(
                event.checkout_session_id
            )
            if payment is None:
                # Raise instead of committing: the session row may still be on
                # its way (accept commits it after the provider call), and a
                # failed response lets the provider retry once it exists.
                raise UnknownPaymentSessionError(event.checkout_session_id)
            now = _now()
            if event.outcome is PaymentEventOutcome.PENDING:
                # Checkout completed but a delayed payment is still settling:
                # pin the attempt so expiry cannot retire it in between.
                await unit_of_work.quote_payments.save(
                    payment.mark_pending(now), expected_from=payment.status
                )
                await unit_of_work.commit()
                return True
            if event.outcome is PaymentEventOutcome.FAILED:
                await unit_of_work.quote_payments.save(
                    payment.mark_failed(now), expected_from=payment.status
                )
                await unit_of_work.commit()
                return True
            if payment.status is not QuotePaymentStatus.PAID:
                await unit_of_work.quote_payments.save(
                    payment.mark_paid(event.payment_intent_id, now),
                    expected_from=payment.status,
                )
            superseded = await unit_of_work.quote_payments.find_open(
                payment.business_id, payment.quote_id
            )
            if superseded is not None and superseded.payment_id != payment.payment_id:
                # A session the customer completed late may have been replaced
                # by a regenerated checkout; close the sibling out so two
                # sessions for one quote are not both payable.
                await unit_of_work.quote_payments.save(
                    superseded.mark_expired(now), expected_from=superseded.status
                )
            quote = await unit_of_work.quotes.get(payment.business_id, payment.quote_id)
            if quote is None:
                await unit_of_work.commit()
                return True
            draft = quote.draft
            collected = (
                payment.amount_minor if event.collected_minor is None else event.collected_minor
            )
            # A card plan that raced a manual one: keep both, count the
            # check, and let the owner decide which to end.
            manual_plan = None
            if event.subscription is not None:
                manual_plan = next(
                    (
                        plan
                        for plan in await unit_of_work.quote_subscriptions.list_for_quote(
                            payment.business_id, payment.quote_id
                        )
                        if plan.is_manual and plan.is_live
                    ),
                    None,
                )
            if collected > 0:
                # A zero-charge checkout (e.g. a full discount) settles the
                # quote but collected no money, so it adds nothing to the ledger.
                ledger_row = await unit_of_work.payments.record(
                    LedgerPayment(
                        payment_id=uuid4(),
                        business_id=payment.business_id,
                        quote_id=payment.quote_id,
                        kind=(
                            PaymentKind.ONE_OFF if event.subscription is None else PaymentKind.PLAN
                        ),
                        source=PaymentSource.STRIPE,
                        method=PaymentMethod.CARD,
                        reference=payment.checkout_session_id,
                        amount_minor=collected,
                        currency=payment.currency,
                        paid_at=event.occurred_at or now,
                        months_covered=(
                            None
                            if event.subscription is None or draft is None
                            else months_for(draft.interval)
                        ),
                        recorded_at=now,
                        duplicate=manual_plan is not None,
                    )
                )
                if ledger_row is not None and event.subscription is None:
                    await self._flag_paid_twice(unit_of_work, quote, ledger_row)
            if manual_plan is not None:
                name = draft.recipient.display_name if draft is not None else None
                await self._enqueue_owner_notice(
                    unit_of_work,
                    quote,
                    correlation_id=f"quote:{quote.quote_id}:card-plan-over-manual",
                    text=(
                        f"Quote {public_quote_id(quote.quote_id)} for {name or 'customer'}"
                        f" also started a card plan ({format_money(collected, payment.currency)}"
                        " by card) after you recorded a check or cash payment. Stripe will keep"
                        " billing the card: cancel and refund it in Stripe, or undo the manual"
                        " payments in the dashboard."
                    ),
                )
            paid = quote.record_customer_payment(now)
            if event.subscription is not None:
                paid, _ = await link_quote_customer(unit_of_work, paid, now)
            if paid is not quote:
                await unit_of_work.quotes.save(paid, expected_version=quote.version)
            if event.subscription is not None:
                await self._record_new_subscription(unit_of_work, paid, event.subscription, now)
            await self._enqueue_paid_notice(unit_of_work, paid)
            await unit_of_work.commit()
            return True

    async def _record_new_subscription(
        self,
        unit_of_work: UnitOfWork,
        quote: Quote,
        data: SubscriptionEventData,
        now: datetime,
    ) -> None:
        """A subscription-mode checkout completed: the quote is now a
        subscription. A lifecycle event that raced ahead may already have
        created the row, in which case it only absorbs the new fields."""

        draft = quote.draft
        if draft is None or draft.interval is None or quote.customer_id is None:
            logger.warning("subscription checkout completed for a quote without recurring terms")
            return
        existing = await unit_of_work.quote_subscriptions.find_by_subscription_ref(
            data.subscription_ref
        )
        if existing is not None:
            await unit_of_work.quote_subscriptions.save(existing.apply(data, now))
            return
        record = QuoteSubscriptionRecord(
            subscription_id=SubscriptionId(uuid4()),
            business_id=quote.business_id,
            quote_id=quote.quote_id,
            customer_id=quote.customer_id,
            provider=STRIPE_PROVIDER,
            subscription_ref=data.subscription_ref,
            status=data.status or "active",
            interval=data.interval or draft.interval,
            amount_minor=data.amount_minor if data.amount_minor is not None else draft.total_minor,
            currency=(data.currency or draft.currency).upper(),
            current_period_end=data.current_period_end,
            cancel_at_period_end=bool(data.cancel_at_period_end),
            created_at=now,
            updated_at=now,
        )
        try:
            await unit_of_work.quote_subscriptions.create(record)
        except QuotePaymentConflictError:
            raced = await unit_of_work.quote_subscriptions.find_by_subscription_ref(
                data.subscription_ref
            )
            if raced is not None:
                await unit_of_work.quote_subscriptions.save(raced.apply(data, now))

    async def _apply_subscription_event(
        self, unit_of_work: UnitOfWork, event: PaymentWebhookEvent
    ) -> None:
        """Fold a lifecycle event into the subscription row and tell the owner.

        A subscription we do not know is ours when it carries our quote
        metadata — then the checkout completion is still in flight and the
        event is retried; otherwise it belongs to something else in the same
        provider account and is recorded as seen.
        """

        data = event.subscription
        if data is None:
            return
        now = _now()
        subscription = await unit_of_work.quote_subscriptions.find_by_subscription_ref(
            data.subscription_ref
        )
        if subscription is None:
            if QUOTE_METADATA_KEY in event.metadata:
                raise UnknownPaymentSessionError(data.subscription_ref)
            logger.info("ignoring %s for a subscription that is not a quote", event.event_type)
            return
        updated = subscription.apply(data, now)
        await unit_of_work.quote_subscriptions.save(updated)
        quote = await unit_of_work.quotes.get(subscription.business_id, subscription.quote_id)
        if quote is None:
            return
        customer = (
            None
            if subscription.customer_id is None
            else await unit_of_work.customers.get(
                subscription.business_id, subscription.customer_id
            )
        )
        who = (customer.display_name or customer.email) if customer is not None else "customer"
        amount = format_money(updated.amount_minor, updated.currency)
        if event.outcome is PaymentEventOutcome.SUBSCRIPTION_RENEWED:
            paid = data.paid_minor if data.paid_minor is not None else updated.amount_minor
            await unit_of_work.payments.record(
                LedgerPayment(
                    payment_id=uuid4(),
                    business_id=updated.business_id,
                    quote_id=updated.quote_id,
                    kind=PaymentKind.PLAN,
                    source=PaymentSource.STRIPE,
                    method=PaymentMethod.CARD,
                    reference=data.invoice_ref or event.event_id,
                    amount_minor=paid,
                    currency=data.currency or updated.currency,
                    paid_at=event.occurred_at or now,
                    months_covered=months_for(updated.interval),
                    recorded_at=now,
                )
            )
            text = f"Subscription for {who} renewed {format_money(paid, updated.currency)}"
        elif event.outcome is PaymentEventOutcome.SUBSCRIPTION_PAYMENT_FAILED:
            text = f"Subscription for {who} payment failed ({amount} {updated.interval.value}ly)"
        elif event.outcome is PaymentEventOutcome.SUBSCRIPTION_CANCELLED:
            text = f"Subscription for {who} cancelled ({amount} {updated.interval.value}ly)"
        else:
            return
        await self._enqueue_owner_notice(
            unit_of_work,
            quote,
            correlation_id=f"subscription:{updated.subscription_id}:{event.event_id}",
            text=text,
        )

    async def _enqueue_paid_notice(self, unit_of_work: UnitOfWork, quote: Quote) -> None:
        """Tell the owner the money landed, in the conversation where the
        quote was approved. The correlation id makes the notice replay-safe
        beyond the event ledger."""

        draft = quote.draft
        customer = ""
        total = ""
        if draft is not None:
            customer = f" for {draft.recipient.display_name or 'customer'}"
            total = f" — {format_money(draft.total_minor, draft.currency)} paid"
            if draft.interval is not None:
                total += f" ({draft.interval.value}ly subscription started)"
        await self._enqueue_owner_notice(
            unit_of_work,
            quote,
            correlation_id=f"quote:{quote.quote_id}:paid",
            text=f"Quote {public_quote_id(quote.quote_id)}{customer}{total}",
        )

    async def _flag_paid_twice(
        self, unit_of_work: UnitOfWork, quote: Quote, payment: LedgerPayment
    ) -> None:
        """A card payment that raced a payment the owner marked: both are
        kept, one counts, and the owner decides which to refund."""

        active = [
            row
            for row in await unit_of_work.payments.list_for_quote(
                payment.business_id, payment.quote_id
            )
            if row.kind is PaymentKind.ONE_OFF and row.voided_at is None
        ]
        if len(active) < 2:
            return
        name = quote.draft.recipient.display_name if quote.draft is not None else None
        await self._enqueue_owner_notice(
            unit_of_work,
            quote,
            correlation_id=f"quote:{quote.quote_id}:paid-twice:{payment.reference}",
            text=(
                f"Quote {public_quote_id(quote.quote_id)} for {name or 'customer'} was paid"
                f" twice: {format_money(payment.amount_minor, payment.currency)} by card"
                " as well as the payment you marked. It counts once."
                " Refund one of them in Stripe."
            ),
        )

    @staticmethod
    async def _enqueue_owner_notice(
        unit_of_work: UnitOfWork, quote: Quote, *, correlation_id: str, text: str
    ) -> None:
        sent = await enqueue_quote_owner_notice(
            unit_of_work, quote, correlation_id=correlation_id, text=text
        )
        if not sent:
            logger.warning("owner notice has no anchor message for the quote")

    async def _find_claimable(self, unit_of_work: UnitOfWork, claim_token: str) -> Quote:
        quote = await unit_of_work.quotes.get_by_claim_hash(hash_claim_token(claim_token))
        if quote is None or quote.claim_token_hash is None:
            raise QuoteNotFoundError("unknown or expired quote link")
        if not claim_token_matches(claim_token, quote.claim_token_hash):
            raise QuoteNotFoundError("unknown or expired quote link")
        if not quote.is_claimable():
            raise QuoteNotFoundError("unknown or expired quote link")
        return quote

    @staticmethod
    async def _save_customer_move(unit_of_work: UnitOfWork, quote: Quote, updated: Quote) -> None:
        """Losing the optimistic write means a racing request changed the
        quote first — a decline is not an accept, so the loser stops and the
        caller retries against the fresh row."""

        try:
            await unit_of_work.quotes.save(updated, expected_version=quote.version)
        except QuoteConcurrencyError as error:
            raise InvalidQuoteTransitionError(
                "the quote changed while accepting; retry the request"
            ) from error

    async def _business(self, unit_of_work: UnitOfWork, business_id: BusinessId) -> BusinessRecord:
        business = await unit_of_work.businesses.get(business_id)
        if business is None:
            raise QuoteNotFoundError("unknown or expired quote link")
        return business

    async def _hosted_business(
        self, unit_of_work: UnitOfWork, business_id: BusinessId
    ) -> BusinessRecord:
        """The public API serves hosted sites only: a business without a
        ``site_url`` never distributes claim links, so its quotes answer like
        unknown tokens rather than leaking outside the configured flow."""

        business = await self._business(unit_of_work, business_id)
        if business.site_url is None:
            raise QuoteNotFoundError("unknown or expired quote link")
        return business


SUBSCRIPTION_OUTCOMES = frozenset(
    {
        PaymentEventOutcome.SUBSCRIPTION_RENEWED,
        PaymentEventOutcome.SUBSCRIPTION_PAYMENT_FAILED,
        PaymentEventOutcome.SUBSCRIPTION_UPDATED,
        PaymentEventOutcome.SUBSCRIPTION_CANCELLED,
    }
)


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "OpenCheckoutUnavailableError",
    "PublicQuoteService",
    "UnknownPaymentSessionError",
    "PublicQuoteView",
    "QuoteNotFoundError",
    "checkout_line_items",
]
