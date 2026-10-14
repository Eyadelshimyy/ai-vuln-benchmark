"""The owner dashboard: sign-in, the business at a glance, approvals and the
settings GVAS needs to run.

Every read and write is scoped to the business of the owner's session; a
session never reaches another business, and customer portal sessions never
reach these methods. Approvals run the same transitions as approving by text
or the owner channel (``gvas.domain.owner_actions``).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from uuid import uuid4

from gvas.domain.customer_linking import link_quote_customer
from gvas.domain.customers import (
    CustomerRecord,
    ServiceRequest,
    hash_portal_token,
    new_portal_token,
    portal_token_matches,
)
from gvas.domain.enums import BillingInterval, CustomerQuoteStatus, QuoteBilling, QuoteStatus
from gvas.domain.identifiers import BusinessId, MessageKey, SubscriptionId
from gvas.domain.intake import (
    INTAKE_BRIEF_MAX_CHARS,
    INTAKE_OFFER_LINE_MAX_CHARS,
    INTAKE_OPENING_MAX_CHARS,
    INTAKE_QUESTIONS_MAX_CHARS,
    INTAKE_VISIT_QUESTION_MAX_CHARS,
    BookingDecision,
    BookingDecisionAction,
    IntakeConversation,
    sanitize_owner_reason,
)
from gvas.domain.invoices import (
    Invoice,
    InvoiceConflictError,
    InvoiceLine,
    invoice_number,
)
from gvas.domain.invoices import due_on as invoice_due_on
from gvas.domain.money import format_money
from gvas.domain.owner import (
    CALENDAR_WINDOW_MAX_DAYS,
    DISPLAY_NAME_MAX_CHARS,
    OWNER_SESSION_TTL,
    CalendarEvent,
    CalendarEventSource,
    OwnerCalendarError,
    OwnerSession,
    normalize_booking_link,
    normalize_calendar_feed_url,
)
from gvas.domain.owner_actions import (
    BookingDecisionOutcome,
    approve_quote,
    decide_booking,
    reject_quote,
)
from gvas.domain.payments import (
    MANUAL_PROVIDER,
    PLAN_MONTHS_MAX,
    PLAN_NUDGE_LEAD,
    LedgerPayment,
    PaymentKind,
    PaymentMethod,
    PaymentSource,
    QuotePaymentConflictError,
    QuoteSubscriptionRecord,
    checkout_expire_command,
    manual_receipt_command,
    paid_through,
    plan_nudge_command,
)
from gvas.domain.ports import BookedEventsPort, CalendarFeedPort
from gvas.domain.quotes import (
    InvalidQuoteTransitionError,
    Quote,
    QuoteConcurrencyError,
    normalize_customer_email,
    public_quote_id,
)
from gvas.domain.reporting import normalize_email_address
from gvas.domain.repositories import BusinessRecord, UnitOfWork
from gvas.domain.time_zones import business_zone, normalize_time_zone

logger = logging.getLogger(__name__)

QUOTE_LIST_LIMIT = 500
BOOKING_LIST_LIMIT = 200
REQUEST_LIST_LIMIT = 200
DUPLICATE_WINDOW = timedelta(minutes=1)
PAYMENT_NOTE_MAX_CHARS = 500
MANUAL_PAYMENT_METHODS = frozenset({PaymentMethod.CHECK, PaymentMethod.CASH, PaymentMethod.OTHER})
_RECEIPT_METHOD = {PaymentMethod.CHECK: " by check", PaymentMethod.CASH: " in cash"}
PLAN_AMOUNT_MAX_MINOR = 10_000_000
#: The form's one-time key: a retry or double-click reuses it.
PAYMENT_KEY_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class OwnerAuthenticationError(PermissionError):
    """Unknown, expired, revoked or no-longer-owner credentials. Callers map
    every case to the same generic 401."""


class OwnerNotFoundError(LookupError):
    pass


class OwnerConflictError(ValueError):
    """The item is not in a state the action applies to; message is safe."""


class OwnerInputError(ValueError):
    """A settings value was rejected; message is safe to show."""


@dataclass(frozen=True)
class OwnerContext:
    session: OwnerSession
    business: BusinessRecord


@dataclass(frozen=True)
class CustomerSummary:
    customer: CustomerRecord
    quotes: tuple[Quote, ...]


@dataclass(frozen=True)
class ServiceRequestView:
    request: ServiceRequest
    customer: CustomerRecord | None


@dataclass(frozen=True)
class CalendarView:
    events: tuple[CalendarEvent, ...]
    #: Owner-facing notices for sources that failed; the rest still render.
    problems: tuple[str, ...]


@dataclass(frozen=True)
class SettingsUpdate:
    """``None`` leaves a field alone. An empty string clears the intake
    texts, the notification e-mail and the calendar link."""

    display_name: str | None = None
    calendly_url: str | None = None
    intake_brief: str | None = None
    intake_questions: str | None = None
    intake_opening: str | None = None
    intake_offer_line: str | None = None
    intake_visit_question: str | None = None
    notification_email: str | None = None
    calendar_feed_url: str | None = None
    timezone: str | None = None


class OwnerService:
    def __init__(
        self,
        unit_of_work_factory: Callable[[], UnitOfWork],
        *,
        booked_events: BookedEventsPort | None = None,
        calendar_feed: CalendarFeedPort | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory
        self._booked_events = booked_events
        self._calendar_feed = calendar_feed
        self._now = now

    # -- sign-in -----------------------------------------------------------

    async def exchange_login_token(self, raw_token: str) -> tuple[str, OwnerContext] | None:
        """``None`` when ``raw_token`` is not an owner link (the caller then
        tries the customer portal); raises when it is one but cannot be used."""

        now = self._now()
        token_hash = hash_portal_token(raw_token)
        async with self._unit_of_work_factory() as unit_of_work:
            token = await unit_of_work.owner_login_tokens.find_by_hash(token_hash)
            if token is None or not portal_token_matches(raw_token, token.token_hash):
                return None
            if not token.is_usable(now):
                raise OwnerAuthenticationError("owner login token is not usable")
            business = await unit_of_work.businesses.get(token.business_id)
            if business is None or not _is_owner(business, token.email):
                raise OwnerAuthenticationError("owner login token is not usable")
            if not await unit_of_work.owner_login_tokens.mark_used(token_hash, now):
                raise OwnerAuthenticationError("owner login token is not usable")
            raw_session = new_portal_token()
            session = OwnerSession(
                token_hash=hash_portal_token(raw_session),
                business_id=business.business_id,
                email=token.email,
                expires_at=now + OWNER_SESSION_TTL,
                created_at=now,
            )
            await unit_of_work.owner_sessions.add(session)
            await unit_of_work.commit()
        return raw_session, OwnerContext(session=session, business=business)

    async def authenticate(self, raw_session_token: str) -> OwnerContext:
        now = self._now()
        token_hash = hash_portal_token(raw_session_token)
        async with self._unit_of_work_factory() as unit_of_work:
            session = await unit_of_work.owner_sessions.find_by_hash(token_hash)
            if (
                session is None
                or not portal_token_matches(raw_session_token, session.token_hash)
                or not session.is_active(now)
            ):
                raise OwnerAuthenticationError("owner session is not active")
            business = await unit_of_work.businesses.get(session.business_id)
            # Changing the owner e-mail ends every session of the old one.
            if business is None or not _is_owner(business, session.email):
                raise OwnerAuthenticationError("owner session is not active")
        return OwnerContext(session=session, business=business)

    async def revoke_session(self, raw_session_token: str) -> None:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.owner_sessions.revoke(
                hash_portal_token(raw_session_token), self._now()
            )
            await unit_of_work.commit()

    # -- reads -------------------------------------------------------------

    async def quotes(self, context: OwnerContext) -> tuple[Quote, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            quotes = await unit_of_work.quotes.list_for_business(
                context.business.business_id, limit=QUOTE_LIST_LIMIT
            )
        return tuple(quote for quote in quotes if quote.draft is not None)

    async def customers(self, context: OwnerContext) -> tuple[CustomerSummary, ...]:
        business_id = context.business.business_id
        async with self._unit_of_work_factory() as unit_of_work:
            customers = await unit_of_work.customers.list_for_business(business_id)
            # Per customer, so lifetime totals don't depend on a business-wide page.
            summaries: list[CustomerSummary] = []
            for customer in customers:
                quotes = await unit_of_work.quotes.list_for_customer(
                    business_id, customer.customer_id
                )
                summaries.append(
                    CustomerSummary(
                        customer=customer,
                        quotes=tuple(quote for quote in quotes if quote.draft is not None),
                    )
                )
        return tuple(summaries)

    async def payments(self, context: OwnerContext) -> tuple[LedgerPayment, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.payments.list_for_business(context.business.business_id)

    async def subscriptions(self, context: OwnerContext) -> tuple[QuoteSubscriptionRecord, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.quote_subscriptions.list_for_business(
                context.business.business_id
            )

    async def bookings(self, context: OwnerContext) -> tuple[IntakeConversation, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.intake_conversations.list_booking_requests(
                context.business.business_id, limit=BOOKING_LIST_LIMIT
            )

    async def _awaiting_owner(self, business_id: BusinessId) -> tuple[IntakeConversation, ...]:
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.intake_conversations.list_awaiting_owner(
                business_id, limit=BOOKING_LIST_LIMIT
            )

    async def service_requests(self, context: OwnerContext) -> tuple[ServiceRequestView, ...]:
        business_id = context.business.business_id
        async with self._unit_of_work_factory() as unit_of_work:
            requests = await unit_of_work.service_requests.list_for_business(
                business_id, limit=REQUEST_LIST_LIMIT
            )
            customers = {
                customer.customer_id: customer
                for customer in await unit_of_work.customers.list_for_business(business_id)
            }
        return tuple(
            ServiceRequestView(request=request, customer=customers.get(request.customer_id))
            for request in requests
        )

    async def calendar(self, context: OwnerContext, start: datetime, end: datetime) -> CalendarView:
        return await self.business_calendar(context.business, start, end)

    async def business_calendar(
        self, business: BusinessRecord, start: datetime, end: datetime
    ) -> CalendarView:
        if start.tzinfo is None or end.tzinfo is None:
            raise OwnerInputError("start and end must carry a timezone")
        if end <= start or end - start > timedelta(days=CALENDAR_WINDOW_MAX_DAYS):
            raise OwnerInputError(
                f"the window must be positive and at most {CALENDAR_WINDOW_MAX_DAYS} days"
            )
        problems: list[str] = []
        booked: tuple[CalendarEvent, ...] = ()
        own: tuple[CalendarEvent, ...] = ()
        if self._booked_events is not None and self._booked_events.serves(business.business_id):
            try:
                booked = await self._booked_events.upcoming(business.business_id, start, end)
            except OwnerCalendarError as error:
                problems.append(str(error))
        if business.calendar_feed_url and self._calendar_feed is not None:
            try:
                own = await self._calendar_feed.events(business.calendar_feed_url, start, end)
            except OwnerCalendarError as error:
                problems.append(str(error))
        requests = tuple(
            CalendarEvent(
                source=CalendarEventSource.REQUEST,
                title=f"Booking request: {conversation.collected.name or 'customer'}",
                start=conversation.requested_slot_start,
                end=conversation.requested_slot_end,
                location=conversation.collected.address,
                invitee_name=conversation.collected.name,
                invitee_email=conversation.collected.email,
                reference=conversation.reference,
            )
            for conversation in await self._awaiting_owner(business.business_id)
            if conversation.requested_slot_start is not None
            and start <= conversation.requested_slot_start < end
        )
        # Calendly writes each booking into the owner's own calendar too; show
        # it once, as the booking (it carries the customer).
        own = tuple(event for event in own if not _mirrors_booking(event, booked))
        events = sorted((*booked, *own, *requests), key=lambda event: event.start)
        return CalendarView(events=tuple(events), problems=tuple(problems))

    # -- actions -----------------------------------------------------------

    async def decide_booking(
        self, context: OwnerContext, reference: str, *, approve: bool, reason: str | None = None
    ) -> BookingDecisionOutcome:
        decision = BookingDecision(
            action=BookingDecisionAction.APPROVE if approve else BookingDecisionAction.DECLINE,
            reference=reference.strip().lower(),
            reason=sanitize_owner_reason(reason) if reason and reason.strip() else None,
        )
        async with self._unit_of_work_factory() as unit_of_work:
            return await decide_booking(
                unit_of_work, context.business.business_id, decision, self._now()
            )

    async def decide_quote(self, context: OwnerContext, quote_id: str, *, approve: bool) -> Quote:
        now = self._now()
        async with self._unit_of_work_factory() as unit_of_work:
            # Only quotes still waiting can be decided, so search those alone:
            # an old pending quote stays reachable however many newer ones exist.
            quotes = await unit_of_work.quotes.list_for_business(
                context.business.business_id,
                limit=QUOTE_LIST_LIMIT,
                status=QuoteStatus.AWAITING_APPROVAL,
            )
            quote = next(
                (
                    candidate
                    for candidate in quotes
                    if public_quote_id(candidate.quote_id) == quote_id
                ),
                None,
            )
            if quote is None:
                if any(
                    public_quote_id(candidate.quote_id) == quote_id
                    for candidate in await unit_of_work.quotes.list_for_business(
                        context.business.business_id, limit=QUOTE_LIST_LIMIT
                    )
                ):
                    raise OwnerConflictError("This quote isn't waiting for your OK.")
                raise OwnerNotFoundError("quote not found")
            key = MessageKey(f"dashboard:{uuid4()}")
            try:
                if approve:
                    decided = await approve_quote(unit_of_work, quote, key, now)
                else:
                    decided = await reject_quote(unit_of_work, quote, key, now)
                await unit_of_work.commit()
            except (InvalidQuoteTransitionError, QuoteConcurrencyError) as error:
                raise OwnerConflictError("This quote changed; refresh and try again.") from error
        return decided

    async def mark_paid(
        self,
        context: OwnerContext,
        quote_id: str,
        *,
        paid_on: date,
        method: PaymentMethod,
        note: str | None = None,
    ) -> Quote:
        """Record a check, cash or other payment for the full amount of a
        sent one-off quote, accepted or not. The ledger keeps one active payment per
        quote, an open card checkout is closed, and the customer is e-mailed
        a short receipt."""

        if method not in MANUAL_PAYMENT_METHODS:
            raise OwnerInputError("Pick check, cash or other.")
        note = (note or "").strip() or None
        if note is not None and len(note) > PAYMENT_NOTE_MAX_CHARS:
            raise OwnerInputError(f"Keep the note under {PAYMENT_NOTE_MAX_CHARS} characters.")
        now = self._now()
        zone = business_zone(context.business.timezone) or UTC
        if paid_on > now.astimezone(zone).date():
            raise OwnerInputError("The paid date can't be in the future.")
        business_id = context.business.business_id
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_quote(unit_of_work, context, quote_id)
            draft = quote.draft
            if draft is None or draft.billing is not QuoteBilling.ONE_TIME:
                raise OwnerConflictError("Only one-off quotes can be marked paid.")
            if quote.customer_status is CustomerQuoteStatus.PAID:
                raise OwnerConflictError("This quote is already paid.")
            if quote.customer_status is CustomerQuoteStatus.DECLINED:
                raise OwnerConflictError("The customer declined this quote.")
            # Approved quotes are claimable before delivery runs; until then
            # only a customer who has seen it makes it payable.
            if not quote.is_claimable() or (
                quote.customer_status is None and quote.status is QuoteStatus.APPROVED
            ):
                raise OwnerConflictError("Only quotes sent to the customer can be marked paid.")
            payment_id = uuid4()
            recorded = await unit_of_work.payments.record(
                LedgerPayment(
                    payment_id=payment_id,
                    business_id=business_id,
                    quote_id=quote.quote_id,
                    kind=PaymentKind.ONE_OFF,
                    source=PaymentSource.MANUAL,
                    method=method,
                    reference=f"manual:{payment_id}",
                    amount_minor=draft.total_minor,
                    currency=draft.currency,
                    # Midday in the business's zone keeps the day stable.
                    paid_at=datetime.combine(paid_on, time(12), tzinfo=zone).astimezone(UTC),
                    recorded_by=context.session.email,
                    recorded_at=now,
                    note=note,
                    customer_status_before=_status_before(quote.customer_status),
                )
            )
            if recorded is None or recorded.duplicate:
                raise OwnerConflictError("This quote already has a payment; refresh.")
            paid = quote.record_customer_payment(now)
            try:
                await unit_of_work.quotes.save(paid, expected_version=quote.version)
            except QuoteConcurrencyError as error:
                raise OwnerConflictError("This quote changed; refresh and try again.") from error
            checkout = await unit_of_work.quote_payments.find_open(business_id, quote.quote_id)
            if checkout is not None:
                await unit_of_work.quote_payments.save(
                    checkout.mark_expired(now), expected_from=checkout.status
                )
                if checkout.checkout_session_id:
                    await unit_of_work.outbox.enqueue(
                        checkout_expire_command(business_id, checkout.checkout_session_id)
                    )
            email = draft.recipient.email_address
            if email is not None:
                business = context.business.display_name or context.business.name
                work = ", ".join(item.description for item in draft.line_items)
                await unit_of_work.outbox.enqueue(
                    manual_receipt_command(
                        business_id=business_id,
                        quote_id=quote.quote_id,
                        payment_id=payment_id,
                        to=email,
                        subject=f"Receipt from {business}",
                        body="\n\n".join(
                            [
                                f"Thanks! {business} received your payment of"
                                f" {format_money(draft.total_minor, draft.currency)}"
                                f"{_RECEIPT_METHOD.get(method, '')}"
                                f" on {paid_on:%B} {paid_on.day}, {paid_on.year}.",
                                f"For: {work}" if work else "",
                                "This quote is paid in full. Keep this e-mail as your receipt.",
                            ]
                        ).replace("\n\n\n\n", "\n\n"),
                    )
                )
            await unit_of_work.commit()
        return paid

    async def record_plan_payment(
        self,
        context: OwnerContext,
        quote_id: str,
        *,
        key: str,
        paid_on: date,
        method: PaymentMethod,
        months: int,
        amount_minor: int,
        note: str | None = None,
    ) -> QuoteSubscriptionRecord:
        """Record a check, cash or other payment covering ``months`` of a
        sent recurring quote. The first one starts a manual plan paid from
        ``paid_on``; each later one moves its paid-through date forward.
        ``key`` comes from the form, so a retry records it once."""

        if method not in MANUAL_PAYMENT_METHODS:
            raise OwnerInputError("Pick check, cash or other.")
        if not 1 <= months <= PLAN_MONTHS_MAX:
            raise OwnerInputError(f"A payment covers 1-{PLAN_MONTHS_MAX} months.")
        if not 0 <= amount_minor <= PLAN_AMOUNT_MAX_MINOR:
            raise OwnerInputError("Enter the amount received.")
        note = (note or "").strip() or None
        if note is not None and len(note) > PAYMENT_NOTE_MAX_CHARS:
            raise OwnerInputError(f"Keep the note under {PAYMENT_NOTE_MAX_CHARS} characters.")
        now = self._now()
        zone = business_zone(context.business.timezone) or UTC
        if paid_on > now.astimezone(zone).date():
            raise OwnerInputError("The paid date can't be in the future.")
        business_id = context.business.business_id
        reference = f"manual:{key}"
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_quote(unit_of_work, context, quote_id)
            draft = quote.draft
            if draft is None or draft.billing is QuoteBilling.ONE_TIME or draft.interval is None:
                raise OwnerConflictError("Only recurring quotes have a plan.")
            if draft.interval is BillingInterval.YEAR and months % 12:
                raise OwnerInputError("A yearly plan is paid in whole years.")
            if quote.customer_status is CustomerQuoteStatus.DECLINED:
                raise OwnerConflictError("The customer declined this quote.")
            if not quote.is_claimable() or (
                quote.customer_status is None and quote.status is QuoteStatus.APPROVED
            ):
                raise OwnerConflictError("Only quotes sent to the customer can be marked paid.")
            plans = await unit_of_work.quote_subscriptions.list_for_quote(
                business_id, quote.quote_id
            )
            payments = await unit_of_work.payments.list_for_quote(business_id, quote.quote_id)
            plan = next((p for p in plans if p.is_manual), None)
            known = next(
                (
                    p
                    for p in payments
                    if p.reference == reference and p.source is PaymentSource.MANUAL
                ),
                None,
            )
            if known is not None:
                # A retry of a payment undone since is not a new payment.
                if plan is None or known.voided_at is not None:
                    raise OwnerConflictError("This payment changed; refresh and try again.")
                return plan
            if any(not p.is_manual and p.is_live for p in plans):
                raise OwnerConflictError("This customer pays by card; Stripe bills the plan.")
            if plan is not None:
                plan = await unit_of_work.quote_subscriptions.get(
                    business_id, plan.subscription_id, for_update=True
                )
                assert plan is not None
                payments = await unit_of_work.payments.list_for_quote(business_id, quote.quote_id)
            linked = quote
            if quote.customer_id is None:
                linked, _ = await link_quote_customer(unit_of_work, quote, now)
            starts_paid = quote.customer_status is not CustomerQuoteStatus.PAID
            if linked is not quote or starts_paid:
                saved = linked.record_customer_payment(now) if starts_paid else linked
                try:
                    await unit_of_work.quotes.save(saved, expected_version=quote.version)
                except QuoteConcurrencyError as error:
                    raise OwnerConflictError(
                        "This quote changed; refresh and try again."
                    ) from error
            if starts_paid:
                await self._close_open_checkout(unit_of_work, quote)
            covered = sum(
                p.months_covered or 0
                for p in payments
                if p.kind is PaymentKind.PLAN
                and p.source is PaymentSource.MANUAL
                and p.voided_at is None
            )
            fresh = plan is None or not plan.is_live or plan.paid_from is None or not covered
            paid_from = paid_on if fresh or plan is None else plan.paid_from
            assert paid_from is not None
            covered = months if fresh else covered + months
            payment_id = uuid4()
            recorded = await unit_of_work.payments.record(
                LedgerPayment(
                    payment_id=payment_id,
                    business_id=business_id,
                    quote_id=quote.quote_id,
                    kind=PaymentKind.PLAN,
                    source=PaymentSource.MANUAL,
                    method=method,
                    reference=reference,
                    amount_minor=amount_minor,
                    currency=draft.currency,
                    paid_at=datetime.combine(paid_on, time(12), tzinfo=zone).astimezone(UTC),
                    months_covered=months,
                    recorded_by=context.session.email,
                    recorded_at=now,
                    note=note,
                    customer_status_before=(
                        _status_before(quote.customer_status) if starts_paid else None
                    ),
                )
            )
            if recorded is None:
                raise OwnerConflictError("This payment changed; refresh and try again.")
            if plan is None:
                plan = QuoteSubscriptionRecord(
                    subscription_id=SubscriptionId(uuid4()),
                    business_id=business_id,
                    quote_id=quote.quote_id,
                    customer_id=linked.customer_id,
                    provider=MANUAL_PROVIDER,
                    subscription_ref=f"manual:{quote.quote_id}",
                    status="active",
                    interval=draft.interval,
                    amount_minor=draft.total_minor,
                    currency=draft.currency,
                    current_period_end=None,
                    created_at=now,
                    updated_at=now,
                    paid_from=paid_from,
                    paid_through=paid_through(paid_from, covered),
                )
                try:
                    await unit_of_work.quote_subscriptions.create(plan)
                except QuotePaymentConflictError as error:
                    raise OwnerConflictError("This plan changed; refresh and try again.") from error
            else:
                plan = plan.model_copy(
                    update={
                        "status": "active",
                        "customer_id": plan.customer_id or linked.customer_id,
                        "paid_from": paid_from,
                        "paid_through": paid_through(paid_from, covered),
                        "updated_at": now,
                    }
                )
                await unit_of_work.quote_subscriptions.save(plan)
            await self._queue_plan_nudge(unit_of_work, plan, now, key=f"pay:{payment_id}")
            email = draft.recipient.email_address
            if email is not None:
                business = context.business.display_name or context.business.name
                await unit_of_work.outbox.enqueue(
                    manual_receipt_command(
                        business_id=business_id,
                        quote_id=quote.quote_id,
                        payment_id=payment_id,
                        to=email,
                        subject=f"Receipt from {business}",
                        body="\n\n".join(
                            [
                                f"Thanks! {business} received your payment of"
                                f" {format_money(amount_minor, draft.currency)}"
                                f"{_RECEIPT_METHOD.get(method, '')}"
                                f" on {paid_on:%B} {paid_on.day}, {paid_on.year}.",
                                "Keep this e-mail as your receipt.",
                            ]
                        ),
                        subscription_id=plan.subscription_id,
                    )
                )
            await unit_of_work.commit()
        return plan

    async def void_plan_payment(
        self, context: OwnerContext, quote_id: str, payment_id: str
    ) -> QuoteSubscriptionRecord:
        """Void one manual plan payment, keeping who voided it and when; the
        paid-through date moves back by the months it covered. Voiding the
        last one ends the plan and puts the quote back where it was."""

        now = self._now()
        business_id = context.business.business_id
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_quote(unit_of_work, context, quote_id)
            plans = await unit_of_work.quote_subscriptions.list_for_quote(
                business_id, quote.quote_id
            )
            found = next((p for p in plans if p.is_manual), None)
            if found is None:
                raise OwnerNotFoundError("plan not found")
            plan = await unit_of_work.quote_subscriptions.get(
                business_id, found.subscription_id, for_update=True
            )
            assert plan is not None
            ledger = await unit_of_work.payments.list_for_quote(business_id, quote.quote_id)
            manual = [
                p for p in ledger if p.kind is PaymentKind.PLAN and p.source is PaymentSource.MANUAL
            ]
            target = next((p for p in manual if str(p.payment_id) == payment_id), None)
            if target is None:
                raise OwnerNotFoundError("payment not found")
            if target.voided_at is not None or not await unit_of_work.payments.void_manual(
                business_id, target.payment_id, by=context.session.email, at=now
            ):
                raise OwnerConflictError("This payment changed; refresh and try again.")
            remaining = [
                p for p in manual if p.voided_at is None and p.payment_id != target.payment_id
            ]
            if remaining and plan.paid_from is not None:
                covered = sum(p.months_covered or 0 for p in remaining)
                plan = plan.model_copy(
                    update={
                        "paid_through": paid_through(plan.paid_from, covered),
                        "updated_at": now,
                    }
                )
                await unit_of_work.quote_subscriptions.save(plan)
                await self._queue_plan_nudge(
                    unit_of_work, plan, now, key=f"undo:{target.payment_id}"
                )
            else:
                plan = plan.model_copy(
                    update={"status": "canceled", "paid_through": None, "updated_at": now}
                )
                await unit_of_work.quote_subscriptions.save(plan)
                # A card payment that raced the manual plan counts instead, and
                # keeps the quote paid even if that card plan has since ended.
                card = [
                    row
                    for row in ledger
                    if row.source is PaymentSource.STRIPE and row.voided_at is None
                ]
                for row in card:
                    if row.duplicate:
                        await unit_of_work.payments.mark_duplicate(
                            business_id, row.payment_id, False
                        )
                if card or any(not p.is_manual and p.is_live for p in plans):
                    await unit_of_work.commit()
                    return plan
                before = next(
                    (
                        p.customer_status_before
                        for p in sorted(manual, key=lambda p: p.recorded_at, reverse=True)
                        if p.customer_status_before is not None
                    ),
                    None,
                )
                updated = quote.undo_customer_payment(now, _status_after_undo(before))
                if updated is not quote:
                    try:
                        await unit_of_work.quotes.save(updated, expected_version=quote.version)
                    except QuoteConcurrencyError as error:
                        raise OwnerConflictError(
                            "This quote changed; refresh and try again."
                        ) from error
            await unit_of_work.commit()
        return plan

    async def _queue_plan_nudge(
        self,
        unit_of_work: UnitOfWork,
        plan: QuoteSubscriptionRecord,
        now: datetime,
        *,
        key: str,
    ) -> None:
        """Nudge the owner a week before the plan runs out, at 9am in the
        business's zone; the worker drops it if the plan moved on."""

        through = plan.paid_through
        business = await unit_of_work.businesses.get(plan.business_id)
        zone = business_zone(business.timezone if business else None) or UTC
        if through is None or through < now.astimezone(zone).date():
            return
        at = datetime.combine(through - PLAN_NUDGE_LEAD, time(9), tzinfo=zone).astimezone(UTC)
        await unit_of_work.outbox.enqueue(plan_nudge_command(plan, at=max(at, now), key=key))

    async def _close_open_checkout(self, unit_of_work: UnitOfWork, quote: Quote) -> None:
        checkout = await unit_of_work.quote_payments.find_open(quote.business_id, quote.quote_id)
        if checkout is None:
            return
        await unit_of_work.quote_payments.save(
            checkout.mark_expired(self._now()), expected_from=checkout.status
        )
        if checkout.checkout_session_id:
            await unit_of_work.outbox.enqueue(
                checkout_expire_command(quote.business_id, checkout.checkout_session_id)
            )

    async def invoices(self, context: OwnerContext) -> tuple[Invoice, ...]:
        if not context.business.invoice_prefix:
            return ()
        async with self._unit_of_work_factory() as unit_of_work:
            return await unit_of_work.invoices.list_for_business(context.business.business_id)

    async def send_invoice(self, context: OwnerContext, quote_id: str) -> Invoice:
        """Invoice an accepted one-off quote: an exact copy of its lines and
        total, due in 14 days. Sending only records it for now; the customer
        is not e-mailed."""

        prefix = context.business.invoice_prefix
        if not prefix:
            raise OwnerNotFoundError("invoices are off for this business")
        now = self._now()
        zone = business_zone(context.business.timezone) or UTC
        business_id = context.business.business_id
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_quote(unit_of_work, context, quote_id)
            draft = quote.draft
            if draft is None or draft.billing is not QuoteBilling.ONE_TIME:
                raise OwnerConflictError("Only one-off quotes can be invoiced.")
            if quote.customer_status not in (
                CustomerQuoteStatus.ACCEPTED,
                CustomerQuoteStatus.PAID,
            ):
                raise OwnerConflictError("Only accepted quotes can be invoiced.")
            if await unit_of_work.invoices.for_quote(business_id, quote.quote_id) is not None:
                raise OwnerConflictError("This quote already has an invoice.")
            sequence = await unit_of_work.invoices.next_sequence(business_id)
            recipient = draft.recipient
            invoice = Invoice(
                invoice_id=uuid4(),
                business_id=business_id,
                quote_id=quote.quote_id,
                sequence=sequence,
                number=invoice_number(prefix, sequence),
                customer_name=recipient.display_name,
                customer_email=recipient.email_address,
                customer_phone=recipient.phone,
                service_address=recipient.service_address,
                line_items=tuple(
                    InvoiceLine(
                        description=item.description,
                        quantity=item.quantity,
                        unit_price_minor=item.unit_price_minor,
                    )
                    for item in draft.line_items
                ),
                total_minor=draft.total_minor,
                currency=draft.currency,
                sent_at=now,
                due_on=invoice_due_on(now, zone),
                sent_by=context.session.email,
                created_at=now,
            )
            try:
                await unit_of_work.invoices.create(invoice)
            except InvoiceConflictError as error:
                raise OwnerConflictError("This quote changed; refresh and try again.") from error
            await unit_of_work.commit()
        return invoice

    async def mark_unpaid(self, context: OwnerContext, quote_id: str) -> Quote:
        """Void the manual payment of a one-off quote, keeping who voided it
        and when. Card payments are refunded in Stripe, never undone here."""

        now = self._now()
        business_id = context.business.business_id
        async with self._unit_of_work_factory() as unit_of_work:
            quote = await self._find_quote(unit_of_work, context, quote_id)
            payments = [
                payment
                for payment in await unit_of_work.payments.list_for_quote(
                    business_id, quote.quote_id
                )
                if payment.kind is PaymentKind.ONE_OFF and payment.voided_at is None
            ]
            manual = next((p for p in payments if p.source is PaymentSource.MANUAL), None)
            if manual is None:
                if payments:
                    raise OwnerConflictError(
                        "Card payments can't be undone here. Refund it in Stripe."
                    )
                raise OwnerConflictError("This quote has no payment to undo.")
            if not await unit_of_work.payments.void_manual(
                business_id, manual.payment_id, by=context.session.email, at=now
            ):
                raise OwnerConflictError("This payment changed; refresh and try again.")
            # A card payment that raced the manual one counts instead: it
            # already did when it settled first, or it takes over now.
            standby = next(
                (p for p in payments if p.duplicate and p.payment_id != manual.payment_id),
                None,
            )
            if manual.duplicate:
                updated = quote
            elif standby is not None:
                await unit_of_work.payments.mark_duplicate(
                    business_id, standby.payment_id, duplicate=False
                )
                updated = quote
            else:
                updated = quote.undo_customer_payment(
                    now, _status_after_undo(manual.customer_status_before)
                )
                try:
                    await unit_of_work.quotes.save(updated, expected_version=quote.version)
                except QuoteConcurrencyError as error:
                    raise OwnerConflictError(
                        "This quote changed; refresh and try again."
                    ) from error
            await unit_of_work.commit()
        return updated

    async def _find_quote(
        self, unit_of_work: UnitOfWork, context: OwnerContext, quote_id: str
    ) -> Quote:
        for candidate in await unit_of_work.quotes.list_for_business(
            context.business.business_id, limit=QUOTE_LIST_LIMIT
        ):
            if public_quote_id(candidate.quote_id) == quote_id:
                return candidate
        raise OwnerNotFoundError("quote not found")

    async def update_settings(
        self, context: OwnerContext, update: SettingsUpdate
    ) -> BusinessRecord:
        fields: dict[str, str] = {}
        if update.display_name is not None:
            name = " ".join(update.display_name.split())
            if not name or len(name) > DISPLAY_NAME_MAX_CHARS:
                raise OwnerInputError(
                    f"Business name must be 1-{DISPLAY_NAME_MAX_CHARS} characters."
                )
            fields["display_name"] = name
        if update.calendly_url is not None:
            try:
                fields["calendly_url"] = normalize_booking_link(update.calendly_url)
            except ValueError as error:
                raise OwnerInputError(f"Calendly link {error}.") from error
        for field, label, limit in (
            ("intake_brief", "About your business", INTAKE_BRIEF_MAX_CHARS),
            ("intake_questions", "Questions Gus asks", INTAKE_QUESTIONS_MAX_CHARS),
            ("intake_opening", "Gus's greeting", INTAKE_OPENING_MAX_CHARS),
            ("intake_offer_line", "Gus's line above the times", INTAKE_OFFER_LINE_MAX_CHARS),
            (
                "intake_visit_question",
                "Gus's question before the visit",
                INTAKE_VISIT_QUESTION_MAX_CHARS,
            ),
        ):
            value = getattr(update, field)
            if value is None:
                continue
            text = value.strip()
            if len(text) > limit:
                raise OwnerInputError(f"{label} must be at most {limit} characters.")
            fields[field] = text
        if update.notification_email is not None:
            raw = update.notification_email.strip()
            if raw:
                address = normalize_email_address(raw)
                if address is None:
                    raise OwnerInputError("Notification email isn't a valid address.")
                fields["notification_email"] = address
            else:
                fields["notification_email"] = ""
        if update.calendar_feed_url is not None:
            raw = update.calendar_feed_url.strip()
            if raw:
                try:
                    fields["calendar_feed_url"] = normalize_calendar_feed_url(raw)
                except ValueError as error:
                    raise OwnerInputError(f"Calendar link {error}.") from error
            else:
                fields["calendar_feed_url"] = ""
        if update.timezone is not None:
            try:
                fields["timezone"] = (
                    normalize_time_zone(update.timezone) if update.timezone.strip() else ""
                )
            except ValueError as error:
                raise OwnerInputError(f"Time zone {error}.") from error
        async with self._unit_of_work_factory() as unit_of_work:
            record = await unit_of_work.businesses.configure_site(
                context.business.business_id,
                now=self._now(),
                **fields,
            )
            await unit_of_work.commit()
        return record


def _is_owner(business: BusinessRecord, email: str) -> bool:
    return bool(business.owner_email) and normalize_customer_email(
        business.owner_email or ""
    ) == normalize_customer_email(email)


def _mirrors_booking(event: CalendarEvent, booked: tuple[CalendarEvent, ...]) -> bool:
    return any(
        abs(event.start - booking.start) < DUPLICATE_WINDOW
        and (
            event.end is None
            or booking.end is None
            or abs(event.end - booking.end) < DUPLICATE_WINDOW
        )
        for booking in booked
    )


__all__ = [
    "CalendarView",
    "CustomerSummary",
    "OwnerAuthenticationError",
    "OwnerConflictError",
    "OwnerContext",
    "OwnerInputError",
    "OwnerNotFoundError",
    "OwnerService",
    "ServiceRequestView",
    "SettingsUpdate",
]


def _status_before(status: CustomerQuoteStatus | None) -> Literal["sent", "viewed", "accepted"]:
    if status is CustomerQuoteStatus.VIEWED:
        return "viewed"
    if status is CustomerQuoteStatus.ACCEPTED:
        return "accepted"
    return "sent"


def _status_after_undo(
    before: Literal["sent", "viewed", "accepted"] | None,
) -> CustomerQuoteStatus | None:
    # Rows recorded before this was kept were all on accepted quotes.
    if before == "sent":
        return None
    if before == "viewed":
        return CustomerQuoteStatus.VIEWED
    return CustomerQuoteStatus.ACCEPTED
