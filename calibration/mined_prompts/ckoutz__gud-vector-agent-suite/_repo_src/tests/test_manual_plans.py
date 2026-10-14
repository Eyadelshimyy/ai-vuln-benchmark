"""Manual plans: a check or cash prepay for a recurring quote, recorded once,
moves a paid-through date; a one-time key stops double submits, undo keeps
a trail, and the owner is nudged a week before the plan runs out."""

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.manual_payments import ManualPaymentEffectsService
from gvas.application.owner import (
    OwnerConflictError,
    OwnerContext,
    OwnerInputError,
    OwnerNotFoundError,
)
from gvas.domain.enums import BillingInterval, RecipientAddressKind
from gvas.domain.identifiers import BusinessId, CustomerId, QuoteId, SubscriptionId
from gvas.domain.payments import (
    MANUAL_RECEIPT_COMMAND_TYPE,
    PLAN_NUDGE_COMMAND_TYPE,
    PaymentMethod,
    QuoteSubscriptionRecord,
    add_months,
    month_totals,
    paid_through,
)
from gvas.domain.quotes import public_quote_id
from gvas.domain.time_zones import business_zone
from gvas.infrastructure.models import OutboxMessage, QuoteRecord
from gvas.infrastructure.payment_models import LedgerPaymentRow, QuoteSubscription
from gvas.infrastructure.payment_repositories import (
    SqlPaymentLedgerRepository,
    SqlQuoteSubscriptionRepository,
)
from gvas.infrastructure.stripe import StripeWebhookVerifier
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory
from test_customer_portal import (
    BillingFake,
    Portal,
    client,
    portal_business,
    post_event,
    subscription_checkout_event,
)
from test_hosted_quotes import WEBHOOK_SECRET, CheckoutFake
from test_mark_paid import OWNER, _context
from test_owner_dashboard import bearer, http_client, owner_business, sign_in
from test_pilot_runtime import immediate_worker, texts_of


async def _plan_quote(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    recurring: bool = True,
    checkout: CheckoutFake | None = None,
) -> tuple[Portal, OwnerContext, str, date]:
    portal = await portal_business(
        session_factory,
        recurring=recurring,
        public_key=f"gvb_plan_{uuid4().hex[:12]}",
        checkout=checkout,
        billing=BillingFake() if checkout is not None else None,
    )
    async with session_factory() as session:
        row = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == portal.business_id)
        )
    assert row is not None
    context = await _context(session_factory, portal.business_id)
    zone = business_zone(context.business.timezone) or UTC
    today = datetime.now(UTC).astimezone(zone).date()
    return portal, context, public_quote_id(QuoteId(row.id)), today


async def _status(session_factory: async_sessionmaker[AsyncSession], portal: Portal) -> str | None:
    async with session_factory() as session:
        row = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == portal.business_id)
        )
    assert row is not None
    status: str | None = row.customer_status
    return status


async def _rows(
    session_factory: async_sessionmaker[AsyncSession], portal: Portal
) -> list[LedgerPaymentRow]:
    async with session_factory() as session:
        rows = await session.scalars(
            select(LedgerPaymentRow)
            .where(LedgerPaymentRow.business_id == portal.business_id)
            .order_by(LedgerPaymentRow.recorded_at)
        )
        return list(rows.all())


async def _commands(
    session_factory: async_sessionmaker[AsyncSession], command_type: str
) -> list[OutboxMessage]:
    async with session_factory() as session:
        rows = await session.scalars(
            select(OutboxMessage)
            .where(OutboxMessage.command_type == command_type)
            .order_by(OutboxMessage.available_at)
        )
        return list(rows.all())


def test_paid_through_counts_whole_months_and_clamps_short_ones() -> None:
    assert paid_through(date(2026, 1, 1), 2) == date(2026, 2, 28)
    assert paid_through(date(2026, 1, 15), 1) == date(2026, 2, 14)
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert paid_through(date(2026, 11, 10), 3) == date(2027, 2, 9)


@pytest.mark.asyncio
async def test_a_prepay_starts_a_plan_a_retry_records_once_and_later_payments_extend_it(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory)
    owner = portal.application.owner

    async def pay(key: str, months: int, amount: int) -> QuoteSubscriptionRecord:
        return await owner.record_plan_payment(
            context,
            quote_id,
            key=key,
            paid_on=today,
            method=PaymentMethod.CHECK,
            months=months,
            amount_minor=amount,
            note="check #1043",
        )

    # A discounted two-month prepay: the owner's amount is what is recorded.
    plan = await pay("first-payment-key", 2, 18_000)
    assert plan.is_manual and plan.status == "active"
    assert (plan.paid_from, plan.paid_through) == (today, paid_through(today, 2))
    assert (plan.amount_minor, plan.interval) == (9_900, BillingInterval.MONTH)
    assert await _status(session_factory, portal) == "paid"
    [row] = await _rows(session_factory, portal)
    assert (row.kind, row.source, row.method, row.amount_cents, row.months_covered) == (
        "plan",
        "manual",
        "check",
        18_000,
        2,
    )
    assert row.recorded_by == OWNER

    # A double-click or retry with the same key changes nothing.
    again = await pay("first-payment-key", 2, 18_000)
    assert again.paid_through == plan.paid_through
    assert len(await _rows(session_factory, portal)) == 1

    later = await pay("second-payment-key", 1, 9_900)
    assert later.subscription_id == plan.subscription_id
    assert (later.paid_from, later.paid_through) == (today, paid_through(today, 3))
    assert len(await _rows(session_factory, portal)) == 2

    # Paid this month: each check in full, in the month it arrived.
    async with session_factory() as session:
        payments = await SqlPaymentLedgerRepository(session).list_for_business(portal.business_id)
    zone = business_zone(context.business.timezone) or UTC
    noon = datetime.combine(today, time(12), tzinfo=zone)
    assert month_totals(payments, zone, noon) == {"USD": 27_900}

    receipts = await _commands(session_factory, MANUAL_RECEIPT_COMMAND_TYPE)
    assert len(receipts) == 2
    body = str(receipts[-1].payload["body"])
    assert "USD 99.00 by check" in body
    # The paid-through line is added when the receipt goes out.
    assert receipts[-1].payload["subscription_id"] == str(plan.subscription_id)
    assert "1043" not in body


@pytest.mark.asyncio
async def test_undo_moves_paid_through_back_and_the_last_undo_ends_the_plan(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory)
    owner = portal.application.owner
    for key, months in (("undo-first-key", 2), ("undo-second-key", 1)):
        await owner.record_plan_payment(
            context,
            quote_id,
            key=key,
            paid_on=today,
            method=PaymentMethod.CASH,
            months=months,
            amount_minor=9_900 * months,
        )
    first, second = await _rows(session_factory, portal)

    plan = await owner.void_plan_payment(context, quote_id, str(second.id))
    assert plan.status == "active" and plan.paid_through == paid_through(today, 2)
    # Re-sending the undone payment's form is not a new payment.
    with pytest.raises(OwnerConflictError):
        await owner.record_plan_payment(
            context,
            quote_id,
            key="undo-second-key",
            paid_on=today,
            method=PaymentMethod.CASH,
            months=1,
            amount_minor=9_900,
        )
    assert len(await _rows(session_factory, portal)) == 2
    with pytest.raises(OwnerConflictError):
        await owner.void_plan_payment(context, quote_id, str(second.id))
    with pytest.raises(OwnerNotFoundError):
        await owner.void_plan_payment(context, quote_id, "not-a-payment")

    plan = await owner.void_plan_payment(context, quote_id, str(first.id))
    assert plan.status == "canceled" and plan.paid_through is None
    # Back to sent, as before the first payment; nothing is deleted.
    assert await _status(session_factory, portal) is None
    rows = await _rows(session_factory, portal)
    assert len(rows) == 2 and all(r.voided_by == OWNER and r.voided_at for r in rows)

    restarted = await owner.record_plan_payment(
        context,
        quote_id,
        key="undo-restart-key",
        paid_on=today,
        method=PaymentMethod.CHECK,
        months=1,
        amount_minor=9_900,
    )
    assert restarted.subscription_id == plan.subscription_id
    assert restarted.status == "active" and restarted.paid_through == paid_through(today, 1)


@pytest.mark.asyncio
async def test_only_sent_recurring_quotes_of_the_owners_business_get_a_plan(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory)
    owner = portal.application.owner

    async def pay(ctx: OwnerContext, qid: str, **changes: object) -> QuoteSubscriptionRecord:
        fields: dict[str, object] = {
            "key": "refusal-key-0001",
            "paid_on": today,
            "method": PaymentMethod.CHECK,
            "months": 1,
            "amount_minor": 9_900,
        }
        fields.update(changes)
        return await owner.record_plan_payment(ctx, qid, **fields)  # type: ignore[arg-type]

    for bad in (
        {"method": PaymentMethod.CARD},
        {"months": 0},
        {"months": 25},
        {"amount_minor": -1},
        {"paid_on": today + timedelta(days=2)},
    ):
        with pytest.raises(OwnerInputError):
            await pay(context, quote_id, **bad)

    other, other_context, other_quote, _ = await _plan_quote(session_factory)
    with pytest.raises(OwnerNotFoundError):
        await pay(other_context, quote_id)
    await pay(context, quote_id)
    [row] = await _rows(session_factory, portal)
    with pytest.raises(OwnerNotFoundError):
        await other.application.owner.void_plan_payment(other_context, quote_id, str(row.id))
    assert await _rows(session_factory, other) == []

    # A card plan already bills the other quote: no manual plan beside it.
    async with session_factory() as session:
        quote = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == other.business_id)
        )
        assert quote is not None and quote.customer_id is not None
        now = datetime.now(UTC)
        await SqlQuoteSubscriptionRepository(session).create(
            QuoteSubscriptionRecord(
                subscription_id=SubscriptionId(quote.id),
                business_id=other.business_id,
                quote_id=QuoteId(quote.id),
                customer_id=CustomerId(quote.customer_id),
                provider="stripe",
                subscription_ref="sub_card",
                status="active",
                interval=BillingInterval.MONTH,
                amount_minor=9_900,
                currency="USD",
                current_period_end=None,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    with pytest.raises(OwnerConflictError, match="card"):
        await pay(other_context, other_quote, key="card-plan-key-01")

    one_off, one_off_context, one_off_quote, _ = await _plan_quote(session_factory, recurring=False)
    with pytest.raises(OwnerConflictError, match="recurring"):
        await pay(one_off_context, one_off_quote)


@pytest.mark.asyncio
async def test_the_owner_is_nudged_a_week_before_only_for_the_current_date(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory)
    owner = portal.application.owner
    zone = business_zone(context.business.timezone) or UTC
    plan = await owner.record_plan_payment(
        context,
        quote_id,
        key="nudge-first-key",
        paid_on=today,
        method=PaymentMethod.CHECK,
        months=1,
        amount_minor=9_900,
    )
    assert plan.paid_through is not None
    [nudge] = await _commands(session_factory, PLAN_NUDGE_COMMAND_TYPE)
    expected = datetime.combine(plan.paid_through - timedelta(days=7), time(9), tzinfo=zone)
    assert nudge.available_at.replace(tzinfo=nudge.available_at.tzinfo or UTC) == expected

    later = await owner.record_plan_payment(
        context,
        quote_id,
        key="nudge-second-key",
        paid_on=today,
        method=PaymentMethod.CHECK,
        months=1,
        amount_minor=9_900,
    )
    assert later.paid_through is not None
    assert len(await _commands(session_factory, PLAN_NUDGE_COMMAND_TYPE)) == 2

    # Make the nudges due (and nothing else), then let the worker run.
    now = datetime.now(UTC)
    async with session_factory() as session:
        await session.execute(update(OutboxMessage).values(available_at=now + timedelta(days=1)))
        await session.execute(
            update(OutboxMessage)
            .where(OutboxMessage.command_type == PLAN_NUDGE_COMMAND_TYPE)
            .values(available_at=now - timedelta(minutes=1))
        )
        await session.commit()
    await immediate_worker(portal.application).drain()
    through = later.paid_through
    [text] = texts_of(portal.owner_replies, "Jane Doe's plan")
    assert f"paid through {through:%b} {through.day}." in text


@pytest.mark.asyncio
async def test_the_owner_api_validates_and_scopes_plan_payments(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory, approve=True)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        [quote] = (await http.get("/v1/owner/quotes", headers=bearer(owner))).json()["quotes"]
        url = f"/v1/owner/quotes/{quote['id']}/plan-payments"
        body = {
            "key": "api-plan-key-01",
            "paidOn": (date.today() - timedelta(days=1)).isoformat(),
            "method": "check",
            "months": 2,
            "amountCents": 18_000,
        }
        assert (await http.post(url, json=body)).status_code == 401
        for bad in ({"key": "short"}, {"months": 0}, {"amountCents": -5}, {"method": "card"}):
            response = await http.post(url, json={**body, **bad}, headers=bearer(owner))
            assert response.status_code == 422, bad
        # A one-off quote has no plan.
        assert (await http.post(url, json=body, headers=bearer(owner))).status_code == 409
        unknown = await http.post(
            "/v1/owner/quotes/gvq_unknown/plan-payments", json=body, headers=bearer(owner)
        )
        assert unknown.status_code == 404
        undo = await http.post(f"{url}/00000000-0000-0000-0000-000000000000/undo")
        assert undo.status_code == 401
        undo = await http.post(
            f"{url}/00000000-0000-0000-0000-000000000000/undo", headers=bearer(owner)
        )
        assert undo.status_code == 404


def _effects(
    session_factory: async_sessionmaker[AsyncSession], sent: list[dict[str, object]]
) -> ManualPaymentEffectsService:
    async def receipts(_: BusinessId, payload: Mapping[str, object]) -> None:
        sent.append(dict(payload))

    return ManualPaymentEffectsService(
        SqlUnitOfWorkFactory(session_factory), checkout=None, receipts=receipts
    )


@pytest.mark.asyncio
async def test_a_receipt_shows_the_paid_through_date_as_it_is_when_it_goes_out(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory)
    owner = portal.application.owner
    for key, months in (("receipt-first-key", 2), ("receipt-second-key", 1)):
        await owner.record_plan_payment(
            context,
            quote_id,
            key=key,
            paid_on=today,
            method=PaymentMethod.CHECK,
            months=months,
            amount_minor=9_900 * months,
        )
    first, second = await _rows(session_factory, portal)
    await owner.void_plan_payment(context, quote_id, str(first.id))
    payloads = {
        str(r.payload["payment_id"]): r.payload
        for r in await _commands(session_factory, MANUAL_RECEIPT_COMMAND_TYPE)
        if isinstance(r.payload, dict)
    }
    sent: list[dict[str, object]] = []
    effects = _effects(session_factory, sent)

    assert await effects.send_receipt(portal.business_id, payloads[str(first.id)]) == "voided"
    assert await effects.send_receipt(portal.business_id, payloads[str(second.id)]) == "sent"
    [receipt] = sent
    now_through = paid_through(today, 1)
    body = str(receipt["body"])
    assert f"paid through {now_through:%B} {now_through.day}, {now_through.year}." in body
    stale = paid_through(today, 3)
    assert f"{stale:%B} {stale.day}, {stale.year}" not in body


@pytest.mark.asyncio
async def test_a_nudge_that_runs_after_the_plan_lapsed_is_dropped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory)
    plan = await portal.application.owner.record_plan_payment(
        context,
        quote_id,
        key="lapsed-nudge-key",
        paid_on=today,
        method=PaymentMethod.CHECK,
        months=1,
        amount_minor=9_900,
    )
    effects = _effects(session_factory, [])
    lapsed = today - timedelta(days=2)
    async with session_factory() as session:
        await session.execute(
            update(QuoteSubscription)
            .where(QuoteSubscription.business_id == portal.business_id)
            .values(paid_through=lapsed)
        )
        await session.commit()
    payload = {"subscription_id": str(plan.subscription_id), "paid_through": lapsed.isoformat()}
    assert await effects.nudge_plan(portal.business_id, payload) == "lapsed"
    assert texts_of(portal.owner_replies, "Jane Doe's plan") == []


@pytest.mark.asyncio
async def test_a_card_plan_racing_a_manual_one_is_kept_uncounted_and_the_owner_told(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory, checkout=CheckoutFake())
    owner = portal.application.owner
    async with client(portal, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as http:
        assert (await http.post(f"/v1/quotes/{portal.claim_token}/accept")).status_code == 200
        await owner.record_plan_payment(
            context,
            quote_id,
            key="race-manual-key",
            paid_on=today,
            method=PaymentMethod.CHECK,
            months=2,
            amount_minor=19_800,
        )
        # The customer was already on Stripe's page and subscribed anyway.
        completed = await post_event(http, subscription_checkout_event())
        assert completed.status_code == 200
    rows = await _rows(session_factory, portal)
    [manual] = [r for r in rows if r.source == "manual"]
    [card] = [r for r in rows if r.source == "stripe"]
    assert not manual.duplicate and card.duplicate
    async with session_factory() as session:
        plans = (
            await session.scalars(
                select(QuoteSubscription).where(QuoteSubscription.business_id == portal.business_id)
            )
        ).all()
    assert sorted(p.provider for p in plans) == ["manual", "stripe"]
    await immediate_worker(portal.application).drain()
    assert any("also started a card plan" in str(m) for _, m in portal.owner_replies.sent)
    with pytest.raises(OwnerConflictError, match="card"):
        await owner.record_plan_payment(
            context,
            quote_id,
            key="race-second-key",
            paid_on=today,
            method=PaymentMethod.CHECK,
            months=1,
            amount_minor=9_900,
        )

    # Undoing the check leaves the card payment as the one that counts, even
    # once the card plan has ended.
    async with session_factory() as session:
        await session.execute(
            update(QuoteSubscription)
            .where(QuoteSubscription.provider == "stripe")
            .where(QuoteSubscription.business_id == portal.business_id)
            .values(status="canceled")
        )
        await session.commit()
    await owner.void_plan_payment(context, quote_id, str(manual.id))
    after = {r.source: r for r in await _rows(session_factory, portal)}
    assert after["manual"].voided_at is not None and not after["stripe"].duplicate
    assert await _status(session_factory, portal) == "paid"


@pytest.mark.asyncio
async def test_a_plan_works_for_a_quote_with_no_customer_email(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal, context, quote_id, today = await _plan_quote(session_factory)
    async with session_factory() as session:
        row = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == portal.business_id)
        )
        assert row is not None and row.draft is not None
        draft = dict(row.draft)
        original = dict(draft["recipient"])  # type: ignore[arg-type]
        recipient = dict(original)
        recipient.update(address_kind=RecipientAddressKind.PHONE.value, address="+15415550123")
        draft["recipient"] = recipient
        row.draft = draft
        row.customer_id = None
        await session.commit()
    owner = portal.application.owner
    for key, months in (("no-email-first", 2), ("no-email-second", 1)):
        plan = await owner.record_plan_payment(
            context,
            quote_id,
            key=key,
            paid_on=today,
            method=PaymentMethod.CASH,
            months=months,
            amount_minor=9_900 * months,
        )
    assert plan.customer_id is None and plan.paid_through == paid_through(today, 3)
    assert len(await _rows(session_factory, portal)) == 2
    assert await _commands(session_factory, MANUAL_RECEIPT_COMMAND_TYPE) == []
    assert await _status(session_factory, portal) == "paid"
    payload = {
        "subscription_id": str(plan.subscription_id),
        "paid_through": plan.paid_through.isoformat(),
    }
    assert await _effects(session_factory, []).nudge_plan(portal.business_id, payload) == "sent"

    # Once the quote has an e-mail, the next payment links the plan to the customer.
    async with session_factory() as session:
        row = await session.scalar(
            select(QuoteRecord).where(QuoteRecord.business_id == portal.business_id)
        )
        assert row is not None and row.draft is not None
        row.draft = {**row.draft, "recipient": original}
        await session.commit()
    await owner.record_plan_payment(
        context,
        quote_id,
        key="no-email-third",
        paid_on=today,
        method=PaymentMethod.CASH,
        months=1,
        amount_minor=9_900,
    )
    async with session_factory() as session:
        stored = await session.scalar(
            select(QuoteSubscription).where(QuoteSubscription.business_id == portal.business_id)
        )
    assert stored is not None and stored.customer_id is not None

    other = await _plan_quote(session_factory)
    with pytest.raises(OwnerNotFoundError):
        await owner.record_plan_payment(
            other[1],
            quote_id,
            key="no-email-other-business",
            paid_on=today,
            method=PaymentMethod.CASH,
            months=1,
            amount_minor=9_900,
        )
