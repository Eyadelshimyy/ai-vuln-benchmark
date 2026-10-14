"""Payments ledger: one row per settled payment, on the provider's settle
day, counted once, and only ever read for the signed-in owner's business."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.identifiers import BusinessId, QuoteId
from gvas.domain.payments import (
    LedgerPayment,
    PaymentKind,
    PaymentMethod,
    PaymentSource,
    month_totals,
)
from gvas.domain.quotes import public_quote_id
from gvas.infrastructure.models import QuoteRecord
from gvas.infrastructure.payment_models import LedgerPaymentRow
from gvas.infrastructure.payment_repositories import SqlPaymentLedgerRepository
from gvas.infrastructure.stripe import StripeWebhookVerifier
from gvas.infrastructure.stripe.signature import SIGNATURE_HEADER
from test_customer_portal import (
    BillingFake,
    client,
    invoice_event,
    portal_business,
    post_event,
    subscription_checkout_event,
)
from test_hosted_quotes import (
    SESSION_ID,
    WEBHOOK_SECRET,
    CheckoutFake,
    checkout_event,
    hosted_quote,
    public_client,
    sign,
)
from test_owner_dashboard import bearer, http_client, owner_business, sign_in

SETTLED = datetime(2026, 1, 1, 8, 30, tzinfo=UTC)


def _event(**kwargs: object) -> bytes:
    body = json.loads(checkout_event(**kwargs))  # type: ignore[arg-type]
    body["created"] = int(SETTLED.timestamp())
    return json.dumps(body).encode()


def _payment(
    business_id: BusinessId,
    quote_id: QuoteId,
    *,
    reference: str,
    paid_at: datetime,
    kind: PaymentKind = PaymentKind.ONE_OFF,
    amount: int = 25_000,
    voided: bool = False,
) -> LedgerPayment:
    return LedgerPayment(
        payment_id=uuid4(),
        business_id=business_id,
        quote_id=quote_id,
        kind=kind,
        source=PaymentSource.MANUAL,
        method=PaymentMethod.CHECK,
        reference=reference,
        amount_minor=amount,
        currency="usd",
        paid_at=paid_at,
        recorded_by="owner@example.test",
        recorded_at=paid_at,
        voided_at=paid_at if voided else None,
        voided_by="owner@example.test" if voided else None,
    )


async def _quote_of(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId | None = None
) -> tuple[BusinessId, QuoteId]:
    async with session_factory() as session:
        query = select(QuoteRecord)
        if business_id is not None:
            query = query.where(QuoteRecord.business_id == business_id)
        row = await session.scalar(query)
    assert row is not None
    return BusinessId(row.business_id), QuoteId(row.id)


@pytest.mark.asyncio
async def test_a_card_payment_is_recorded_once_on_stripes_settle_day(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=CheckoutFake())
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
        for body in (
            _event(),
            _event(),
            _event(event_id="evt_2", event_type="checkout.session.async_payment_succeeded"),
        ):
            response = await client.post(
                "/webhooks/stripe", content=body, headers={SIGNATURE_HEADER: sign(body)}
            )
            assert response.status_code == 200

    async with session_factory() as session:
        rows = (await session.scalars(select(LedgerPaymentRow))).all()
    assert len(rows) == 1
    row = rows[0]
    assert (row.kind, row.source, row.method, row.reference) == (
        "one_off",
        "stripe",
        "card",
        SESSION_ID,
    )
    assert (row.amount_cents, row.currency, row.months_covered) == (25_000, "USD", None)
    assert row.paid_at.replace(tzinfo=row.paid_at.tzinfo or UTC) == SETTLED
    assert row.voided_at is None and not row.duplicate


@pytest.mark.asyncio
async def test_a_second_one_off_payment_is_kept_but_flagged_and_never_counted(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await hosted_quote(session_factory, checkout=CheckoutFake())
    business_id, quote_id = await _quote_of(session_factory)
    async with session_factory() as session:
        ledger = SqlPaymentLedgerRepository(session)
        first = await ledger.record(
            _payment(business_id, quote_id, reference="manual-1", paid_at=SETTLED)
        )
        again = await ledger.record(
            _payment(business_id, quote_id, reference="manual-1", paid_at=SETTLED)
        )
        second = await ledger.record(
            _payment(business_id, quote_id, reference="manual-2", paid_at=SETTLED)
        )
        renewals = [
            await ledger.record(
                _payment(
                    business_id,
                    quote_id,
                    reference=f"in_{n}",
                    paid_at=SETTLED + timedelta(days=n),
                    kind=PaymentKind.PLAN,
                    amount=7_425,
                )
            )
            for n in (1, 2)
        ]
        await session.commit()
        payments = await ledger.list_for_business(business_id)

    assert first is not None and not first.duplicate
    assert again is None
    assert second is not None and second.duplicate
    assert all(renewal is not None and not renewal.duplicate for renewal in renewals)
    assert len(payments) == 4
    assert month_totals(payments, UTC, SETTLED) == {"USD": 25_000 + 2 * 7_425}


@pytest.mark.asyncio
async def test_owner_payments_and_paid_on_are_scoped_to_the_owners_business(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    app_a, business_a = await owner_business(session_factory, approve=True)
    _, business_b = await owner_business(
        session_factory,
        public_key="gvb_owner_b",
        owner_email="owner-b@example.test",
        customer_email="bob@example.test",
    )
    _, quote_a = await _quote_of(session_factory, business_a)
    _, quote_b = await _quote_of(session_factory, business_b)
    now = datetime.now(UTC)
    async with session_factory() as session:
        ledger = SqlPaymentLedgerRepository(session)
        await ledger.record(_payment(business_a, quote_a, reference="a-paid", paid_at=now))
        await ledger.record(
            _payment(
                business_a,
                quote_a,
                reference="a-old",
                paid_at=now - timedelta(days=400),
                kind=PaymentKind.PLAN,
            )
        )
        await ledger.record(
            _payment(
                business_a,
                quote_a,
                reference="a-void",
                paid_at=now,
                kind=PaymentKind.PLAN,
                voided=True,
            )
        )
        await ledger.record(
            _payment(business_b, quote_b, reference="b-paid", paid_at=now, amount=99_900)
        )
        await session.commit()

    async with http_client(app_a) as http:
        owner_a = await sign_in(session_factory, http, business_a)
        response = await http.get("/v1/owner/payments", headers=bearer(owner_a))
        assert response.status_code == 200
        body = response.json()
        assert {row["quoteId"] for row in body["payments"]} == {public_quote_id(quote_a)}
        assert len(body["payments"]) == 3
        assert body["paidThisMonth"] == {"USD": 25_000}
        assert body["month"] == now.strftime("%Y-%m")
        voided = next(row for row in body["payments"] if row["voidedAt"] is not None)
        assert voided["counts"] is False and voided["voidedBy"] == "owner@example.test"

        quotes = (await http.get("/v1/owner/quotes", headers=bearer(owner_a))).json()["quotes"]
        assert quotes[0]["paidOn"] is not None
        assert datetime.fromisoformat(quotes[0]["paidOn"]) == now - timedelta(days=400)

        customer = await http.get("/v1/owner/payments")
        assert customer.status_code == 401


@pytest.mark.asyncio
async def test_each_plan_renewal_invoice_is_one_ledger_row_on_its_settle_day(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    portal = await portal_business(
        session_factory, recurring=True, checkout=CheckoutFake(), billing=BillingFake()
    )
    renewal = json.loads(
        invoice_event(event_id="evt_renew", event_type="invoice.paid", amount_paid=7_425)
    )
    renewal["created"] = int(SETTLED.timestamp())
    retried = {**renewal, "id": "evt_renew_again"}
    async with client(portal, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as http:
        assert (await http.post(f"/v1/quotes/{portal.claim_token}/accept")).status_code == 200
        assert (await post_event(http, subscription_checkout_event())).status_code == 200
        for body in (renewal, renewal, retried):
            response = await post_event(http, json.dumps(body).encode())
            assert response.status_code == 200

    async with session_factory() as session:
        payments = await SqlPaymentLedgerRepository(session).list_for_business(portal.business_id)
    by_reference = {p.reference: p for p in payments}
    assert len(payments) == 2 and set(by_reference) == {SESSION_ID, "in_1"}
    renewed = by_reference["in_1"]
    assert (renewed.kind, renewed.amount_minor, renewed.months_covered) == (
        PaymentKind.PLAN,
        7_425,
        1,
    )
    assert renewed.paid_at == SETTLED
    started = by_reference[SESSION_ID]
    assert (started.kind, started.amount_minor, started.months_covered) == (
        PaymentKind.PLAN,
        9_900,
        1,
    )
    assert all(p.source is PaymentSource.STRIPE and p.counts for p in payments)


@pytest.mark.asyncio
async def test_the_payment_that_settled_first_counts_even_when_its_webhook_comes_last(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await hosted_quote(session_factory, checkout=CheckoutFake())
    business_id, quote_id = await _quote_of(session_factory)
    later = SETTLED + timedelta(days=40)
    async with session_factory() as session:
        ledger = SqlPaymentLedgerRepository(session)
        arrived_first = await ledger.record(
            _payment(business_id, quote_id, reference="late-settle", paid_at=later)
        )
        arrived_last = await ledger.record(
            _payment(business_id, quote_id, reference="early-settle", paid_at=SETTLED)
        )
        await session.commit()
        payments = await ledger.list_for_business(business_id)

    assert arrived_first is not None and arrived_last is not None
    assert not arrived_last.duplicate
    by_reference = {p.reference: p for p in payments}
    assert by_reference["early-settle"].counts
    assert by_reference["late-settle"].duplicate
    assert month_totals(payments, UTC, SETTLED) == {"USD": 25_000}
    assert month_totals(payments, UTC, later) == {}


@pytest.mark.parametrize(("collected", "expected"), [(0, None), (20_000, 20_000)])
@pytest.mark.asyncio
async def test_the_ledger_records_what_the_checkout_actually_collected(
    session_factory: async_sessionmaker[AsyncSession], collected: int, expected: int | None
) -> None:
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=CheckoutFake())
    body = json.loads(_event(payment_status="paid" if collected else "no_payment_required"))
    body["data"]["object"]["amount_total"] = collected
    raw = json.dumps(body).encode()
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
        response = await client.post(
            "/webhooks/stripe", content=raw, headers={SIGNATURE_HEADER: sign(raw)}
        )
        assert response.status_code == 200

    async with session_factory() as session:
        rows = (await session.scalars(select(LedgerPaymentRow))).all()
        quote = await session.scalar(select(QuoteRecord))
    assert [row.amount_cents for row in rows] == ([] if expected is None else [expected])
    assert quote is not None and quote.customer_status == "paid"
