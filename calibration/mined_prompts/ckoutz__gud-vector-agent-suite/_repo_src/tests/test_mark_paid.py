"""Mark paid: a check, cash or other payment for a sent one-off quote,
one active payment per quote, an undo with a trail, the open card checkout
closed, a card payment that races it flagged, and a receipt e-mail."""

import json
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import CustomerDeliveryFake
from gvas.application.manual_payments import ManualPaymentEffectsService
from gvas.application.owner import (
    OwnerConflictError,
    OwnerContext,
    OwnerInputError,
    OwnerNotFoundError,
)
from gvas.composition import Application
from gvas.domain.identifiers import BusinessId
from gvas.domain.owner import OwnerSession
from gvas.domain.payments import (
    CHECKOUT_EXPIRE_COMMAND_TYPE,
    MANUAL_RECEIPT_COMMAND_TYPE,
    PaymentCheckoutClosedError,
    PaymentCheckoutError,
    PaymentMethod,
)
from gvas.domain.quotes import public_quote_id
from gvas.infrastructure.models import Business, OutboxMessage, QuoteRecord
from gvas.infrastructure.payment_models import LedgerPaymentRow, QuotePayment
from gvas.infrastructure.repositories import SqlBusinessRepository
from gvas.infrastructure.stripe import StripeWebhookVerifier
from gvas.infrastructure.stripe.api import StripeCheckout
from gvas.infrastructure.stripe.config import StripeSettings
from gvas.infrastructure.stripe.signature import SIGNATURE_HEADER
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
from test_payments_ledger import _quote_of
from test_pilot_runtime import immediate_worker

OWNER = "owner@example.test"
PAID_ON = date(2025, 12, 1)


async def _context(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> OwnerContext:
    async with session_factory() as session:
        business = await SqlBusinessRepository(session).get(business_id)
    assert business is not None
    now = datetime.now(UTC)
    return OwnerContext(
        session=OwnerSession(
            token_hash="0" * 64,
            business_id=business_id,
            email=OWNER,
            expires_at=now + timedelta(days=1),
            created_at=now,
        ),
        business=business,
    )


async def _accepted(
    session_factory: async_sessionmaker[AsyncSession], checkout: CheckoutFake
) -> tuple[Application, OwnerContext, str, str]:
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
    business_id, quote_id = await _quote_of(session_factory)
    context = await _context(session_factory, business_id)
    return application, context, public_quote_id(quote_id), claim_token


async def _rows(session_factory: async_sessionmaker[AsyncSession]) -> list[LedgerPaymentRow]:
    async with session_factory() as session:
        return list((await session.scalars(select(LedgerPaymentRow))).all())


async def _quote_status(session_factory: async_sessionmaker[AsyncSession]) -> str | None:
    async with session_factory() as session:
        row = await session.scalar(select(QuoteRecord))
    assert row is not None
    status: str | None = row.customer_status
    return status


@pytest.mark.asyncio
async def test_marking_paid_records_one_manual_payment_closes_checkout_and_emails_a_receipt(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    application, context, quote_id, _ = await _accepted(session_factory, checkout)

    quote = await application.owner.mark_paid(
        context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK, note=" check #1042 "
    )

    assert quote.customer_status is not None and quote.customer_status.value == "paid"
    assert await _quote_status(session_factory) == "paid"
    [row] = await _rows(session_factory)
    assert (row.source, row.method, row.kind, row.amount_cents) == (
        "manual",
        "check",
        "one_off",
        25_000,
    )
    assert row.recorded_by == OWNER and row.note == "check #1042"
    assert row.paid_at.replace(tzinfo=row.paid_at.tzinfo or UTC).date() == PAID_ON
    assert row.voided_at is None and not row.duplicate

    # Closing the card checkout is the worker's job, so an outage is retried.
    assert checkout.expired == []
    async with session_factory() as session:
        payment = await session.scalar(select(QuotePayment))
        receipts = (
            await session.scalars(
                select(OutboxMessage).where(
                    OutboxMessage.command_type == MANUAL_RECEIPT_COMMAND_TYPE
                )
            )
        ).all()
    assert payment is not None and payment.status == "expired"
    [receipt] = receipts
    assert receipt.payload["to"] == "jane@example.test"
    body = str(receipt.payload["body"])
    assert "USD 250.00 by check on December 1, 2025" in body
    # The owner's private note stays out of the customer's receipt.
    assert "1042" not in body


@pytest.mark.asyncio
async def test_one_active_payment_per_quote_and_only_accepted_one_off_quotes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, _ = await _accepted(session_factory, CheckoutFake())
    owner = application.owner

    with pytest.raises(OwnerInputError):
        await owner.mark_paid(
            context, quote_id, paid_on=date(2030, 1, 1), method=PaymentMethod.CASH
        )
    with pytest.raises(OwnerInputError):
        await owner.mark_paid(context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CARD)
    with pytest.raises(OwnerInputError):
        await owner.mark_paid(
            context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CASH, note="x" * 501
        )
    await owner.mark_paid(context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CASH)
    with pytest.raises(OwnerConflictError):
        await owner.mark_paid(context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CASH)
    assert len(await _rows(session_factory)) == 1
    with pytest.raises(OwnerNotFoundError):
        await owner.mark_paid(context, "gvq_unknown", paid_on=PAID_ON, method=PaymentMethod.CASH)


@pytest.mark.asyncio
async def test_mark_unpaid_voids_with_a_trail_and_reopens_the_quote(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, _ = await _accepted(session_factory, CheckoutFake())
    await application.owner.mark_paid(
        context, quote_id, paid_on=PAID_ON, method=PaymentMethod.OTHER
    )

    quote = await application.owner.mark_unpaid(context, quote_id)

    assert quote.customer_status is not None and quote.customer_status.value == "accepted"
    assert await _quote_status(session_factory) == "accepted"
    [row] = await _rows(session_factory)
    assert row.voided_at is not None and row.voided_by == OWNER
    with pytest.raises(OwnerConflictError):
        await application.owner.mark_unpaid(context, quote_id)
    # Marked again after the undo: a fresh payment, the voided one kept.
    await application.owner.mark_paid(
        context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
    )
    rows = await _rows(session_factory)
    assert len(rows) == 2 and sum(r.voided_at is None for r in rows) == 1


@pytest.mark.asyncio
async def test_a_card_payment_cannot_be_marked_unpaid(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, claim_token = await _accepted(session_factory, CheckoutFake())
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        body = checkout_event()
        response = await client.post(
            "/webhooks/stripe", content=body, headers={SIGNATURE_HEADER: sign(body)}
        )
        assert response.status_code == 200
    with pytest.raises(OwnerConflictError, match="Stripe"):
        await application.owner.mark_unpaid(context, quote_id)
    with pytest.raises(OwnerConflictError):
        await application.owner.mark_paid(
            context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
        )


@pytest.mark.asyncio
async def test_a_card_payment_racing_a_manual_one_is_flagged_and_the_owner_told(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, owner_replies, _, _ = await hosted_quote(session_factory, checkout=CheckoutFake())
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        claim = await session_claim(session_factory)
        assert (await client.post(f"/v1/quotes/{claim}/accept")).status_code == 200
        business_id, quote = await _quote_of(session_factory)
        context = await _context(session_factory, business_id)
        quote_id = public_quote_id(quote)
        await application.owner.mark_paid(
            context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
        )
        # The customer was already on Stripe's page and paid anyway.
        body = checkout_event()
        response = await client.post(
            "/webhooks/stripe", content=body, headers={SIGNATURE_HEADER: sign(body)}
        )
        assert response.status_code == 200
    await immediate_worker(application).drain()

    assert await _quote_status(session_factory) == "paid"
    rows = {row.source: row for row in await _rows(session_factory)}
    assert not rows["manual"].duplicate and rows["stripe"].duplicate
    assert any("paid twice" in str(message) for _, message in owner_replies.sent)

    # Undoing the check leaves the card payment as the one that counts.
    await application.owner.mark_unpaid(context, quote_id)
    rows = {row.source: row for row in await _rows(session_factory)}
    assert rows["manual"].voided_at is not None and not rows["stripe"].duplicate
    assert await _quote_status(session_factory) == "paid"


async def session_claim(session_factory: async_sessionmaker[AsyncSession]) -> str:
    async with session_factory() as session:
        row = await session.scalar(select(QuoteRecord))
    assert row is not None and row.claim_token is not None
    return str(row.claim_token)


@pytest.mark.asyncio
async def test_another_business_owner_cannot_mark_or_unmark_the_quote(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, _ = await _accepted(session_factory, CheckoutFake())
    _, other_business = await owner_business(session_factory)
    other = await _context(session_factory, other_business)
    with pytest.raises(OwnerNotFoundError):
        await application.owner.mark_paid(
            other, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
        )
    await application.owner.mark_paid(
        context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
    )
    with pytest.raises(OwnerNotFoundError):
        await application.owner.mark_unpaid(other, quote_id)
    [row] = await _rows(session_factory)
    assert row.voided_at is None


@pytest.mark.asyncio
async def test_the_owner_api_validates_and_scopes_mark_paid(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory, approve=True)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        [quote] = (await http.get("/v1/owner/quotes", headers=bearer(owner))).json()["quotes"]
        assert quote["paidOn"] is None and quote["paidBy"] is None
        url = f"/v1/owner/quotes/{quote['id']}/mark-paid"
        payment = {"paidOn": PAID_ON.isoformat(), "method": "check"}

        assert (await http.post(url, json=payment)).status_code == 401
        bad = {"paidOn": PAID_ON.isoformat(), "method": "bitcoin"}
        assert (await http.post(url, json=bad, headers=bearer(owner))).status_code == 422
        future = {"paidOn": "2030-01-01", "method": "cash"}
        assert (await http.post(url, json=future, headers=bearer(owner))).status_code == 422
        unknown = await http.post(
            "/v1/owner/quotes/gvq_unknown/mark-paid", json=payment, headers=bearer(owner)
        )
        assert unknown.status_code == 404
        unpaid_url = f"/v1/owner/quotes/{quote['id']}/mark-unpaid"
        assert (await http.post(unpaid_url, headers=bearer(owner))).status_code == 409
        # Sent but not accepted yet: it can still be marked paid.
        assert (await http.post(url, json=payment, headers=bearer(owner))).status_code == 200
        assert (await http.post(url, json=payment, headers=bearer(owner))).status_code == 409
        assert (await http.post(unpaid_url, headers=bearer(owner))).status_code == 200


@pytest.mark.asyncio
async def test_the_worker_closes_the_checkout_and_sends_the_receipt(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    emails = CustomerDeliveryFake()
    application, _, _, claim_token = await hosted_quote(
        session_factory, checkout=checkout, customer_email=emails
    )
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
    business_id, quote = await _quote_of(session_factory)
    context = await _context(session_factory, business_id)
    await application.owner.mark_paid(
        context, public_quote_id(quote), paid_on=PAID_ON, method=PaymentMethod.CASH
    )
    async with session_factory() as session:
        types = set(await session.scalars(select(OutboxMessage.command_type)))
    assert {CHECKOUT_EXPIRE_COMMAND_TYPE, MANUAL_RECEIPT_COMMAND_TYPE} <= types

    await immediate_worker(application).drain()

    assert checkout.expired == [SESSION_ID]
    [receipt] = [r for r in emails.requests if (r.subject or "").startswith("Receipt")]
    assert "in cash" in (receipt.body_text or "")


@pytest.mark.asyncio
async def test_a_payment_voided_before_delivery_sends_no_receipt(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    emails = CustomerDeliveryFake()
    application, _, _, claim_token = await hosted_quote(
        session_factory, checkout=CheckoutFake(), customer_email=emails
    )
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 200
    business_id, quote = await _quote_of(session_factory)
    context = await _context(session_factory, business_id)
    await application.owner.mark_paid(
        context, public_quote_id(quote), paid_on=PAID_ON, method=PaymentMethod.CHECK
    )
    await application.owner.mark_unpaid(context, public_quote_id(quote))

    await immediate_worker(application).drain()

    assert not [r for r in emails.requests if (r.subject or "").startswith("Receipt")]


class _ExpiryFails(CheckoutFake):
    def __init__(self, error: PaymentCheckoutError) -> None:
        super().__init__()
        self.error = error

    async def expire_checkout(self, session_id: str) -> None:
        raise self.error


@pytest.mark.asyncio
async def test_an_expiry_outage_is_retried_but_an_already_closed_session_is_done(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    def effects(error: PaymentCheckoutError) -> ManualPaymentEffectsService:
        return ManualPaymentEffectsService(
            session_factory,  # type: ignore[arg-type]
            checkout=_ExpiryFails(error),
            receipts=None,
        )

    payload = {"session_id": SESSION_ID}
    with pytest.raises(PaymentCheckoutError):
        await effects(PaymentCheckoutError("unreachable")).expire_checkout(payload)
    closed = effects(PaymentCheckoutClosedError("closed"))
    assert await closed.expire_checkout(payload) == "already closed"


@pytest.mark.asyncio
async def test_a_card_payment_settled_before_the_check_date_still_lets_the_owner_undo_the_check(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, _ = await _accepted(session_factory, CheckoutFake())
    await application.owner.mark_paid(
        context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
    )
    body = json.loads(checkout_event())
    body["created"] = int(datetime(2025, 11, 1, tzinfo=UTC).timestamp())
    raw = json.dumps(body).encode()
    async with public_client(application, verifier=StripeWebhookVerifier(WEBHOOK_SECRET)) as client:
        response = await client.post(
            "/webhooks/stripe", content=raw, headers={SIGNATURE_HEADER: sign(raw)}
        )
        assert response.status_code == 200
    rows = {row.source: row for row in await _rows(session_factory)}
    assert rows["manual"].duplicate and not rows["stripe"].duplicate

    await application.owner.mark_unpaid(context, quote_id)

    rows = {row.source: row for row in await _rows(session_factory)}
    assert rows["manual"].voided_at is not None and not rows["stripe"].duplicate
    assert await _quote_status(session_factory) == "paid"


@pytest.mark.asyncio
async def test_marking_paid_while_checkout_opens_closes_the_new_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)

    async def owner_marks_paid() -> None:
        business_id, quote = await _quote_of(session_factory)
        context = await _context(session_factory, business_id)
        await application.owner.mark_paid(
            context, public_quote_id(quote), paid_on=PAID_ON, method=PaymentMethod.CASH
        )

    checkout.hook = owner_marks_paid
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 409
    async with session_factory() as session:
        payment = await session.scalar(select(QuotePayment))
    assert payment is not None and payment.status == "expired"

    await immediate_worker(application).drain()

    assert checkout.expired == [SESSION_ID]
    assert await _quote_status(session_factory) == "paid"


@pytest.mark.asyncio
async def test_declining_while_checkout_opens_closes_the_new_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    checkout = CheckoutFake()
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=checkout)

    async def customer_declines() -> None:
        async with public_client(application) as other_tab:
            response = await other_tab.post(f"/v1/quotes/{claim_token}/decline")
            assert response.status_code == 200

    checkout.hook = customer_declines
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 409
    await immediate_worker(application).drain()

    assert checkout.expired == [SESSION_ID]
    assert await _quote_status(session_factory) == "declined"


def _stripe_expiry(status_code: int, session_status: str) -> StripeCheckout:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"id": SESSION_ID, "status": session_status})
        return httpx.Response(status_code, json={"error": {"type": "invalid_request_error"}})

    settings = StripeSettings(secret_key="sk_test_secret", webhook_secret="whsec_x")  # noqa: S106
    return StripeCheckout(settings, httpx.AsyncClient(transport=httpx.MockTransport(handle)))


@pytest.mark.asyncio
async def test_stripe_expiry_is_done_only_when_the_session_is_no_longer_open() -> None:
    with pytest.raises(PaymentCheckoutClosedError):
        await _stripe_expiry(400, "complete").expire_checkout(SESSION_ID)
    with pytest.raises(PaymentCheckoutClosedError):
        await _stripe_expiry(404, "open").expire_checkout(SESSION_ID)
    with pytest.raises(PaymentCheckoutError) as raised:
        await _stripe_expiry(400, "open").expire_checkout(SESSION_ID)
    assert not isinstance(raised.value, PaymentCheckoutClosedError)


async def _sent(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[Application, OwnerContext, str, str]:
    application, _, _, claim_token = await hosted_quote(session_factory, checkout=CheckoutFake())
    business_id, quote_id = await _quote_of(session_factory)
    context = await _context(session_factory, business_id)
    return application, context, public_quote_id(quote_id), claim_token


@pytest.mark.asyncio
async def test_a_sent_quote_can_be_marked_paid_and_undo_puts_it_back_to_sent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, claim_token = await _sent(session_factory)
    assert await _quote_status(session_factory) is None

    await application.owner.mark_paid(
        context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
    )
    assert await _quote_status(session_factory) == "paid"
    [row] = await _rows(session_factory)
    assert row.customer_status_before == "sent"
    with pytest.raises(OwnerConflictError):
        await application.owner.mark_paid(
            context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CASH
        )
    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/accept")).status_code == 409

    quote = await application.owner.mark_unpaid(context, quote_id)

    assert quote.customer_status is None
    assert await _quote_status(session_factory) is None


@pytest.mark.asyncio
async def test_a_viewed_quote_goes_back_to_viewed_and_a_declined_one_cannot_be_marked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, claim_token = await _sent(session_factory)
    async with public_client(application) as client:
        assert (await client.get(f"/v1/quotes/{claim_token}")).status_code == 200
    assert await _quote_status(session_factory) == "viewed"

    await application.owner.mark_paid(context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CASH)
    await application.owner.mark_unpaid(context, quote_id)
    assert await _quote_status(session_factory) == "viewed"

    async with public_client(application) as client:
        assert (await client.post(f"/v1/quotes/{claim_token}/decline")).status_code == 200
    with pytest.raises(OwnerConflictError):
        await application.owner.mark_paid(
            context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CASH
        )
    assert sum(r.voided_at is None for r in await _rows(session_factory)) == 0


@pytest.mark.asyncio
async def test_another_business_owner_cannot_mark_a_sent_quote_paid(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, _, quote_id, _ = await _sent(session_factory)
    _, other_business = await owner_business(session_factory)
    intruder = await _context(session_factory, other_business)

    with pytest.raises(OwnerNotFoundError):
        await application.owner.mark_paid(
            intruder, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
        )
    assert await _rows(session_factory) == []
    assert await _quote_status(session_factory) is None


@pytest.mark.asyncio
async def test_a_quote_not_delivered_yet_cannot_be_marked_paid(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, context, quote_id, _ = await _sent(session_factory)
    async with session_factory() as session:
        row = await session.scalar(select(QuoteRecord))
        assert row is not None and row.status in {"delivery_pending", "delivered"}
        row.status = "approved"
        await session.commit()

    with pytest.raises(OwnerConflictError):
        await application.owner.mark_paid(
            context, quote_id, paid_on=PAID_ON, method=PaymentMethod.CHECK
        )
    assert await _rows(session_factory) == []


async def _set_invoice_prefix(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId, prefix: str
) -> None:
    async with session_factory() as session:
        row = await session.get(Business, business_id)
        assert row is not None
        row.invoice_prefix = prefix
        await session.commit()


@pytest.mark.asyncio
async def test_invoices_stay_off_without_a_prefix_and_mark_paid_settles_the_invoice(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, business_id = await owner_business(session_factory, approve=True)
    async with http_client(application) as http:
        owner = await sign_in(session_factory, http, business_id)
        [quote] = (await http.get("/v1/owner/quotes", headers=bearer(owner))).json()["quotes"]
        send_url = f"/v1/owner/quotes/{quote['id']}/send-invoice"

        me = (await http.get("/v1/owner/me", headers=bearer(owner))).json()
        assert me["business"]["invoices"] is False
        off = (await http.get("/v1/owner/invoices", headers=bearer(owner))).json()
        assert off == {"enabled": False, "invoices": []}
        assert (await http.post(send_url, headers=bearer(owner))).status_code == 404

        await _set_invoice_prefix(session_factory, business_id, "LL")
        assert (await http.post(send_url)).status_code == 401
        # Sent, not accepted yet: nothing to invoice.
        assert (await http.post(send_url, headers=bearer(owner))).status_code == 409
        async with session_factory() as session:
            row = await session.scalar(select(QuoteRecord))
            assert row is not None
            row.customer_status = "accepted"
            await session.commit()

        sent = await http.post(send_url, headers=bearer(owner))
        assert sent.status_code == 200
        invoice = sent.json()["invoice"]
        assert invoice["number"] == "LL-1001" and invoice["status"] == "sent"
        assert invoice["totalCents"] == quote["totalCents"]
        assert invoice["quoteId"] == quote["id"]
        assert (await http.post(send_url, headers=bearer(owner))).status_code == 409

        payment = {"paidOn": date.today().isoformat(), "method": "cash"}
        paid_url = f"/v1/owner/quotes/{quote['id']}/mark-paid"
        assert (await http.post(paid_url, json=payment, headers=bearer(owner))).status_code == 200
        listed = (await http.get("/v1/owner/invoices", headers=bearer(owner))).json()
        assert listed["enabled"] is True
        [paid] = listed["invoices"]
        assert paid["status"] == "paid" and paid["paidBy"]["method"] == "cash"
        unpaid_url = f"/v1/owner/quotes/{quote['id']}/mark-unpaid"
        assert (await http.post(unpaid_url, headers=bearer(owner))).status_code == 200
        [again] = (await http.get("/v1/owner/invoices", headers=bearer(owner))).json()["invoices"]
        assert again["status"] == "sent" and again["paidOn"] is None
