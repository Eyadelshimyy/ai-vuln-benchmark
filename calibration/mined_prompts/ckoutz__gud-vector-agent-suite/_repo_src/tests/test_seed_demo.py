"""The demo seed: a believable week for a fictional business, dates relative to now."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.owner import (
    OwnerAuthenticationError,
    OwnerConflictError,
    OwnerContext,
    OwnerNotFoundError,
    OwnerService,
)
from gvas.composition.production import ProductionConfigurationError
from gvas.domain.enums import CustomerQuoteStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.invoices import InvoiceStatus, invoice_status
from gvas.domain.owner import OwnerSession
from gvas.domain.payments import first_counted, month_totals, paid_through
from gvas.domain.quotes import public_quote_id
from gvas.infrastructure.demo import CLOSED_WEEKDAYS, DemoBookedEvents
from gvas.infrastructure.models import Business, Customer, QuoteRecord
from gvas.infrastructure.repositories import SqlBusinessRepository
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory
from gvas.interfaces import seed_demo
from gvas.interfaces.seed_demo import SeedError, mint_owner_sign_in
from test_composition import seed_business

NOW = datetime(2026, 10, 14, 18, 30, tzinfo=UTC)  # a Wednesday, 11:30 in Oakland
ZONE = ZoneInfo("America/Los_Angeles")
OWNER = "owner@larkspur.example"


async def _business(session_factory: async_sessionmaker[AsyncSession]) -> BusinessId:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    async with session_factory() as session:
        await session.execute(
            update(Business)
            .where(Business.id == business_id)
            .values(timezone="America/Los_Angeles", owner_email=OWNER)
        )
        await session.commit()
    return business_id


def _service(session_factory: async_sessionmaker[AsyncSession]) -> OwnerService:
    return OwnerService(
        SqlUnitOfWorkFactory(session_factory),
        booked_events=DemoBookedEvents(session_factory),
        now=lambda: NOW,
    )


async def _context(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> OwnerContext:
    async with session_factory() as session:
        business = await SqlBusinessRepository(session).get(business_id)
    assert business is not None
    return OwnerContext(
        session=OwnerSession(
            token_hash="0" * 64,
            business_id=business_id,
            email=OWNER,
            expires_at=NOW + timedelta(days=1),
            created_at=NOW,
        ),
        business=business,
    )


@pytest.mark.asyncio
async def test_the_seed_gives_the_owner_a_full_week(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await _business(session_factory)
    summary = await seed_demo.seed_demo(session_factory, business_id, reset=False, now=NOW)
    assert (summary.customers, summary.quotes, summary.invoices) == (12, 14, 0)

    service = _service(session_factory)
    context = await _context(session_factory, business_id)
    quotes = await service.quotes(context)
    assert len(quotes) == 14
    assert len([q for q in quotes if q.status.value == "awaiting_approval"]) == 1
    accepted = [q for q in quotes if q.customer_status is CustomerQuoteStatus.ACCEPTED]
    assert sum(q.draft.total_minor for q in accepted if q.draft is not None) == 867_500
    assert all(q.is_claimable() for q in quotes if q.status.value == "delivered")

    payments = await service.payments(context)
    # This month: spring cleanup, the Mark-paid check and the plan prepay;
    # the oak pruning and the sod were paid last month.
    assert month_totals(payments, ZONE, NOW) == {"USD": 64_000 + 27_500 + 54_000}
    assert any(p.paid_at.astimezone(ZONE).month == 9 for p in payments)

    (plan,) = await service.subscriptions(context)
    assert plan.is_manual and plan.paid_from is not None
    assert plan.paid_through == paid_through(plan.paid_from, 3)

    bookings = await service.bookings(context)
    assert [b.collected.name for b in bookings if b.state.value == "awaiting_owner"] == [
        "Olivia Grant"
    ]
    start = datetime(2026, 10, 14, tzinfo=ZONE)
    calendar = await service.calendar(context, start, start + timedelta(days=7))
    today = [e for e in calendar.events if e.start.astimezone(ZONE).date() == start.date()]
    assert len(today) == 2
    assert len(calendar.events) == 6
    assert calendar.problems == ()


@pytest.mark.asyncio
async def test_reset_replaces_only_this_business_and_a_plain_rerun_refuses(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await _business(session_factory)
    other = await _business(session_factory)
    await seed_demo.seed_demo(session_factory, business_id, reset=False, now=NOW)
    await seed_demo.seed_demo(session_factory, other, reset=False, now=NOW)

    with pytest.raises(SeedError, match="--reset"):
        await seed_demo.seed_demo(session_factory, business_id, reset=False, now=NOW)
    later = NOW + timedelta(days=9)
    await seed_demo.seed_demo(session_factory, business_id, reset=True, now=later)

    async with session_factory() as session:
        for business, count in ((business_id, 12), (other, 12)):
            quotes = await session.scalar(
                select(func.count())
                .select_from(QuoteRecord)
                .where(QuoteRecord.business_id == business)
            )
            customers = await session.scalar(
                select(func.count()).select_from(Customer).where(Customer.business_id == business)
            )
            assert (quotes, customers) == (count + 2, count)
        assert await session.get(Business, business_id) is not None
    context = await _context(session_factory, business_id)
    payments = await _service(session_factory).payments(context)
    assert len(payments) == 5
    assert max(p.paid_at for p in payments) <= later
    assert month_totals(payments, ZONE, later) == {"USD": 145_500}


@pytest.mark.asyncio
async def test_an_unknown_business_is_refused(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(SeedError, match="gvas-bootstrap"):
        await seed_demo.seed_demo(session_factory, BusinessId(uuid4()), reset=True, now=NOW)


@pytest.mark.asyncio
async def test_the_printed_sign_in_link_signs_the_owner_in(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await _business(session_factory)
    token = await mint_owner_sign_in(session_factory, business_id, now=NOW)
    signed_in = await _service(session_factory).exchange_login_token(token)
    assert signed_in is not None
    assert signed_in[1].session.email == OWNER
    with pytest.raises(OwnerAuthenticationError):
        await _service(session_factory).exchange_login_token(token)


def test_the_command_refuses_outside_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GVAS_DEMO_MODE", raising=False)
    monkeypatch.setattr(seed_demo, "DemoSettings", lambda: type("S", (), {"mode": False})())
    with pytest.raises(SeedError, match="GVAS_DEMO_MODE"):
        seed_demo.main(["--business-id", str(uuid4()), "--reset"])


def _demo_environment(monkeypatch: pytest.MonkeyPatch, *, isolated: bool = True) -> None:
    def load() -> None:
        if not isolated:
            raise ProductionConfigurationError("a demo must not hold GVAS_SLACK_BOT_TOKEN")

    monkeypatch.setattr(seed_demo, "DemoSettings", lambda: type("S", (), {"mode": True})())
    monkeypatch.setattr(seed_demo, "load_production_settings", load)


def test_the_command_refuses_an_environment_holding_real_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _demo_environment(monkeypatch, isolated=False)
    with pytest.raises(SeedError, match="not an isolated demo"):
        seed_demo.main(["--business-id", str(uuid4()), "--reset"])


def test_a_sign_in_link_must_be_https(monkeypatch: pytest.MonkeyPatch) -> None:
    _demo_environment(monkeypatch)
    with pytest.raises(SeedError, match="https"):
        seed_demo.main(
            ["--business-id", str(uuid4()), "--no-seed", "--sign-in-link", "http://dash.example"]
        )


@pytest.mark.asyncio
async def test_a_plain_seed_refuses_any_existing_data(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await _business(session_factory)
    async with session_factory() as session:
        session.add(Customer(business_id=business_id, email="real@example.com", created_at=NOW))
        await session.commit()
    with pytest.raises(SeedError, match="--reset"):
        await seed_demo.seed_demo(session_factory, business_id, reset=False, now=NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 11, 1, 7, 30, tzinfo=UTC),  # 00:30 on the 1st in Oakland
        datetime(2026, 10, 31, 20, 0, tzinfo=UTC),  # the 31st
        datetime(2026, 10, 18, 19, 0, tzinfo=UTC),  # a Sunday
    ],
)
async def test_dates_hold_up_on_awkward_days(
    session_factory: async_sessionmaker[AsyncSession], now: datetime
) -> None:
    business_id = await _business(session_factory)
    await seed_demo.seed_demo(session_factory, business_id, reset=False, now=now)
    service = OwnerService(
        SqlUnitOfWorkFactory(session_factory),
        booked_events=DemoBookedEvents(session_factory),
        now=lambda: now,
    )
    context = await _context(session_factory, business_id)
    quotes = {quote.quote_id: quote for quote in await service.quotes(context)}
    for payment in await service.payments(context):
        assert payment.paid_at <= now
        assert quotes[payment.quote_id].created_at < payment.paid_at
    slots = []
    for booking in await service.bookings(context):
        assert booking.requested_slot_start is not None
        assert booking.requested_slot_end is not None
        assert booking.requested_slot_start.astimezone(ZONE).weekday() not in CLOSED_WEEKDAYS
        slots.append((booking.requested_slot_start, booking.requested_slot_end))
    slots.sort()
    for (_, earlier_end), (later_start, _) in zip(slots, slots[1:], strict=False):
        assert earlier_end <= later_start


@pytest.mark.asyncio
async def test_with_invoices_on_the_seed_has_eight_and_an_accepted_quote_can_be_invoiced(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await _business(session_factory)
    async with session_factory() as session:
        await session.execute(
            update(Business).where(Business.id == business_id).values(invoice_prefix="LL")
        )
        await session.commit()
    summary = await seed_demo.seed_demo(session_factory, business_id, reset=False, now=NOW)
    assert summary.invoices == 8

    service = _service(session_factory)
    context = await _context(session_factory, business_id)
    invoices = await service.invoices(context)
    assert [i.number for i in invoices] == [f"LL-{n}" for n in range(1041, 1033, -1)]
    paid = first_counted(await service.payments(context))
    today = NOW.astimezone(ZONE).date()
    statuses = [invoice_status(i, paid.get(i.quote_id), today=today) for i in invoices]
    assert statuses.count(InvoiceStatus.PAID) == 4
    assert statuses.count(InvoiceStatus.SENT) == 3
    assert statuses.count(InvoiceStatus.OVERDUE) == 1
    paid_now = [
        i
        for i in invoices
        if i.quote_id in paid and paid[i.quote_id].paid_at.astimezone(ZONE).month == NOW.month
    ]
    assert len(paid_now) == 2
    for invoice in invoices:
        assert (invoice.due_on - invoice.sent_at.astimezone(ZONE).date()).days == 14
        assert invoice.sent_at <= NOW
        assert sum(x.quantity * x.unit_price_minor for x in invoice.line_items) == (
            invoice.total_minor
        )

    quotes = await service.quotes(context)
    invoiced = {i.quote_id for i in invoices}
    (beds,) = [
        q
        for q in quotes
        if q.customer_status is CustomerQuoteStatus.ACCEPTED and q.quote_id not in invoiced
    ]
    sent = await service.send_invoice(context, public_quote_id(beds.quote_id))
    assert sent.number == "LL-1042"
    assert sent.total_minor == 145_500 and len(sent.line_items) == 1
    assert sent.due_on == today + timedelta(days=14)
    with pytest.raises(OwnerConflictError, match="already has an invoice"):
        await service.send_invoice(context, public_quote_id(beds.quote_id))
    (pending,) = [q for q in quotes if q.status.value == "awaiting_approval"]
    with pytest.raises(OwnerConflictError, match="accepted"):
        await service.send_invoice(context, public_quote_id(pending.quote_id))


@pytest.mark.asyncio
async def test_with_invoices_off_there_are_none_and_none_can_be_sent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await _business(session_factory)
    await seed_demo.seed_demo(session_factory, business_id, reset=False, now=NOW)
    service = _service(session_factory)
    context = await _context(session_factory, business_id)
    assert await service.invoices(context) == ()
    accepted = next(
        q
        for q in await service.quotes(context)
        if q.customer_status is CustomerQuoteStatus.ACCEPTED
    )
    with pytest.raises(OwnerNotFoundError):
        await service.send_invoice(context, public_quote_id(accepted.quote_id))
