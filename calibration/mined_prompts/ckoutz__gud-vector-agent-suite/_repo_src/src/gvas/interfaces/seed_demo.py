"""Fill a demo business with fictional customers, quotes, payments and bookings.

Dates are relative to now, in the business's zone, so the dashboard always
looks like a business mid-week: a quote and a booking waiting for the owner,
quotes sent and accepted, payments this month and last, a check-paid plan and
appointments today and this week. Refuses to run unless demo mode is on.

    gvas-seed-demo --business-id <uuid> --reset
    gvas-seed-demo --business-id <uuid> --sign-in-link https://dashboard.example

``--reset`` deletes the business's customers, quotes, payments, bookings and
messages first, keeping the business and its templates. ``--sign-in-link``
prints a single-use owner sign-in link (15 minutes), since demo mode logs
e-mail without sending it.
"""

import argparse
import asyncio
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from urllib.parse import urlsplit
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.composition.production import ProductionConfigurationError, load_production_settings
from gvas.config import DemoSettings, Settings
from gvas.domain.customers import hash_portal_token, new_portal_token
from gvas.domain.enums import (
    BillingInterval,
    CustomerQuoteStatus,
    DeliveryStatus,
    QuoteBilling,
    QuoteStatus,
    RecipientAddressKind,
)
from gvas.domain.identifiers import BusinessId, QuoteId
from gvas.domain.intake import INTAKE_CHANNEL_WEB, BookingKind, IntakeState, new_reference
from gvas.domain.invoices import due_on as invoice_due_on
from gvas.domain.invoices import invoice_number
from gvas.domain.messages import CustomerRecipient
from gvas.domain.owner import OWNER_LOGIN_TOKEN_TTL
from gvas.domain.payments import (
    MANUAL_PROVIDER,
    PaymentKind,
    PaymentMethod,
    PaymentSource,
    paid_through,
)
from gvas.domain.quotes import QuoteDraftProposal, QuoteLineItem, hash_claim_token, new_claim_token
from gvas.domain.time_zones import business_zone
from gvas.infrastructure.db import create_engine, create_session_factory
from gvas.infrastructure.demo import CLOSED_WEEKDAYS
from gvas.infrastructure.intake_models import IntakeConversation, IntakeMessage
from gvas.infrastructure.models import (
    Base,
    Business,
    Conversation,
    Customer,
    OwnerChannelEndpoint,
    OwnerLoginTokenRecord,
    QuoteRecord,
)
from gvas.infrastructure.payment_models import InvoiceRow, LedgerPaymentRow, QuoteSubscription

#: Any of these rows means the business already has data a plain seed would mix into.
SEEDED_MODELS = (Customer, QuoteRecord, IntakeConversation, LedgerPaymentRow, InvoiceRow)
LOCAL_HOSTS = ("localhost", "127.0.0.1")

CURRENCY = "USD"
ENDPOINT_NAMESPACE = "demo-seed"
# Kept on reset: the business itself and what bootstrap published for it.
KEPT_TABLES = frozenset(
    {
        "businesses",
        "business_template_profiles",
        "field_note_checklists",
        "field_note_report_templates",
        "field_note_template_sets",
    }
)


class SeedError(RuntimeError):
    """The seed cannot run against this database or business."""


@dataclass(frozen=True)
class FakeCustomer:
    name: str
    email: str
    phone: str
    address: str


CUSTOMERS = (
    FakeCustomer(
        "Hannah Brooks", "hannah.brooks@example.com", "+15105550118", "412 Mariposa Ave, Oakland"
    ),
    FakeCustomer("Daniel Ortiz", "d.ortiz@example.com", "+15105550127", "1820 Grove St, Berkeley"),
    FakeCustomer(
        "Priya Natarajan", "priya.n@example.com", "+15105550133", "77 Sea View Pkwy, Alameda"
    ),
    FakeCustomer(
        "Tom Whitaker", "tom.whitaker@example.com", "+15105550146", "9 Crocker Ave, Piedmont"
    ),
    FakeCustomer(
        "Grace Kim", "grace.kim@example.com", "+15105550152", "2301 Tulare Ave, El Cerrito"
    ),
    FakeCustomer("Luis Romero", "luis.romero@example.com", "+15105550161", "64 Solano Ave, Albany"),
    FakeCustomer("Ellen Park", "ellen.park@example.com", "+15105550174", "15 Estates Dr, Oakland"),
    FakeCustomer(
        "Marcus Bell", "marcus.bell@example.com", "+15105550185", "1188 Euclid Ave, Berkeley"
    ),
    FakeCustomer(
        "Sofia Alvarez", "sofia.alvarez@example.com", "+15105550192", "3410 Fruitvale Ave, Oakland"
    ),
    FakeCustomer(
        "Ruth Holloway", "ruth.holloway@example.com", "+15105550109", "52 Lakeshore Ave, Oakland"
    ),
    FakeCustomer(
        "Ben Carter", "ben.carter@example.com", "+15105550138", "705 Bancroft Ave, San Leandro"
    ),
    FakeCustomer(
        "Ana Delgado",
        "ana.delgado@example.com",
        "+15105550157",
        "2919 Lake Chabot Rd, Castro Valley",
    ),
)


@dataclass(frozen=True)
class FakeQuote:
    customer: int
    items: tuple[tuple[str, int, int], ...]  # description, quantity, unit price in cents
    # "pending" (needs the owner), "sent", "viewed", "accepted", "card", "check", "plan"
    stage: str
    days_ago: int
    note: str | None = None
    #: For paid quotes: days before now it was paid, clamped into this month when ``this_month``.
    paid_days_ago: int = 0
    this_month: bool = True
    #: Where invoices are on: days before now (or before it was paid) its invoice went out.
    invoice_days_ago: int | None = None


QUOTES = (
    FakeQuote(
        0,
        (("Backyard cleanup and haul-away", 1, 68_000), ("Mulch, 6 cu yd, spread", 1, 56_000)),
        "pending",
        0,
        "Side gate code on file.",
    ),
    FakeQuote(1, (("Drip irrigation repair, 3 zones", 1, 38_500),), "viewed", 2),
    FakeQuote(
        2,
        (
            ("Front-yard drought-tolerant replant", 1, 312_000),
            ("Smart irrigation controller", 1, 73_000),
        ),
        "sent",
        5,
    ),
    FakeQuote(3, (("Hedge trimming", 12, 3_500),), "sent", 1),
    FakeQuote(
        4,
        (("Paver patio, 12 x 14 ft", 1, 620_000),),
        "accepted",
        4,
        "Includes base prep and edging.",
        invoice_days_ago=3,
    ),
    FakeQuote(5, (("Lawn aeration and overseed", 1, 31_000),), "accepted", 12, invoice_days_ago=9),
    FakeQuote(6, (("Raised cedar beds, 4 x 8 ft", 3, 48_500),), "accepted", 2),
    FakeQuote(7, (("Spring cleanup", 1, 64_000),), "card", 9, paid_days_ago=5, invoice_days_ago=3),
    FakeQuote(
        8,
        (("Oak pruning", 2, 49_000),),
        "card",
        30,
        paid_days_ago=26,
        this_month=False,
        invoice_days_ago=5,
    ),
    FakeQuote(
        9, (("Gutter and bed cleanup", 1, 27_500),), "check", 8, paid_days_ago=3, invoice_days_ago=4
    ),
    FakeQuote(
        10,
        (("Sod install, 800 sq ft", 1, 235_000),),
        "card",
        40,
        paid_days_ago=33,
        this_month=False,
        invoice_days_ago=6,
    ),
    FakeQuote(
        11,
        (("Garden maintenance, 2 visits a month", 1, 18_000),),
        "plan",
        20,
        "3 months by check: $540.",
        paid_days_ago=2,
    ),
    FakeQuote(
        1, (("Sprinkler valve replacement", 2, 14_500),), "accepted", 26, invoice_days_ago=20
    ),
    FakeQuote(3, (("Fall leaf cleanup", 1, 42_000),), "accepted", 6, invoice_days_ago=1),
)
# The demo's next invoice is LL-1042.
SEED_FIRST_INVOICE = 1034
PAID_STAGES = frozenset({"card", "check", "plan"})
PLAN_MONTHS = 3
PLAN_AMOUNT = 54_000


@dataclass(frozen=True)
class FakeBooking:
    customer: FakeCustomer
    details: str
    day_offset: int
    hour: int
    hours: int
    waiting: bool = False
    notes: str | None = None


BOOKINGS = (
    FakeBooking(
        CUSTOMERS[4],
        "Paver patio: mark out and base prep",
        0,
        9,
        3,
        notes="About 12 x 14 ft, slight slope to the fence. "
        "Before the visit: side gate is unlocked.",
    ),
    FakeBooking(CUSTOMERS[7], "Monthly visit: mow, edge, blow", 0, 14, 2),
    FakeBooking(
        CUSTOMERS[5],
        "Lawn aeration and overseed",
        1,
        10,
        2,
        notes="Front lawn about 800 sq ft, bare patches by the driveway. "
        "Before the visit: friendly dog in the back yard.",
    ),
    FakeBooking(CUSTOMERS[11], "Garden maintenance visit", 3, 8, 2),
    FakeBooking(
        CUSTOMERS[6],
        "Raised beds: build and fill",
        4,
        13,
        3,
        notes="Two 4 x 8 ft beds for vegetables, full sun. Before the visit: gate code 4412.",
    ),
    FakeBooking(
        FakeCustomer(
            "Olivia Grant",
            "olivia.grant@example.com",
            "+15105550199",
            "48 Monte Vista Ave, Oakland",
        ),
        "Estimate: new lawn and a flagstone path in the front yard",
        2,
        15,
        1,
        waiting=True,
        notes="Front lawn about 600 sq ft, mostly weeds; flagstone path about 25 ft. "
        "Before the visit: will be home, two cats indoors.",
    ),
)


@dataclass(frozen=True)
class SeedSummary:
    customers: int
    quotes: int
    payments: int
    bookings: int
    invoices: int = 0


def _local_noon(day: date, zone: tzinfo) -> datetime:
    return datetime.combine(day, time(12), tzinfo=zone).astimezone(UTC)


def _paid_day(today: date, quote: FakeQuote) -> date:
    day = today - timedelta(days=quote.paid_days_ago)
    month_start = today.replace(day=1)
    if quote.this_month:
        return max(day, month_start)
    return min(day, month_start - timedelta(days=1))


def _open_day(today: date, offset: int) -> date:
    """The ``offset``-th open day from today (today itself is 0 when open), so
    distinct offsets never land on the same day."""

    day = today
    while day.weekday() in CLOSED_WEEKDAYS:
        day += timedelta(days=1)
    for _ in range(offset):
        day += timedelta(days=1)
        while day.weekday() in CLOSED_WEEKDAYS:
            day += timedelta(days=1)
    return day


def _sign_in_base(raw: str) -> str:
    parts = urlsplit(raw)
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOCAL_HOSTS):
        return raw.rstrip("/")
    raise SeedError("--sign-in-link must be an https:// dashboard address")


async def _reset(session: AsyncSession, business_id: BusinessId) -> None:
    for table in reversed(Base.metadata.sorted_tables):
        if table.name in KEPT_TABLES or "business_id" not in table.c:
            continue
        await session.execute(delete(table).where(table.c.business_id == business_id))


async def seed_demo(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    *,
    reset: bool,
    now: datetime | None = None,
) -> SeedSummary:
    now = now or datetime.now(UTC)
    async with session_factory() as session:
        business = await session.get(Business, business_id)
        if business is None:
            raise SeedError(f"business {business_id} does not exist; run gvas-bootstrap first")
        zone = business_zone(business.timezone) or ZoneInfo(DemoSettings().timezone)
        if reset:
            await _reset(session, business_id)
        else:
            for model in SEEDED_MODELS:
                if await session.scalar(
                    select(func.count()).select_from(model).where(model.business_id == business_id)
                ):
                    raise SeedError("the business already has data; pass --reset to replace it")
        today = now.astimezone(zone).date()
        owner = business.owner_email

        endpoint = OwnerChannelEndpoint(
            business_id=business_id,
            source_namespace=ENDPOINT_NAMESPACE,
            external_endpoint_id="owner",
            routing={},
        )
        session.add(endpoint)
        customers = []
        for index, person in enumerate(CUSTOMERS):
            customer = Customer(
                business_id=business_id,
                email=person.email,
                display_name=person.name,
                phone=person.phone,
                created_at=now - timedelta(days=40 - index * 2),
            )
            session.add(customer)
            customers.append(customer)
        await session.flush()

        payments = 0
        invoiced: list[tuple[datetime, QuoteId, FakeCustomer, QuoteDraftProposal]] = []
        for index, fake in enumerate(QUOTES):
            person = CUSTOMERS[fake.customer]
            quote_id = QuoteId(uuid4())
            created = now - timedelta(days=fake.days_ago, hours=3 + index % 4)
            invoiced_at = None
            if fake.stage in PAID_STAGES:
                # A quote goes out a couple of days before it's paid, whatever the date.
                paid = _local_noon(_paid_day(today, fake), zone)
                created = min(created, paid - timedelta(days=2))
                if fake.invoice_days_ago is not None:
                    invoiced_at = paid - timedelta(days=fake.invoice_days_ago)
            elif fake.invoice_days_ago is not None:
                invoiced_at = min(
                    _local_noon(today - timedelta(days=fake.invoice_days_ago), zone), now
                )
            if invoiced_at is not None:
                created = min(created, invoiced_at - timedelta(days=2))
            recurring = fake.stage == "plan"
            draft = QuoteDraftProposal(
                quote_id=quote_id,
                business_id=business_id,
                recipient=CustomerRecipient(
                    address=person.email,
                    address_kind=RecipientAddressKind.EMAIL,
                    display_name=person.name,
                    phone=person.phone,
                    service_address=person.address,
                ),
                currency=CURRENCY,
                line_items=tuple(
                    QuoteLineItem(description=text, quantity=quantity, unit_price_minor=price)
                    for text, quantity, price in fake.items
                ),
                owner_note=fake.note,
                billing=QuoteBilling.RECURRING if recurring else QuoteBilling.ONE_TIME,
                interval=BillingInterval.MONTH if recurring else None,
            )
            conversation = Conversation(
                business_id=business_id,
                endpoint_id=endpoint.id,
                external_conversation_id=f"demo-quote-{index + 1}",
                routing={},
            )
            session.add(conversation)
            await session.flush()
            pending = fake.stage == "pending"
            sent_at = created + timedelta(minutes=40)
            token = None if pending else new_claim_token()
            customer_status = {
                "viewed": CustomerQuoteStatus.VIEWED,
                "accepted": CustomerQuoteStatus.ACCEPTED,
                "card": CustomerQuoteStatus.PAID,
                "check": CustomerQuoteStatus.PAID,
                "plan": CustomerQuoteStatus.PAID,
            }.get(fake.stage)
            session.add(
                QuoteRecord(
                    id=quote_id,
                    business_id=business_id,
                    conversation_id=conversation.id,
                    active_conversation_id=conversation.id if pending else None,
                    external_conversation_id=conversation.external_conversation_id,
                    status=(
                        QuoteStatus.AWAITING_APPROVAL if pending else QuoteStatus.DELIVERED
                    ).value,
                    revision=1,
                    source_message_key=f"demo-{index + 1}",
                    last_message_key=f"demo-{index + 1}",
                    pending_request_text=fake.items[0][0],
                    draft=draft.model_dump(mode="json"),
                    approval_correlation_id=f"demo-approval-{index + 1}",
                    delivery_receipt=None
                    if pending
                    else {
                        "status": DeliveryStatus.DELIVERED.value,
                        "provider_message_id": f"demo-{index + 1}",
                        "occurred_at": sent_at.isoformat(),
                        "emailed": True,
                    },
                    claim_token=token,
                    claim_token_hash=None if token is None else hash_claim_token(token),
                    customer_status=None if customer_status is None else customer_status.value,
                    approved_at=None if pending else sent_at,
                    customer_id=None if pending else customers[fake.customer].id,
                    billing=draft.billing.value,
                    billing_interval=None if draft.interval is None else draft.interval.value,
                    version=1,
                    created_at=created,
                    updated_at=sent_at if not pending else created,
                )
            )
            await session.flush()
            if invoiced_at is not None and business.invoice_prefix:
                invoiced.append((invoiced_at, quote_id, person, draft))
            if fake.stage not in PAID_STAGES:
                continue
            paid_on = _paid_day(today, fake)
            paid_at = min(_local_noon(paid_on, zone), now)
            manual = fake.stage != "card"
            session.add(
                LedgerPaymentRow(
                    business_id=business_id,
                    quote_id=quote_id,
                    kind=(PaymentKind.PLAN if recurring else PaymentKind.ONE_OFF).value,
                    source=(PaymentSource.MANUAL if manual else PaymentSource.STRIPE).value,
                    method=(PaymentMethod.CHECK if manual else PaymentMethod.CARD).value,
                    reference=f"{'manual' if manual else 'demo_cs'}:{secrets.token_hex(8)}",
                    amount_cents=PLAN_AMOUNT if recurring else draft.total_minor,
                    currency=CURRENCY,
                    paid_at=paid_at,
                    months_covered=PLAN_MONTHS if recurring else None,
                    recorded_by=owner if manual else None,
                    recorded_at=paid_at,
                    note="Check #2207" if recurring else ("Check #1184" if manual else None),
                    customer_status_before="accepted" if manual else None,
                )
            )
            payments += 1
            if recurring:
                session.add(
                    QuoteSubscription(
                        business_id=business_id,
                        quote_id=quote_id,
                        customer_id=customers[fake.customer].id,
                        provider=MANUAL_PROVIDER,
                        stripe_subscription_id=f"manual:{quote_id}",
                        status="active",
                        interval=BillingInterval.MONTH.value,
                        amount_cents=draft.total_minor,
                        currency=CURRENCY,
                        created_at=paid_at,
                        updated_at=paid_at,
                        paid_from=paid_on,
                        paid_through=paid_through(paid_on, PLAN_MONTHS),
                    )
                )

        for sequence, (sent_at, quote_id, person, draft) in enumerate(
            sorted(invoiced, key=lambda entry: entry[0]), start=SEED_FIRST_INVOICE
        ):
            session.add(
                InvoiceRow(
                    business_id=business_id,
                    quote_id=quote_id,
                    sequence=sequence,
                    number=invoice_number(business.invoice_prefix or "", sequence),
                    customer_name=person.name,
                    customer_email=person.email,
                    customer_phone=person.phone,
                    service_address=person.address,
                    line_items=[
                        {
                            "description": item.description,
                            "quantity": item.quantity,
                            "unit_price_minor": item.unit_price_minor,
                        }
                        for item in draft.line_items
                    ],
                    total_cents=draft.total_minor,
                    currency=CURRENCY,
                    sent_at=sent_at,
                    due_on=invoice_due_on(sent_at, zone),
                    sent_by=owner,
                    created_at=sent_at,
                )
            )
        await session.flush()

        for booking in BOOKINGS:
            start = datetime.combine(
                _open_day(today, booking.day_offset),
                time(booking.hour),
                tzinfo=zone,
            ).astimezone(UTC)
            asked = now - timedelta(hours=2 if booking.waiting else 30)
            person = booking.customer
            intake = IntakeConversation(
                business_id=business_id,
                reference=new_reference(),
                token_hash=hash_portal_token(new_portal_token()),
                channel=INTAKE_CHANNEL_WEB,
                state=(
                    IntakeState.AWAITING_OWNER if booking.waiting else IntakeState.APPROVED
                ).value,
                collected={
                    "name": person.name,
                    "email": person.email,
                    "phone": person.phone,
                    "details": booking.details,
                    "address": person.address,
                    "notes": booking.notes,
                },
                requested_slot_start=start,
                requested_slot_end=start + timedelta(hours=booking.hours),
                booking_kind=None if booking.waiting else BookingKind.BOOKED.value,
                decision_at=None if booking.waiting else asked + timedelta(hours=1),
                owner_notified_at=asked,
                expires_at=now + timedelta(days=30),
                created_at=asked,
                updated_at=asked if booking.waiting else asked + timedelta(hours=1),
            )
            session.add(intake)
            await session.flush()
            session.add(
                IntakeMessage(
                    business_id=business_id,
                    conversation_id=intake.id,
                    role="user",
                    content=booking.details,
                    created_at=asked,
                )
            )
        await session.commit()
    return SeedSummary(
        customers=len(CUSTOMERS),
        quotes=len(QUOTES),
        payments=payments,
        bookings=len(BOOKINGS),
        invoices=len(invoiced),
    )


async def mint_owner_sign_in(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    *,
    now: datetime | None = None,
) -> str:
    """A raw single-use owner login token for the business's owner e-mail."""

    now = now or datetime.now(UTC)
    async with session_factory() as session:
        business = await session.get(Business, business_id)
        if business is None or not business.owner_email:
            raise SeedError("the business needs an owner e-mail (gvas-configure-business)")
        token = new_portal_token()
        session.add(
            OwnerLoginTokenRecord(
                token_hash=hash_portal_token(token),
                business_id=business_id,
                email=business.owner_email,
                expires_at=now + OWNER_LOGIN_TOKEN_TTL,
                created_at=now,
            )
        )
        await session.commit()
    return token


def _business_id(raw: str) -> BusinessId:
    try:
        return BusinessId(UUID(raw))
    except ValueError as error:
        raise SeedError("--business-id must be a UUID") from error


async def _run(arguments: argparse.Namespace) -> None:
    business_id = _business_id(arguments.business_id)
    engine = create_engine(Settings().database_url)
    try:
        session_factory = create_session_factory(engine)
        if arguments.seed or arguments.reset:
            summary = await seed_demo(session_factory, business_id, reset=arguments.reset)
            print(  # noqa: T201
                f"seeded business {business_id}: {summary.customers} customers, "
                f"{summary.quotes} quotes, {summary.payments} payments, "
                f"{summary.bookings} bookings, {summary.invoices} invoices"
            )
        if arguments.sign_in_link:
            token = await mint_owner_sign_in(session_factory, business_id)
            base = _sign_in_base(arguments.sign_in_link)
            print(f"owner sign-in (single use, 15 minutes): {base}/portal/login?token={token}")  # noqa: T201
    finally:
        await engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed a demo business with fictional data")
    parser.add_argument("--business-id", required=True)
    parser.add_argument("--reset", action="store_true", help="replace the business's data")
    parser.add_argument(
        "--no-seed", dest="seed", action="store_false", help="only print a sign-in link"
    )
    parser.add_argument("--sign-in-link", metavar="DASHBOARD_URL")
    arguments = parser.parse_args(argv)
    if not DemoSettings().mode:
        raise SeedError("refusing to seed: GVAS_DEMO_MODE is off, so this may be a real database")
    if arguments.sign_in_link:
        _sign_in_base(arguments.sign_in_link)
    # The same check a demo server starts with: no credential that could reach a
    # real person or account, so this is not a production environment.
    try:
        load_production_settings()
    except ProductionConfigurationError as error:
        raise SeedError(f"refusing to seed: this is not an isolated demo ({error})") from error
    asyncio.run(_run(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
