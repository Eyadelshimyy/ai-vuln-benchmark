"""Demo mode: a fictional business runs the real workflows, nothing is sent."""

import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import CustomerDeliveryFake, OwnerReplyFake, TranscriptionFake
from gvas.composition import build_application
from gvas.composition.production import (
    ProductionConfigurationError,
    build_production_runtime,
    load_production_settings,
)
from gvas.config import DemoSettings, IntakeSettings
from gvas.domain.customers import PortalLoginEmailRequest
from gvas.domain.enums import DeliveryStatus, RecipientAddressKind
from gvas.domain.identifiers import BusinessId, IntakeConversationId
from gvas.domain.intake import (
    BookingKind,
    BookingRequest,
    IntakeCollected,
    IntakeConversation,
    IntakeState,
    IntakeTurn,
    SupersededBooking,
)
from gvas.domain.messages import (
    ConversationRef,
    CustomerDeliveryRequest,
    CustomerRecipient,
    CustomerTextRequest,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.owner import CalendarEventSource
from gvas.infrastructure.demo import (
    DEMO_EVENT_TYPE_URI,
    DemoAvailability,
    DemoBookedEvents,
    DemoModeError,
    LoggedCustomerEmail,
    LoggedCustomerText,
    LoggedOwnerReply,
    LoggedPortalLoginEmail,
)
from gvas.infrastructure.hosted_links import PORTAL_LOGIN_LINK_REFERENCE
from gvas.infrastructure.intake_repositories import SqlIntakeConversationRepository
from gvas.infrastructure.models import Business
from test_composition import Clock, inbound, seed_business
from test_intake_booking import (
    EMAIL,
    PUBLIC_KEY,
    IntakeAgentFake,
    collected_turn,
    conversation_row,
    http_client,
    intake_business,
    seed_owner_thread,
)
from test_pilot_runtime import deterministic_ports, immediate_worker, texts_of

MANAGED_DATABASE_URL = "postgresql+asyncpg://user:pw@demo-db.railway.internal:5432/railway"
DEMO_ENVIRONMENT = {
    "GVAS_DEMO_MODE": "1",
    "GVAS_DATABASE_URL": MANAGED_DATABASE_URL,
    "GVAS_OPENAI_API_KEY": "sk-not-a-real-key",
}
PACIFIC = ZoneInfo("America/Los_Angeles")


@pytest.fixture
def demo_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in DEMO_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


# -- startup -----------------------------------------------------------------


@pytest.mark.usefixtures("demo_environment")
def test_a_demo_starts_with_only_its_database_and_the_model_key() -> None:
    settings = load_production_settings()

    assert settings.demo.mode is True
    runtime = build_production_runtime(settings)
    ports_of = runtime.application
    paths = {getattr(route, "path", "") for route in runtime.app.routes}

    assert "/healthz" in paths
    assert not any(path.startswith("/slack") for path in paths)
    assert not any(path.startswith("/telnyx") for path in paths)
    assert not any(path.startswith("/calendly") for path in paths)
    assert ports_of.intake is not None


@pytest.mark.usefixtures("demo_environment")
def test_a_demo_still_needs_its_own_database_and_the_model_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GVAS_DATABASE_URL")
    monkeypatch.delenv("GVAS_OPENAI_API_KEY")

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "GVAS_DATABASE_URL or DATABASE_URL" in str(error.value)
    assert "GVAS_OPENAI_API_KEY" in str(error.value)


@pytest.mark.usefixtures("demo_environment")
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GVAS_SLACK_BOT_TOKEN", "xoxb-real-looking-token"),
        ("GVAS_SLACK_SIGNING_SECRET", "slack-signing-secret"),
        ("GVAS_SLACK_INSTALLATIONS", f"T0000000000={uuid4()}:U0000000000"),
        ("GVAS_RESEND_API_KEY", "re_real_looking_key"),
        ("GVAS_TELNYX_API_KEY", "KEYreal-looking"),
        ("GVAS_TELNYX_PUBLIC_KEY", "telnyx-public-key"),
        ("GVAS_TELNYX_MESSAGING_PROFILE_ID", "profile-id"),
        ("GVAS_CALENDLY_TOKEN", "calendly-real-looking-token"),
        ("GVAS_CALENDLY_WEBHOOK_SIGNING_KEY", "calendly-signing-key"),
        ("GVAS_PORTAL_API_TOKEN", "portal-token"),
        ("GVAS_R2_SECRET_ACCESS_KEY", "r2-secret"),
    ],
)
def test_a_demo_refuses_any_credential_that_could_reach_someone(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert name in str(error.value)
    assert value not in str(error.value)


@pytest.mark.usefixtures("demo_environment")
@pytest.mark.parametrize("key", ["sk_live_abc123", "rk_live_abc123", "not-a-stripe-key"])
def test_a_demo_refuses_a_stripe_key_that_is_not_test_mode(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setenv("GVAS_STRIPE_SECRET_KEY", key)
    monkeypatch.setenv("GVAS_STRIPE_WEBHOOK_SECRET", "whsec_demo")

    with pytest.raises(ProductionConfigurationError) as error:
        load_production_settings()

    assert "test-mode" in str(error.value)
    assert key not in str(error.value)


@pytest.mark.usefixtures("demo_environment")
def test_a_demo_accepts_a_stripe_test_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GVAS_STRIPE_SECRET_KEY", "sk_test_abc123")
    monkeypatch.setenv("GVAS_STRIPE_WEBHOOK_SECRET", "whsec_demo")
    monkeypatch.setenv("GVAS_STRIPE_DEPLOYMENT", "demo")

    settings = load_production_settings()

    assert settings.stripe.is_configured
    assert settings.stripe.deployment == "demo"
    assert build_production_runtime(settings).application.public_quotes is not None


@pytest.mark.usefixtures("demo_environment")
@pytest.mark.parametrize("deployment", [None, "production"])
def test_a_demo_with_stripe_must_not_claim_productions_events(
    monkeypatch: pytest.MonkeyPatch, deployment: str | None
) -> None:
    """The demo shares production's Stripe test account; without its own
    deployment name it would act on production's events."""

    monkeypatch.setenv("GVAS_STRIPE_SECRET_KEY", "sk_test_abc123")
    monkeypatch.setenv("GVAS_STRIPE_WEBHOOK_SECRET", "whsec_demo")
    if deployment is not None:
        monkeypatch.setenv("GVAS_STRIPE_DEPLOYMENT", deployment)

    with pytest.raises(ProductionConfigurationError, match="GVAS_STRIPE_DEPLOYMENT"):
        load_production_settings()


@pytest.mark.usefixtures("demo_environment")
def test_a_demo_with_half_of_stripe_still_does_not_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GVAS_STRIPE_SECRET_KEY", "sk_test_abc123")

    with pytest.raises(ProductionConfigurationError, match="GVAS_STRIPE_WEBHOOK_SECRET"):
        load_production_settings()


def test_demo_settings_reject_an_unknown_zone_and_a_day_too_short() -> None:
    with pytest.raises(ValueError, match="time zone"):
        DemoSettings(timezone="Mars/Olympus")
    with pytest.raises(ValueError, match="one slot"):
        DemoSettings(day_start_hour=9, day_end_hour=9)


# -- nothing is sent ---------------------------------------------------------


async def test_customer_mail_texts_and_owner_messages_are_logged_not_sent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    business_id = BusinessId(uuid4())
    caplog.set_level(logging.INFO, logger="gvas.infrastructure.demo")

    email = await LoggedCustomerEmail("https://demo.example/portal/login").deliver(
        CustomerDeliveryRequest(
            business_id=business_id,
            recipient=CustomerRecipient(
                address="dana@example.test", address_kind=RecipientAddressKind.EMAIL
            ),
            idempotency_key="quote:1",
            subject="Your quote",
            body_text="Spring cleanup, $480.",
            quote_url="https://larkspur.example/q/abc",
            links=(PORTAL_LOGIN_LINK_REFERENCE,),
        )
    )
    text = await LoggedCustomerText().send_text(
        CustomerTextRequest(
            business_id=business_id,
            phone_number="+15105550142",
            text="Your estimate is booked.",
            idempotency_key="text:1",
        )
    )
    conversation = ConversationRef(business_id=business_id, external_conversation_id="C1:1.0")
    owner = await LoggedOwnerReply().send(
        conversation,
        OutboundOwnerMessage(
            business_id=business_id,
            conversation_ref=conversation,
            parts=(
                TextPart(
                    text="Booking request #abc123: approve at "
                    "https://demo.example/intake/decide?token=d3cide"
                ),
            ),
            correlation_id="notice:1",
        ),
    )

    assert {email.status, text.status, owner.status} == {DeliveryStatus.DELIVERED}
    assert email.emailed is True
    assert email.customer_link == "https://larkspur.example/q/abc"
    assert email.provider_message_id is not None
    assert email.provider_message_id.startswith("demo-")
    logged = caplog.text
    assert "not sent: customer e-mail" in logged
    assert "not sent: text" in logged
    assert "dana@example.test" not in logged
    assert "+15105550142" not in logged
    assert "not sent: owner message" in logged
    # Neither what was written nor its links (bearer tokens) reach the log.
    assert "Booking request #abc123" not in logged
    assert "/q/abc" not in logged
    assert "d3cide" not in logged


async def test_an_unknown_hosted_link_reference_is_refused() -> None:
    with pytest.raises(DemoModeError):
        await LoggedCustomerEmail("https://demo.example/portal/login").deliver(
            CustomerDeliveryRequest(
                business_id=BusinessId(uuid4()),
                recipient=CustomerRecipient(
                    address="dana@example.test", address_kind=RecipientAddressKind.EMAIL
                ),
                idempotency_key="quote:2",
                body_text="Spring cleanup, $480.",
                links=("not-a-reference",),
            )
        )


async def test_a_sign_in_request_is_logged_but_never_its_link(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="gvas.infrastructure.demo")

    receipt = await LoggedPortalLoginEmail().send_login_link(
        PortalLoginEmailRequest(
            business_id=BusinessId(uuid4()),
            to="owner@larkspur.example",
            business_display_name="Larkspur Lawn & Garden",
            login_url="https://demo.example/portal/session?token=t0ken",
            idempotency_key="login:1",
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )
    )

    assert receipt.status is DeliveryStatus.DELIVERED
    assert "not sent: sign-in e-mail" in caplog.text
    assert "owner@larkspur.example" not in caplog.text
    assert "t0ken" not in caplog.text


# -- the calendar ------------------------------------------------------------


async def demo_business(
    session_factory: async_sessionmaker[AsyncSession], zone: str | None = "America/Los_Angeles"
) -> BusinessId:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    async with session_factory() as session:
        await session.execute(
            update(Business).where(Business.id == business_id).values(timezone=zone)
        )
        await session.commit()
    return business_id


async def hold(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    start: datetime,
    *,
    state: IntakeState,
    booking_kind: str | None = None,
    details: str | None = None,
    superseded: datetime | None = None,
) -> str:
    now = datetime.now(UTC)
    reference = uuid4().hex[:8]
    async with session_factory() as session:
        await SqlIntakeConversationRepository(session).add(
            IntakeConversation(
                conversation_id=IntakeConversationId(uuid4()),
                business_id=business_id,
                reference=reference,
                token_hash="0" * 64,
                state=state,
                collected=IntakeCollected(
                    name="Dana Reyes",
                    email="dana@example.test",
                    address="12 Laurel Way",
                    details=details,
                ),
                # Stored as UTC, as Postgres returns it; SQLite keeps no zone.
                requested_slot_start=start.astimezone(UTC),
                requested_slot_end=start.astimezone(UTC) + timedelta(hours=1),
                booking_kind=booking_kind,
                superseded_booking=None
                if superseded is None
                else SupersededBooking(
                    slot_start=superseded.astimezone(UTC),
                    slot_end=superseded.astimezone(UTC) + timedelta(hours=1),
                ),
                expires_at=now + timedelta(days=30),
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    return reference


def next_monday(zone: ZoneInfo) -> datetime:
    today = datetime.now(zone).replace(hour=0, minute=0, second=0, microsecond=0)
    return today + timedelta(days=7 - today.weekday())


async def test_openings_fall_in_working_hours_in_the_business_zone(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await demo_business(session_factory)
    availability = DemoAvailability(DemoSettings(), session_factory)
    monday = next_monday(PACIFIC)

    week = await availability.available_slots(business_id, monday, monday + timedelta(days=7))

    assert week
    for opening in week:
        local = opening.start.astimezone(PACIFIC)
        assert local.weekday() != 6
        assert 8 <= local.hour < 17
        assert opening.end - opening.start == timedelta(hours=1)
    assert len({opening.start.astimezone(PACIFIC).date() for opening in week}) == 6
    # Some hours are shown as taken, and the same ones every time.
    assert len(week) < 6 * 9
    assert week == await availability.available_slots(
        business_id, monday, monday + timedelta(days=7)
    )


async def test_openings_use_the_fallback_zone_when_the_business_has_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await demo_business(session_factory, zone=None)
    eastern = ZoneInfo("America/New_York")
    availability = DemoAvailability(DemoSettings(timezone="America/New_York"), session_factory)
    monday = next_monday(eastern)

    week = await availability.available_slots(business_id, monday, monday + timedelta(days=7))

    assert week
    assert all(8 <= opening.start.astimezone(eastern).hour < 17 for opening in week)


async def test_a_held_slot_is_never_offered_again(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await demo_business(session_factory)
    availability = DemoAvailability(DemoSettings(), session_factory)
    monday = next_monday(PACIFIC)
    window = (monday, monday + timedelta(days=7))
    offered = await availability.available_slots(business_id, *window)
    waiting, booked, declined = offered[0], offered[1], offered[2]

    await hold(session_factory, business_id, waiting.start, state=IntakeState.AWAITING_OWNER)
    await hold(
        session_factory,
        business_id,
        booked.start,
        state=IntakeState.APPROVED,
        booking_kind=BookingKind.BOOKED.value,
    )
    await hold(session_factory, business_id, declined.start, state=IntakeState.DECLINED)

    starts = {opening.start for opening in await availability.available_slots(business_id, *window)}
    assert waiting.start not in starts
    assert booked.start not in starts
    assert declined.start in starts


async def test_booking_touches_no_calendar_and_shows_on_the_dashboard(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await demo_business(session_factory)
    availability = DemoAvailability(DemoSettings(), session_factory)
    monday = next_monday(PACIFIC)
    start = monday.replace(hour=10)

    result = await availability.book(
        BookingRequest(
            business_id=business_id,
            slot_start=start,
            slot_end=start + timedelta(hours=1),
            invitee_name="Dana Reyes",
            invitee_email="dana@example.test",
        )
    )
    assert result.kind is BookingKind.BOOKED
    assert result.event_type_uri == DEMO_EVENT_TYPE_URI

    reference = await hold(
        session_factory,
        business_id,
        start,
        state=IntakeState.APPROVED,
        booking_kind=BookingKind.BOOKED.value,
        details="Spring cleanup estimate",
    )
    await hold(
        session_factory, business_id, start + timedelta(hours=2), state=IntakeState.AWAITING_OWNER
    )
    events = await DemoBookedEvents(session_factory).upcoming(
        business_id, monday, monday + timedelta(days=7)
    )

    assert len(events) == 1
    event = events[0]
    assert event.source is CalendarEventSource.BOOKING
    assert event.title == "Spring cleanup estimate"
    assert event.start == start
    assert event.invitee_name == "Dana Reyes"
    assert event.location == "12 Laurel Way"
    assert event.reference == reference


async def test_a_slot_already_booked_sends_a_link_to_pick_again(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await demo_business(session_factory)
    async with session_factory() as session:
        await session.execute(
            update(Business)
            .where(Business.id == business_id)
            .values(site_url="https://larkspur.example")
        )
        await session.commit()
    availability = DemoAvailability(DemoSettings(), session_factory)
    start = next_monday(PACIFIC).replace(hour=10)
    first = await hold(
        session_factory,
        business_id,
        start,
        state=IntakeState.APPROVED,
        booking_kind=BookingKind.BOOKED.value,
    )
    # A second request waiting for the same hour does not block the first.
    await hold(session_factory, business_id, start, state=IntakeState.AWAITING_OWNER)

    def request(reference: str) -> BookingRequest:
        return BookingRequest(
            business_id=business_id,
            slot_start=start,
            slot_end=start + timedelta(hours=1),
            invitee_name="Sam Ortiz",
            invitee_email="sam@example.test",
            reference=reference,
        )

    again = await availability.book(request(first))
    second = await availability.book(request("other123"))

    assert again.kind is BookingKind.BOOKED
    assert second.kind is BookingKind.LINK
    assert second.link == "https://larkspur.example"


async def test_a_booking_awaiting_its_reschedule_stays_booked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await demo_business(session_factory)
    availability = DemoAvailability(DemoSettings(), session_factory)
    monday = next_monday(PACIFIC)
    window = (monday, monday + timedelta(days=7))
    offered = await availability.available_slots(business_id, *window)
    original, wanted = offered[0], offered[3]

    await hold(
        session_factory,
        business_id,
        wanted.start,
        state=IntakeState.AWAITING_OWNER,
        booking_kind=BookingKind.BOOKED.value,
        superseded=original.start,
    )

    starts = {opening.start for opening in await availability.available_slots(business_id, *window)}
    events = await DemoBookedEvents(session_factory).upcoming(business_id, *window)
    assert original.start not in starts
    assert wanted.start not in starts
    assert [event.start for event in events] == [original.start.astimezone(UTC)]


async def test_a_booking_already_under_way_shows_on_the_calendar(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await demo_business(session_factory)
    start = next_monday(PACIFIC).replace(hour=10)
    await hold(
        session_factory,
        business_id,
        start,
        state=IntakeState.APPROVED,
        booking_kind=BookingKind.BOOKED.value,
    )

    events = await DemoBookedEvents(session_factory).upcoming(
        business_id, start + timedelta(minutes=30), start + timedelta(days=1)
    )

    assert [event.start for event in events] == [start.astimezone(UTC)]


async def test_another_business_never_sees_the_demo_bookings(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    larkspur = await demo_business(session_factory)
    other = await demo_business(session_factory)
    availability = DemoAvailability(DemoSettings(), session_factory)
    monday = next_monday(PACIFIC)
    window = (monday, monday + timedelta(days=7))
    before = await availability.available_slots(other, *window)
    for opening in before[:3]:
        await hold(
            session_factory,
            larkspur,
            opening.start,
            state=IntakeState.APPROVED,
            booking_kind=BookingKind.BOOKED.value,
        )

    assert await DemoBookedEvents(session_factory).upcoming(other, *window) == ()
    assert len(await DemoBookedEvents(session_factory).upcoming(larkspur, *window)) == 3
    # Larkspur's bookings hold only Larkspur's calendar.
    assert await availability.available_slots(other, *window) == before


# -- the real owner approval -------------------------------------------------


async def test_a_website_booking_waits_for_the_owner_then_books_in_the_demo(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # Openings are read on their own connection while the chat's unit of work
    # is open, which in-memory SQLite (one shared connection) cannot do.
    session_factory = postgres_session_factory
    business_id = await intake_business(session_factory)
    async with session_factory() as session:
        await session.execute(
            update(Business).where(Business.id == business_id).values(timezone=PACIFIC.key)
        )
        await session.commit()
    availability = DemoAvailability(DemoSettings(), session_factory)
    owner = OwnerReplyFake()
    agent = IntakeAgentFake(
        [
            collected_turn(name="Dana Reyes", email=EMAIL, phone="+15105550142"),
            collected_turn(address="12 Laurel Way", details="Spring cleanup estimate"),
            IntakeTurn(reply="Here are some openings.", ready_for_slots=True),
        ]
    )
    ports = replace(
        deterministic_ports(owner, TranscriptionFake({}), CustomerDeliveryFake()),
        intake_agent=agent,
        availability=availability,
        booked_events=DemoBookedEvents(session_factory),
        customer_email=CustomerDeliveryFake(),
    )
    application = build_application(
        ports,
        session_factory=session_factory,
        now=Clock(),
        intake_settings=IntakeSettings(max_conversations_per_day=0),
    )
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()

    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        assert created.status_code == 201
        body = created.json()
        headers = {"Authorization": f"Bearer {body['conversationToken']}"}
        messages = f"/v1/intake/conversations/{body['conversationId']}/messages"
        for text in ("Dana, dana's details", "12 Laurel Way", "what times do you have?"):
            reply = await client.post(messages, json={"message": text}, headers=headers)
            assert reply.status_code == 200
        offered = reply.json()["slots"]
        assert offered, "the demo calendar must offer openings"
        picked = await client.post(
            messages, json={"message": f"slot:{offered[0]['start']}"}, headers=headers
        )
        assert picked.json()["state"] == "awaiting_owner"

    for _ in range(4):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    assert row.state == "awaiting_owner"
    assert row.booking_kind is None, "nothing is booked before the owner says yes"

    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {row.reference}", message_key="approve-demo")
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    row = await conversation_row(session_factory, business_id)
    assert row.state == "approved"
    assert row.booking_kind == "booked"
    assert row.booking_event_type_uri == DEMO_EVENT_TYPE_URI
    assert texts_of(owner, "Approved booking")
    start = datetime.fromisoformat(offered[0]["start"])
    window = (start - timedelta(days=1), start + timedelta(days=1))
    events = await DemoBookedEvents(session_factory).upcoming(business_id, *window)
    assert [event.reference for event in events] == [row.reference]
    reoffered = await availability.available_slots(business_id, *window)
    assert start not in {opening.start for opening in reoffered}
