"""Website booking intake: chat lifecycle, tenant isolation, caps, owner
decisions and both booking paths — all behind fakes.

The hard rule under test: nothing reaches the availability provider's ``book``
until the owner replies ``approve booking <ref>``; declining never books.
"""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    CustomerDeliveryFake,
    OwnerReplyFake,
    TranscriptionFake,
)
from gvas.application.intake import (
    NO_AVAILABILITY_REPLY,
    OPENING_REPLY,
    SLOTS_OFFER_REPLY,
    UNAVAILABLE_REPLY,
    VISIT_NOTED_REPLY,
    IntakeClosedError,
    IntakeDeliveryError,
    IntakeTextStatus,
    IntakeVisitor,
    SendIntakeCustomerEmailService,
    SendIntakeCustomerTextService,
)
from gvas.composition import Application, build_application
from gvas.config import IntakeSettings
from gvas.domain.customers import CustomerRecord
from gvas.domain.enums import DeliveryStatus
from gvas.domain.identifiers import BusinessId, CustomerId, IntakeConversationId
from gvas.domain.intake import (
    INTAKE_BOOKING_ARRANGE_COMMAND_TYPE,
    INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE,
    INTAKE_CUSTOMER_TEXT_COMMAND_TYPE,
    INTAKE_NOTES_MAX_CHARS,
    OWNER_NOTICE_EMAIL_COMMAND_TYPE,
    PRICE_GUARD_REPLY,
    AvailabilityError,
    AvailableSlot,
    BookingKind,
    BookingRequest,
    BookingResult,
    IntakeAgentError,
    IntakeCollected,
    IntakeConversation,
    IntakeMessageRole,
    IntakeProfile,
    IntakeState,
    IntakeTurn,
    IntakeTurnRequest,
    booking_request_notice,
    pick_offer_slots,
    visit_note,
    with_visit_note,
)
from gvas.domain.messages import (
    CustomerDeliveryRequest,
    CustomerTextRequest,
    DeliveryReceipt,
)
from gvas.domain.ports import OwnerEmailPort
from gvas.infrastructure.customer_repositories import SqlCustomerRepository
from gvas.infrastructure.intake_models import IntakeConversation as IntakeRow
from gvas.infrastructure.intake_repositories import (
    SqlIntakeConversationRepository,
    SqlIntakeMessageRepository,
)
from gvas.infrastructure.models import OutboxMessage
from gvas.infrastructure.repositories import SqlBusinessRepository
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory
from gvas.interfaces.http.app import create_app
from gvas.interfaces.http.portal import create_portal_router
from gvas.interfaces.http.public import create_public_router
from test_composition import Clock, inbound, seed_business
from test_hosted_quotes import CustomerTextFake
from test_pilot_runtime import deterministic_ports, immediate_worker, texts_of
from test_portal_quote_handoff import consent_to_texts

NOW = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)  # a Monday
PUBLIC_KEY = "gvb_intake_test"
SITE_URL = "https://site.example.test"
CALENDLY_LINK = "https://calendly.com/test/inspection"
EMAIL = "customer@example.test"


def next_weekday(anchor: datetime, *, days: int = 2) -> datetime:
    """A 9 AM slot strictly after ``anchor`` on a weekday."""

    day = anchor.date() + timedelta(days=days)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime(day.year, day.month, day.day, 9, 0, tzinfo=UTC)


def slot(anchor: datetime, *, days: int = 2) -> AvailableSlot:
    start = next_weekday(anchor, days=days)
    return AvailableSlot(start=start, end=start + timedelta(hours=1))


class IntakeAgentFake:
    """Scripted model: pops turns, records every request it was asked."""

    def __init__(
        self, turns: list[IntakeTurn] | None = None, *, error: Exception | None = None
    ) -> None:
        self._turns = list(turns or [])
        self.error = error
        self.requests: list[IntakeTurnRequest] = []
        self.calls = 0

    async def turn(self, request: IntakeTurnRequest) -> IntakeTurn:
        self.requests.append(request)
        self.calls += 1
        if self.error is not None:
            raise self.error
        if not self._turns:
            return IntakeTurn(reply="Anything else I should tell the owner?")
        return self._turns.pop(0)


class AvailabilityFake:
    """Fixed openings; ``book`` returns the configured result or raises."""

    def __init__(
        self,
        slots: tuple[AvailableSlot, ...] = (),
        *,
        result: BookingResult | None = None,
        error: Exception | None = None,
        found: BookingResult | None = None,
        slot_error: Exception | None = None,
    ) -> None:
        self.slots = slots
        self._result = result or BookingResult(kind=BookingKind.BOOKED)
        self._error = error
        self._found = found
        self._slot_error = slot_error
        self.slot_calls: list[tuple[BusinessId, datetime, datetime]] = []
        self.book_calls: list[BookingRequest] = []
        self.find_calls: list[BookingRequest] = []
        self.cancel_calls: list[tuple[BusinessId, str]] = []

    async def available_slots(
        self, business_id: BusinessId, start: datetime, end: datetime
    ) -> tuple[AvailableSlot, ...]:
        self.slot_calls.append((business_id, start, end))
        if self._slot_error is not None:
            raise self._slot_error
        return self.slots

    async def find_booking(self, request: BookingRequest) -> BookingResult | None:
        self.find_calls.append(request)
        return self._found

    async def booking_event_type_uri(self, business_id: BusinessId) -> str | None:
        return self._result.event_type_uri or "cal://event-type"

    async def book(self, request: BookingRequest) -> BookingResult:
        self.book_calls.append(request)
        if self._error is not None:
            raise self._error
        return self._result

    async def cancel_booking(self, business_id: BusinessId, event_uri: str) -> None:
        self.cancel_calls.append((business_id, event_uri))


async def intake_business(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    public_key: str = PUBLIC_KEY,
    notification_email: str | None = None,
) -> BusinessId:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id,
            site_url=SITE_URL,
            display_name="Test Co",
            calendly_url=CALENDLY_LINK,
            public_key=public_key,
            notification_email=notification_email,
            now=NOW,
        )
        await session.commit()
    return business_id


def intake_app(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    agent: IntakeAgentFake | None = None,
    availability: AvailabilityFake | None = None,
    owner_replies: OwnerReplyFake | None = None,
    customer_email: CustomerDeliveryFake | None = None,
    customer_text: CustomerTextFake | None = None,
    intake_settings: IntakeSettings | None = None,
    owner_email: OwnerEmailPort | None = None,
    intake_message_budget: Callable[[UUID, datetime], Awaitable[str | None]] | None = None,
    intake_visitor: Callable[[UUID], Awaitable[IntakeVisitor | None]] | None = None,
) -> tuple[Application, OwnerReplyFake]:
    owner = owner_replies or OwnerReplyFake()
    ports = deterministic_ports(owner, TranscriptionFake({}), CustomerDeliveryFake())
    ports = replace(
        ports,
        intake_agent=agent or IntakeAgentFake(),
        availability=availability,
        customer_email=customer_email or CustomerDeliveryFake(),
        customer_text=customer_text,
        owner_email=owner_email,
    )
    return (
        build_application(
            ports,
            session_factory=session_factory,
            now=Clock(),
            intake_settings=intake_settings or IntakeSettings(max_conversations_per_day=0),
            intake_message_budget=intake_message_budget,
            intake_visitor=intake_visitor,
        ),
        owner,
    )


def http_client(application: Application) -> httpx.AsyncClient:
    async def origins() -> frozenset[str]:
        return frozenset({SITE_URL})

    app = create_app(
        routers=(
            create_public_router(application.public_quotes, intake=application.intake),
            create_portal_router(application.portal, intake=application.intake),
        ),
        cors_origins=origins,
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def seed_owner_thread(application: Application, business_id: BusinessId) -> None:
    """Owner notices anchor to the business's latest inbound message."""

    await application.ingest_service.ingest(
        inbound(business_id, "hi", message_key=f"seed-{business_id}")
    )


async def commands_of(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    command_type: str,
) -> list[OutboxMessage]:
    async with session_factory() as session:
        rows = await session.scalars(
            select(OutboxMessage).where(
                OutboxMessage.business_id == business_id,
                OutboxMessage.command_type == command_type,
            )
        )
        return list(rows.all())


async def conversation_row(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> IntakeRow:
    async with session_factory() as session:
        row = await session.scalar(select(IntakeRow).where(IntakeRow.business_id == business_id))
        assert row is not None
        return row


def collected_turn(**fields: str) -> IntakeTurn:
    return IntakeTurn(
        reply="Noted.",
        collected=IntakeCollected(**fields),
    )


async def reach_awaiting_owner(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    availability: AvailabilityFake,
    agent: IntakeAgentFake | None = None,
    customer_text: CustomerTextFake | None = None,
    sms_consent: bool | None = None,
    notification_email: str | None = None,
    zone: ZoneInfo | None = None,
) -> tuple[Application, OwnerReplyFake, BusinessId, str]:
    """Drives one conversation to ``awaiting_owner`` via HTTP; returns the ref.

    ``zone`` serves the openings in that zone, as Calendly does."""

    business_id = await intake_business(session_factory, notification_email=notification_email)
    offered = slot(datetime.now(UTC))
    if zone is not None:
        offered = AvailableSlot(
            start=offered.start.astimezone(zone), end=offered.end.astimezone(zone)
        )
    availability.slots = (offered,)
    agent = agent or IntakeAgentFake(
        [
            collected_turn(
                name="Jane Doe",
                email=EMAIL,
                phone="+15555550100",
                problem="roof leak",
            ),
            collected_turn(address="1 Main St"),
            IntakeTurn(reply="Great, here are openings.", ready_for_slots=True),
        ]
    )
    application, owner = intake_app(
        session_factory,
        agent=agent,
        availability=availability,
        customer_text=customer_text,
    )
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()

    async with http_client(application) as client:
        start_body = None if sms_consent is None else {"smsConsent": sms_consent}
        created = await client.post(
            f"/v1/businesses/{PUBLIC_KEY}/intake/conversations", json=start_body
        )
        assert created.status_code == 201
        body = created.json()
        assert body["smsConsent"] is sms_consent
        token = body["conversationToken"]
        headers = {"Authorization": f"Bearer {token}"}
        conversation_id = body["conversationId"]

        await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "roof leak at my place"},
            headers=headers,
        )
        await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "1 Main St"},
            headers=headers,
        )
        proposed = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "what times do you have?"},
            headers=headers,
        )
        assert proposed.status_code == 200
        payload = proposed.json()
        assert payload["state"] == "proposing_slots"
        assert payload["slots"] == [
            {"start": offered.start.isoformat(), "end": offered.end.isoformat()}
        ]
        picked = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": f"slot:{offered.start.isoformat()}"},
            headers=headers,
        )
        assert picked.status_code == 200
        assert picked.json()["state"] == "awaiting_owner"

    for _ in range(4):
        await immediate_worker(application).drain()
    row = await conversation_row(session_factory, business_id)
    return application, owner, business_id, row.reference


@pytest.mark.asyncio
async def test_conversation_lifecycle_collects_then_proposes_slots(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    offered = slot(datetime.now(UTC))
    availability = AvailabilityFake((offered,))
    agent = IntakeAgentFake(
        [
            collected_turn(name="Jane Doe", problem="ants"),
            collected_turn(email=EMAIL, phone="+15555550100", address="2 Elm St"),
            IntakeTurn(reply="Here is what is open.", ready_for_slots=True),
        ]
    )
    application, _ = intake_app(session_factory, agent=agent, availability=availability)
    await seed_owner_thread(application, business_id)

    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        assert created.status_code == 201
        body = created.json()
        assert body["state"] == "collecting"
        assert body["slots"] is None
        assert body["reply"]
        token = body["conversationToken"]
        headers = {"Authorization": f"Bearer {token}"}
        conversation_id = body["conversationId"]

        first = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "I saw ants in the kitchen"},
            headers=headers,
        )
        assert first.status_code == 200
        assert first.json()["state"] == "collecting"

        second = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "2 Elm St, jane@example.test"},
            headers=headers,
        )
        assert second.status_code == 200

        third = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "when can someone come?"},
            headers=headers,
        )
        payload = third.json()
        assert payload["state"] == "proposing_slots"
        assert payload["slots"] == [
            {"start": offered.start.isoformat(), "end": offered.end.isoformat()}
        ]
        assert payload["summary"]["email"] == EMAIL

        view = await client.get(f"/v1/intake/conversations/{conversation_id}", headers=headers)
        assert view.status_code == 200
        view_body = view.json()
        assert view_body["state"] == "proposing_slots"
        assert [m["role"] for m in view_body["messages"]][0] == "agent"
        assert any(m["role"] == "user" for m in view_body["messages"])
        assert view_body["slots"] == payload["slots"]

    assert availability.slot_calls
    transcript_ids = [(request.business_id, request.conversation_id) for request in agent.requests]
    assert all(business_id == call[0] for call in transcript_ids)


@pytest.mark.asyncio
async def test_token_from_another_business_cannot_read_or_post(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_a = await intake_business(session_factory)
    business_b = await intake_business(session_factory, public_key="gvb_intake_other")
    assert business_a != business_b
    application, _ = intake_app(session_factory)

    async with http_client(application) as client:
        first = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        second = await client.post("/v1/businesses/gvb_intake_other/intake/conversations")
        assert first.status_code == 201 and second.status_code == 201
        a_id = first.json()["conversationId"]
        b_token = second.json()["conversationToken"]

        read = await client.get(
            f"/v1/intake/conversations/{a_id}",
            headers={"Authorization": f"Bearer {b_token}"},
        )
        assert read.status_code == 401
        posted = await client.post(
            f"/v1/intake/conversations/{a_id}/messages",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {b_token}"},
        )
        assert posted.status_code == 401
        missing_auth = await client.get(f"/v1/intake/conversations/{a_id}")
        assert missing_auth.status_code == 401


@pytest.mark.asyncio
async def test_daily_conversation_cap_rejects_new_conversations(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    application, _ = intake_app(
        session_factory,
        intake_settings=IntakeSettings(max_conversations_per_day=1),
    )
    async with http_client(application) as client:
        first = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        second = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        assert first.status_code == 201
        assert second.status_code == 429


@pytest.mark.asyncio
async def test_per_conversation_message_cap_stops_the_model(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    agent = IntakeAgentFake()
    application, _ = intake_app(
        session_factory,
        agent=agent,
        intake_settings=IntakeSettings(
            max_conversations_per_day=0, max_messages_per_conversation=1
        ),
    )
    assert application.intake is not None
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {token}"}
        await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "hi"},
            headers=headers,
        )
        capped = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "again"},
            headers=headers,
        )
        assert capped.status_code == 200
        assert "message limit" in capped.json()["reply"]
        assert capped.json()["state"] == "closed"
        third = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "still there?"},
            headers=headers,
        )
        assert third.status_code == 409
        view = await client.get(f"/v1/intake/conversations/{conversation_id}", headers=headers)
        user_rows = [m for m in view.json()["messages"] if m["role"] == "user"]
        assert len(user_rows) == 2, "the cap is terminal: nothing past it is stored"
    assert agent.calls == 1


@pytest.mark.asyncio
async def test_a_spent_message_budget_answers_without_the_model(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    agent = IntakeAgentFake()
    asked: list[UUID] = []

    async def budget(business_id: UUID, now: datetime) -> str | None:
        asked.append(business_id)
        return "Gus has answered all he can for this demo." if len(asked) > 1 else None

    application, _ = intake_app(session_factory, agent=agent, intake_message_budget=budget)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        first = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "hi"},
            headers=headers,
        )
        refused = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "again"},
            headers=headers,
        )
    assert first.status_code == refused.status_code == 200
    assert refused.json()["reply"] == "Gus has answered all he can for this demo."
    assert agent.calls == 1
    assert len(set(asked)) == 1


@pytest.mark.asyncio
async def test_slots_are_offered_the_turn_after_details_complete_without_the_model_flag(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    availability = AvailabilityFake((slot(datetime.now(UTC)),))
    agent = IntakeAgentFake(
        [
            collected_turn(
                name="Jane", email=EMAIL, phone="+15555550100", address="2 Elm St", problem="ants"
            ),
            IntakeTurn(reply="Sorry, no times are open; the owner will confirm."),
        ]
    )
    application, _ = intake_app(session_factory, agent=agent, availability=availability)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        first = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "ants"},
            headers=headers,
        )
        second = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "Yes, please."},
            headers=headers,
        )
    assert first.json()["state"] == "collecting", "Gus gets one turn for follow-up questions"
    assert not first.json()["slots"]
    assert second.json()["state"] == "proposing_slots"
    assert second.json()["slots"]
    assert second.json()["reply"] == SLOTS_OFFER_REPLY


@pytest.mark.asyncio
async def test_no_slots_are_offered_until_a_phone_number_is_collected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    availability = AvailabilityFake((slot(datetime.now(UTC)),))
    agent = IntakeAgentFake(
        [
            collected_turn(name="Jane", email=EMAIL, address="2 Elm St", problem="ants"),
            IntakeTurn(reply="Here is what is open.", ready_for_slots=True),
        ]
    )
    application, _ = intake_app(session_factory, agent=agent, availability=availability)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {token}"}
        await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "ants"},
            headers=headers,
        )
        reply = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "when?"},
            headers=headers,
        )
    assert reply.status_code == 200
    assert reply.json()["state"] == "collecting"
    assert not reply.json()["slots"]


@pytest.mark.asyncio
async def test_unlisted_slot_pick_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    offered = slot(datetime.now(UTC))
    not_offered = slot(offered.start, days=1)
    availability = AvailabilityFake((offered,))
    agent = IntakeAgentFake(
        [
            collected_turn(
                name="Jane", email=EMAIL, phone="+15555550100", address="2 Elm St", problem="ants"
            ),
            IntakeTurn(reply="ok", ready_for_slots=True),
        ]
    )
    application, _ = intake_app(session_factory, agent=agent, availability=availability)
    await seed_owner_thread(application, business_id)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {token}"}
        await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "ants"},
            headers=headers,
        )
        proposed = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "times?"},
            headers=headers,
        )
        assert proposed.json()["state"] == "proposing_slots"
        rejected = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": f"slot:{not_offered.start.isoformat()}"},
            headers=headers,
        )
        assert rejected.status_code == 200
        assert rejected.json()["state"] == "proposing_slots"
        assert "listed times" in rejected.json()["reply"]


@pytest.mark.asyncio
async def test_the_transcript_keeps_each_reply_after_the_message_it_answers(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    asked = [f"question {n}" for n in range(6)]
    agent = IntakeAgentFake([IntakeTurn(reply=f"answer {n}") for n in range(6)])
    application, _ = intake_app(session_factory, agent=agent)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        for message in asked:
            await client.post(
                f"/v1/intake/conversations/{conversation_id}/messages",
                json={"message": message},
                headers=headers,
            )
        view = await client.get(f"/v1/intake/conversations/{conversation_id}", headers=headers)
    contents = [m["content"] for m in view.json()["messages"]][1:]
    assert contents == [text for n in range(6) for text in (f"question {n}", f"answer {n}")]


@pytest.mark.asyncio
async def test_agent_reply_with_price_is_scrubbed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    agent = IntakeAgentFake([IntakeTurn(reply="It will cost about $500 for the inspection.")])
    application, _ = intake_app(session_factory, agent=agent)
    await seed_owner_thread(application, business_id)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        reply = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "how much is it?"},
            headers={"Authorization": f"Bearer {token}"},
        )
        body = reply.json()
        assert body["reply"] == PRICE_GUARD_REPLY
        assert "$500" not in body["reply"]
        view = await client.get(
            f"/v1/intake/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        contents = [m["content"] for m in view.json()["messages"]]
        assert not any("$500" in content for content in contents)


@pytest.mark.asyncio
async def test_slot_pick_notifies_owner_with_decision_commands(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    notices = texts_of(owner, "Booking request")
    assert notices
    notice = notices[0]
    assert f"#{reference}" in notice
    assert f"approve booking {reference}" in notice
    assert f"decline booking {reference}" in notice
    assert availability.book_calls == []
    assert await commands_of(session_factory, business_id, OWNER_NOTICE_EMAIL_COMMAND_TYPE) == []


@pytest.mark.asyncio
async def test_booking_request_is_copied_to_the_notification_email(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=AvailabilityFake(), notification_email="owner@example.com"
    )
    assert texts_of(owner, "Booking request")
    copies = await commands_of(session_factory, business_id, OWNER_NOTICE_EMAIL_COMMAND_TYPE)
    assert len(copies) == 1
    assert await commands_of(session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE) == []
    payload = copies[0].payload
    subject, body = payload["subject"], payload["body"]
    assert isinstance(subject, str) and isinstance(body, str)
    assert payload["to"] == "owner@example.com"
    assert subject.startswith(f"[Test Co] New booking request #{reference}")
    assert f"approve booking {reference}" in body
    html = payload["html"]
    assert isinstance(html, str) and f"New booking request #{reference}" in html
    assert "max-width:34rem" in html and "<img" not in html.lower()
    assert "Jane Doe" in html
    # Reply-to-decide is gone: no per-request reply address, no reply prompt.
    assert "reply_to" not in payload and "references" not in payload
    assert "just reply" not in body.lower() and "just reply" not in html.lower()


@pytest.mark.asyncio
async def test_owner_approve_books_directly_and_emails_customer(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.BOOKED))
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-1")
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    replies = texts_of(owner, "Approved booking")
    assert replies and reference in replies[0]
    assert availability.book_calls, "approval must reach the availability port"
    assert availability.book_calls[0].event_type_uri == "cal://event-type"
    row = await conversation_row(session_factory, business_id)
    assert row.state == "approved"
    assert row.booking_kind == "booked"
    assert row.booking_event_type_uri == "cal://event-type"

    email_commands = await commands_of(
        session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE
    )
    assert email_commands
    booked_body = email_commands[0].payload["body"]
    assert isinstance(booked_body, str) and "is booked" in booked_body


@pytest.mark.asyncio
async def test_the_calendar_zone_never_overwrites_one_the_owner_set(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    now = datetime.now(UTC)
    async with session_factory() as session, session.begin():
        repository = SqlBusinessRepository(session)
        # The learner read the zone as unset; the owner saves one before it writes.
        await repository.configure_site(business_id, timezone="America/Chicago", now=now)
        adopted = await repository.adopt_timezone(business_id, "America/Los_Angeles", now)
    assert adopted is False
    async with session_factory() as session:
        business = await SqlBusinessRepository(session).get(business_id)
    assert business is not None and business.timezone == "America/Chicago"


@pytest.mark.asyncio
async def test_times_read_in_the_business_zone_learned_from_the_calendar(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.BOOKED))
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability, zone=ZoneInfo("America/Los_Angeles")
    )
    async with session_factory() as session:
        business = await SqlBusinessRepository(session).get(business_id)
    assert business is not None and business.timezone == "America/Los_Angeles"
    (request,) = texts_of(owner, "Booking request")
    assert "Pacific time" in request and "UTC" not in request

    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-zone")
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    (approved,) = texts_of(owner, "Approved booking")
    assert "Pacific time" in approved and "UTC" not in approved
    emails = await commands_of(session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE)
    body = emails[0].payload["body"]
    assert isinstance(body, str) and "Pacific time" in body and "UTC" not in body


@pytest.mark.asyncio
async def test_owner_approve_with_link_fallback_sends_link_and_text(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    link = "https://calendly.com/test/inspection?s=single-use"
    availability = AvailabilityFake(result=BookingResult(kind=BookingKind.LINK, link=link))
    customer_text = CustomerTextFake()
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability, customer_text=customer_text, sms_consent=True
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-2")
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    row = await conversation_row(session_factory, business_id)
    assert row.state == "approved" and row.booking_kind == "link"
    email_commands = await commands_of(
        session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE
    )
    link_body = email_commands[0].payload["body"] if email_commands else None
    assert isinstance(link_body, str) and link in link_body
    text_commands = await commands_of(
        session_factory, business_id, INTAKE_CUSTOMER_TEXT_COMMAND_TYPE
    )
    assert text_commands, "a phone was collected, so a text goes out"
    assert customer_text.requests, "the text command dispatched"
    assert link in customer_text.requests[0].text


@pytest.mark.asyncio
async def test_owner_decline_marks_declined_and_emails_reason(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(
            business_id,
            f"decline booking {reference} booked solid\nthis week",
            message_key="decline-1",
        )
    )
    for _ in range(4):
        await immediate_worker(application).drain()

    assert texts_of(owner, "Declined booking")
    row = await conversation_row(session_factory, business_id)
    assert row.state == "declined"
    assert row.decision_reason is not None and "\n" not in row.decision_reason
    email_commands = await commands_of(
        session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE
    )
    assert email_commands
    body = email_commands[0].payload["body"]
    assert isinstance(body, str)
    assert "booked solid this week" in body
    assert CALENDLY_LINK in body
    arrange = await commands_of(session_factory, business_id, INTAKE_BOOKING_ARRANGE_COMMAND_TYPE)
    assert arrange == []
    assert availability.book_calls == []


@pytest.mark.asyncio
async def test_unknown_and_repeat_decisions_get_clear_replies(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    worker = immediate_worker(application)

    await application.ingest_service.ingest(
        inbound(business_id, "approve booking deadbeef", message_key="approve-unknown")
    )
    for _ in range(3):
        await worker.drain()
    assert any("can't find booking deadbeef" in t for t in texts_of(owner, "I can't find"))

    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-a")
    )
    for _ in range(5):
        await worker.drain()
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-b")
    )
    for _ in range(3):
        await worker.drain()
    assert any("already approved" in t for t in texts_of(owner, "Booking"))
    arrange = await commands_of(session_factory, business_id, INTAKE_BOOKING_ARRANGE_COMMAND_TYPE)
    assert len(arrange) == 1


@pytest.mark.asyncio
async def test_needs_human_escalates_to_owner_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    agent = IntakeAgentFake(
        [
            IntakeTurn(
                reply="Hmm, I'm not sure.",
                needs_human=True,
                summary="Customer asked about commercial pricing tiers.",
            ),
            IntakeTurn(
                reply="Still unsure.",
                needs_human=True,
                summary="Again ambiguous.",
            ),
        ]
    )
    application, owner = intake_app(session_factory, agent=agent)
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {token}"}
        for text in ("do you do commercial buildings?", "what about warehouses?"):
            reply = await client.post(
                f"/v1/intake/conversations/{conversation_id}/messages",
                json={"message": text},
                headers=headers,
            )
            assert reply.status_code == 200
    for _ in range(3):
        await immediate_worker(application).drain()
    escalations = texts_of(owner, "Chat ")
    assert len(escalations) == 1
    assert "commercial pricing" in escalations[0]


@pytest.mark.asyncio
async def test_portal_conversation_is_prefilled_and_linked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    agent = IntakeAgentFake()
    application, _ = intake_app(session_factory, agent=agent)
    assert application.intake is not None

    customer = CustomerRecord(
        customer_id=CustomerId(uuid4()),
        business_id=business_id,
        email=EMAIL,
        display_name="Jane Doe",
        phone="+15555550100",
        created_at=NOW,
    )
    async with session_factory() as db_session:
        business_row = await SqlBusinessRepository(db_session).get(business_id)
    assert business_row is not None
    start = await application.intake.start_portal_conversation(business_row, customer)
    assert start.conversation.customer_id == customer.customer_id
    assert start.conversation.collected.email == EMAIL
    assert "Welcome back" in start.reply


@pytest.mark.asyncio
async def test_declined_conversation_rejects_new_messages(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, _, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"decline booking {reference}", message_key="decline-x")
    )
    for _ in range(4):
        await immediate_worker(application).drain()

    assert application.intake is not None
    async with application.unit_of_work_factory() as unit_of_work:
        conversation = await unit_of_work.intake_conversations.find_by_reference(
            business_id, reference
        )
        assert conversation is not None
    with pytest.raises(IntakeClosedError):
        await application.intake.post_message(conversation, "one more thing")


@pytest.mark.asyncio
async def test_retried_booking_reconciles_instead_of_double_booking(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A crashed attempt leaves the marker set; the retry reconciles first."""

    availability = AvailabilityFake(found=BookingResult(kind=BookingKind.BOOKED))
    application, _, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    async with session_factory() as session:
        await session.execute(
            update(IntakeRow)
            .where(IntakeRow.business_id == business_id)
            .values(
                booking_attempted_at=NOW,
                booking_event_type_uri="cal://original-type",
            )
        )
        await session.commit()
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-reconciled")
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    assert availability.find_calls, "a marked attempt reconciles before booking"
    assert availability.find_calls[0].event_type_uri == "cal://original-type"
    assert availability.book_calls == [], "the recovered booking is not created twice"
    row = await conversation_row(session_factory, business_id)
    assert row.booking_kind == "booked"
    email_commands = await commands_of(
        session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE
    )
    assert email_commands, "the customer still gets the confirmation"


@pytest.mark.asyncio
async def test_retried_booking_still_books_when_reconcile_finds_nothing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake(found=None)
    application, _, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    async with session_factory() as session:
        await session.execute(
            update(IntakeRow)
            .where(IntakeRow.business_id == business_id)
            .values(booking_attempted_at=NOW)
        )
        await session.commit()
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-retry")
    )
    for _ in range(6):
        await immediate_worker(application).drain()

    assert availability.find_calls
    assert availability.book_calls, "nothing found — the booking is made"
    row = await conversation_row(session_factory, business_id)
    assert row.booking_kind == "booked"


@pytest.mark.asyncio
async def test_approval_after_the_requested_time_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, owner, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability
    )
    past = datetime.now(UTC) - timedelta(hours=2)
    async with session_factory() as session:
        await session.execute(
            update(IntakeRow)
            .where(IntakeRow.business_id == business_id)
            .values(requested_slot_start=past, requested_slot_end=past + timedelta(hours=1))
        )
        await session.commit()
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-late")
    )
    for _ in range(4):
        await immediate_worker(application).drain()

    replies = [text for text in texts_of(owner, "Booking") if "already passed" in text]
    assert replies and reference in replies[0]
    row = await conversation_row(session_factory, business_id)
    assert row.state == "awaiting_owner"
    assert availability.book_calls == []
    arrange = await commands_of(session_factory, business_id, INTAKE_BOOKING_ARRANGE_COMMAND_TYPE)
    assert arrange == []


@pytest.mark.asyncio
async def test_failed_customer_deliveries_raise_so_the_outbox_retries(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    class FailingDelivery:
        async def deliver(self, request: CustomerDeliveryRequest) -> DeliveryReceipt:
            return DeliveryReceipt(
                status=DeliveryStatus.FAILED, detail="mailbox rejected", occurred_at=NOW
            )

    class FailingText:
        async def send_text(self, request: CustomerTextRequest) -> DeliveryReceipt:
            return DeliveryReceipt(
                status=DeliveryStatus.FAILED, detail="carrier rejected", occurred_at=NOW
            )

    email = SendIntakeCustomerEmailService(FailingDelivery())
    with pytest.raises(IntakeDeliveryError):
        await email.send(
            BusinessId(uuid4()),
            {
                "to": "customer@example.test",
                "subject": "About your appointment",
                "body": "details",
                "idempotency_key": "key-1",
            },
        )
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    await consent_to_texts(session_factory, business_id, EMAIL)
    async with session_factory() as session:
        customer = await SqlCustomerRepository(session).find_by_email(business_id, EMAIL)
    assert customer is not None
    text = SendIntakeCustomerTextService(FailingText(), SqlUnitOfWorkFactory(session_factory))
    with pytest.raises(IntakeDeliveryError):
        await text.send(
            business_id,
            {
                "customer_id": str(customer.customer_id),
                "phone": "+15555550100",
                "text": "details",
                "idempotency_key": "key-2",
            },
        )


@pytest.mark.asyncio
async def test_agent_outage_returns_the_unavailable_reply(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A provider failure must not 500 the chat or lose the message."""

    await intake_business(session_factory)
    agent = IntakeAgentFake(error=IntakeAgentError("openai down"))
    application, _ = intake_app(session_factory, agent=agent)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {token}"}
        reply = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "I need an inspection"},
            headers=headers,
        )
        assert reply.status_code == 200
        assert reply.json()["reply"] == UNAVAILABLE_REPLY
        assert reply.json()["state"] == "collecting"
        view = await client.get(f"/v1/intake/conversations/{conversation_id}", headers=headers)
        contents = [m["content"] for m in view.json()["messages"]]
        assert "I need an inspection" in contents


@pytest.mark.asyncio
async def test_availability_outage_keeps_the_conversation_collecting(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    availability = AvailabilityFake(slot_error=AvailabilityError("calendly down"))
    agent = IntakeAgentFake(
        [
            collected_turn(
                name="Jane", email=EMAIL, phone="+15555550100", address="2 Elm St", problem="ants"
            ),
            IntakeTurn(reply="Here is what is open.", ready_for_slots=True),
        ]
    )
    application, _ = intake_app(session_factory, agent=agent, availability=availability)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        token = created.json()["conversationToken"]
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {token}"}
        await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "ants"},
            headers=headers,
        )
        reply = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "when?"},
            headers=headers,
        )
        assert reply.status_code == 200
        assert reply.json()["reply"] == NO_AVAILABILITY_REPLY
        assert reply.json()["state"] == "collecting"


def test_pick_offer_slots_anchors_the_day_window_on_the_slot_timezone() -> None:
    pacific = ZoneInfo("America/Los_Angeles")
    # Sunday 6 PM Pacific is already Monday 2 AM UTC: the business's Monday
    # must still be inside the offer window.
    now = datetime(2026, 1, 12, 2, 0, tzinfo=UTC)
    start = datetime(2026, 1, 12, 9, 0, tzinfo=pacific)
    offered = pick_offer_slots(
        (AvailableSlot(start=start, end=start + timedelta(hours=1)),), now=now
    )
    assert offered, "the business-local Monday is lost when anchored on UTC"
    assert offered[0].start == start


PROFILE_BRIEF = "Acme Web builds websites for plumbers; you book a free discovery call."
PROFILE_QUESTIONS = "whether they want a website, automation, or both; their trade"
PROFILE_OPENING = "Hi! Are you after a website, an automation, or both?"


async def profiled_business(session_factory: async_sessionmaker[AsyncSession]) -> BusinessId:
    business_id = await intake_business(session_factory)
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id,
            intake_brief=PROFILE_BRIEF,
            intake_questions=PROFILE_QUESTIONS,
            intake_opening=PROFILE_OPENING,
            now=NOW,
        )
        await session.commit()
    return business_id


@pytest.mark.asyncio
async def test_intake_profile_is_stored_and_kept_when_other_fields_change(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await profiled_business(session_factory)
    async with session_factory() as session:
        record = await SqlBusinessRepository(session).configure_site(
            business_id, display_name="Renamed Co", now=NOW
        )
        await session.commit()
    assert record.display_name == "Renamed Co"
    assert record.intake_profile == IntakeProfile(
        brief=PROFILE_BRIEF, questions=PROFILE_QUESTIONS, opening=PROFILE_OPENING
    )
    assert record.intake_profile.is_configured


@pytest.mark.asyncio
async def test_intake_profile_drives_opening_agent_request_and_owner_notice(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await profiled_business(session_factory)
    offered = slot(datetime.now(UTC))
    availability = AvailabilityFake((offered,))
    agent = IntakeAgentFake(
        [
            collected_turn(
                name="Jane Doe",
                email=EMAIL,
                phone="+15555550100",
                details="A new website",
                notes="Plumber; wants it live by May",
            ),
            IntakeTurn(reply="Here is what is open.", ready_for_slots=True),
        ]
    )
    application, owner = intake_app(session_factory, agent=agent, availability=availability)
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()

    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        assert created.status_code == 201
        assert created.json()["reply"] == PROFILE_OPENING
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "I need a website for my plumbing business"},
            headers=headers,
        )
        # No address was ever asked for: a profiled business is ready on
        # name, email and details.
        proposed = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": "what times do you have?"},
            headers=headers,
        )
        payload = proposed.json()
        assert payload["state"] == "proposing_slots"
        assert payload["summary"]["details"] == "A new website"
        assert payload["summary"]["problem"] == "A new website"
        assert payload["summary"]["notes"] == "Plumber; wants it live by May"
        picked = await client.post(
            f"/v1/intake/conversations/{conversation_id}/messages",
            json={"message": f"slot:{offered.start.isoformat()}"},
            headers=headers,
        )
        assert picked.json()["state"] == "awaiting_owner"

    assert agent.requests
    assert all(request.brief == PROFILE_BRIEF for request in agent.requests)
    assert all(request.questions == PROFILE_QUESTIONS for request in agent.requests)
    for _ in range(4):
        await immediate_worker(application).drain()
    notices = texts_of(owner, "Booking request")
    assert len(notices) == 1
    assert "Jane Doe. A new website." in notices[0]
    assert "address" not in notices[0]
    assert "Notes: Plumber; wants it live by May" in notices[0]


@pytest.mark.asyncio
async def test_a_known_visitor_opens_with_their_details_and_only_tells_the_job(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    visitor = IntakeVisitor(
        collected=IntakeCollected(
            name="Sam Rivera", email="sam@example.com", phone="+15555550199", address="12 Oak Ave"
        ),
        opening="Hi Sam! I'm {business}'s assistant.",
    )

    async def known(asked: UUID) -> IntakeVisitor | None:
        return visitor if asked == business_id else None

    agent = IntakeAgentFake(
        [collected_turn(name="Someone Else", details="A new patio"), IntakeTurn(reply="ok")]
    )
    availability = AvailabilityFake((slot(datetime.now(UTC)),))
    application, _ = intake_app(
        session_factory, agent=agent, availability=availability, intake_visitor=known
    )
    await seed_owner_thread(application, business_id)
    async with http_client(application) as client:

        async def chat() -> tuple[list[str], dict[str, object]]:
            created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
            assert created.json()["reply"] == "Hi Sam! I'm Test Co's assistant."
            path = f"/v1/intake/conversations/{created.json()['conversationId']}/messages"
            headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
            states: list[str] = []
            body: dict[str, object] = {}
            for text in ("A new patio", "about 200 square feet"):
                response = await client.post(path, json={"message": text}, headers=headers)
                body = response.json()
                states.append(str(body["state"]))
            slots = body["slots"]
            assert isinstance(slots, list)
            picked = await client.post(
                path, json={"message": f"slot:{slots[0]['start']}"}, headers=headers
            )
            return states, picked.json()

        first, booked = await chat()
        # A second chat in the same sandbox (another tab) shares Sam's e-mail
        # with the live request, and still gets openings of its own.
        agent._turns = [collected_turn(details="A new patio"), IntakeTurn(reply="ok")]
        second, _ = await chat()
    assert agent.requests[0].collected == visitor.collected
    assert first == second == ["collecting", "proposing_slots"]
    assert booked["state"] == "awaiting_owner"


@pytest.mark.asyncio
async def test_intake_without_profile_keeps_default_opening_and_needs_an_address(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    offered = slot(datetime.now(UTC))
    availability = AvailabilityFake((offered,))
    agent = IntakeAgentFake(
        [
            collected_turn(name="Jane", email=EMAIL, phone="+15555550100", details="ants"),
            IntakeTurn(reply="ok", ready_for_slots=True),
            IntakeTurn(
                reply="ok",
                collected=IntakeCollected(address="2 Elm St"),
                ready_for_slots=True,
            ),
        ]
    )
    application, _ = intake_app(session_factory, agent=agent, availability=availability)
    await seed_owner_thread(application, business_id)
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        assert created.json()["reply"] == OPENING_REPLY
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        states = []
        for text in ("ants", "times?", "2 Elm St"):
            response = await client.post(
                f"/v1/intake/conversations/{conversation_id}/messages",
                json={"message": text},
                headers=headers,
            )
            states.append(response.json()["state"])
    assert states == ["collecting", "collecting", "proposing_slots"]
    assert all(request.brief is None and request.questions is None for request in agent.requests)


def test_intake_collected_reads_legacy_problem_as_details() -> None:
    legacy = IntakeCollected.model_validate(
        {"name": "Jane", "email": EMAIL, "address": "2 Elm St", "problem": "ants"}
    )
    assert legacy.details == "ants"
    assert legacy.notes is None
    assert "problem" not in legacy.as_stored()
    assert legacy.as_stored()["details"] == "ants"
    both = IntakeCollected.model_validate({"details": "mold", "problem": "ants"})
    assert both.details == "mold"


def test_intake_notes_accumulate_without_repeating() -> None:
    first = IntakeCollected().merge(IntakeCollected(notes="Plumber"))
    assert first.notes == "Plumber"
    assert first.merge(IntakeCollected(notes="plumber")).notes == "Plumber"
    extended = first.merge(IntakeCollected(notes="Plumber, 3 vans"))
    assert extended.notes == "Plumber, 3 vans"
    assert extended.merge(IntakeCollected(notes="Uses Jobber")).notes == (
        "Plumber, 3 vans; Uses Jobber"
    )
    assert extended.merge(IntakeCollected(notes="vans")).notes == "Plumber, 3 vans; vans"
    corrected = IntakeCollected(notes="No water damage").merge(
        IntakeCollected(notes="Water damage")
    )
    assert corrected.notes == "No water damage; Water damage"


def test_intake_notes_do_not_repeat_the_running_summary_on_reschedule() -> None:
    summary = "Business: Ace Plumbing. Trade: plumbing, 3 vans. Current tools: paper and Jobber."
    collected = IntakeCollected(notes=summary)
    for _ in range(3):
        collected = collected.merge(IntakeCollected(notes=summary))
    assert collected.notes == summary
    grown = collected.merge(IntakeCollected(notes=f"{summary} Wants online booking."))
    assert grown.notes == f"{summary} Wants online booking."
    updated = grown.merge(IntakeCollected(notes="Current tools: Jobber only."))
    assert updated.notes is not None
    assert updated.notes.count("Current tools") == 1
    assert "Jobber only" in updated.notes


def test_intake_booking_notice_reports_a_missing_address_only_without_a_profile() -> None:
    conversation = IntakeConversation(
        conversation_id=IntakeConversationId(uuid4()),
        business_id=BusinessId(uuid4()),
        reference="abc123",
        token_hash="0" * 64,
        channel="web",
        collected=IntakeCollected(name="Jane", email=EMAIL, details="ants"),
        expires_at=NOW + timedelta(days=1),
        created_at=NOW,
        updated_at=NOW,
    )
    default = booking_request_notice(conversation, business_name="Test Co")
    assert default.startswith("Booking request #abc123 — Jane, address unknown. ants.")
    profiled = booking_request_notice(
        conversation,
        business_name="Test Co",
        profile=IntakeProfile(brief=PROFILE_BRIEF, questions=PROFILE_QUESTIONS),
    )
    assert profiled.startswith("Booking request #abc123 — Jane. ants.")
    opening_only = booking_request_notice(
        conversation,
        business_name="Test Co",
        profile=IntakeProfile(opening=PROFILE_OPENING),
    )
    assert "address unknown" in opening_only


def test_only_a_business_s_own_questions_waive_the_address() -> None:
    collected = IntakeCollected(name="Jane", email=EMAIL, phone="+15555550100", details="ants")
    assert collected.summary() is None, "the generic flow is not complete without an address"
    assert IntakeProfile(brief=PROFILE_BRIEF, opening=PROFILE_OPENING).requires_address
    assert IntakeProfile().requires_address
    custom = IntakeProfile(questions=PROFILE_QUESTIONS)
    assert not custom.requires_address
    summary = collected.summary(address_required=custom.requires_address)
    assert summary is not None and summary["details"] == "ants"
    with_address = collected.model_copy(update={"address": "2 Elm St"})
    assert with_address.summary() is not None


def test_a_known_customer_without_a_phone_can_still_reach_slots() -> None:
    collected = IntakeCollected(name="Jane", email=EMAIL, details="ants", address="2 Elm St")
    assert not collected.ready_for_slots
    assert collected.summary() is None
    assert collected.is_complete(address_required=True, phone_required=False)
    summary = collected.summary(phone_required=False)
    assert summary is not None and summary["phone"] is None


async def customer_row(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> CustomerRecord:
    async with session_factory() as session:
        customer = await SqlCustomerRepository(session).find_by_email(business_id, EMAIL)
    assert customer is not None
    return customer


async def approve_link_booking(
    session_factory: async_sessionmaker[AsyncSession], *, sms_consent: bool | None
) -> tuple[BusinessId, CustomerTextFake]:
    availability = AvailabilityFake(
        result=BookingResult(kind=BookingKind.LINK, link="https://calendly.com/test/x?s=1")
    )
    customer_text = CustomerTextFake()
    application, _, business_id, reference = await reach_awaiting_owner(
        session_factory,
        availability=availability,
        customer_text=customer_text,
        sms_consent=sms_consent,
    )
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-consent")
    )
    for _ in range(6):
        await immediate_worker(application).drain()
    return business_id, customer_text


@pytest.mark.asyncio
async def test_sms_consent_is_stored_and_copied_to_the_customer_on_the_booking_request(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id, customer_text = await approve_link_booking(session_factory, sms_consent=True)

    row = await conversation_row(session_factory, business_id)
    assert row.sms_consent is True and row.sms_consent_at is not None
    customer = await customer_row(session_factory, business_id)
    assert customer.sms_consent is True and customer.sms_consent_at is not None
    assert len(customer_text.requests) == 1
    assert customer_text.requests[0].phone_number == "+15555550100"


@pytest.mark.asyncio
@pytest.mark.parametrize("sms_consent", [None, False])
async def test_booking_is_emailed_but_never_texted_without_sms_consent(
    session_factory: async_sessionmaker[AsyncSession], sms_consent: bool | None
) -> None:
    business_id, customer_text = await approve_link_booking(
        session_factory, sms_consent=sms_consent
    )

    assert await commands_of(session_factory, business_id, INTAKE_CUSTOMER_EMAIL_COMMAND_TYPE)
    assert await commands_of(session_factory, business_id, INTAKE_CUSTOMER_TEXT_COMMAND_TYPE) == []
    assert customer_text.requests == []
    customer = await customer_row(session_factory, business_id)
    assert customer.sms_consent is sms_consent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sms_consent", "promise"),
    [(True, "by email and text"), (False, "by email once"), (None, "by email once")],
)
async def test_the_pick_reply_promises_a_text_only_with_sms_consent(
    session_factory: async_sessionmaker[AsyncSession], sms_consent: bool | None, promise: str
) -> None:
    _, _, business_id, _ = await reach_awaiting_owner(
        session_factory, availability=AvailabilityFake(), sms_consent=sms_consent
    )

    row = await conversation_row(session_factory, business_id)
    async with session_factory() as session:
        messages = await SqlIntakeMessageRepository(session).list_for(
            business_id, IntakeConversationId(row.id)
        )
    reply = [m.content for m in messages if m.role is IntakeMessageRole.AGENT][-1]
    assert reply.startswith("Great — I've requested")
    assert promise in reply
    if sms_consent is not True:
        assert "text" not in reply


@pytest.mark.asyncio
async def test_sms_consent_on_a_message_updates_the_conversation_and_response(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await intake_business(session_factory)
    application, _ = intake_app(
        session_factory, agent=IntakeAgentFake([IntakeTurn(reply="What do you need?")])
    )
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        body = created.json()
        assert body["smsConsent"] is None
        headers = {"Authorization": f"Bearer {body['conversationToken']}"}
        url = f"/v1/intake/conversations/{body['conversationId']}"
        posted = await client.post(
            f"{url}/messages",
            json={"message": "hi", "sms_consent": True},
            headers=headers,
        )
        assert posted.status_code == 200
        assert posted.json()["smsConsent"] is True
        viewed = await client.get(url, headers=headers)
        assert viewed.json()["smsConsent"] is True


@pytest.mark.asyncio
async def test_queued_intake_text_is_dropped_unless_the_customer_consented(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    customer_text = CustomerTextFake()
    service = SendIntakeCustomerTextService(customer_text, SqlUnitOfWorkFactory(session_factory))
    payload = {"phone": "+15555550100", "text": "details", "idempotency_key": "key-3"}

    assert await service.send(business_id, payload) is IntakeTextStatus.NO_CONSENT
    await consent_to_texts(session_factory, business_id, EMAIL, consent=False)
    customer = await customer_row(session_factory, business_id)
    with_customer = {**payload, "customer_id": str(customer.customer_id)}
    assert await service.send(business_id, with_customer) is IntakeTextStatus.NO_CONSENT
    assert customer_text.requests == []
    await consent_to_texts(session_factory, business_id, EMAIL, consent=True)
    assert await service.send(business_id, with_customer) is IntakeTextStatus.SENT
    assert len(customer_text.requests) == 1


@pytest.mark.asyncio
async def test_an_older_answer_never_overwrites_a_newer_one(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    await consent_to_texts(session_factory, business_id, EMAIL, consent=True)
    customer = await customer_row(session_factory, business_id)
    assert customer.sms_consent_at is not None
    withdrawn_at = customer.sms_consent_at + timedelta(hours=2)
    async with session_factory() as session:
        customers = SqlCustomerRepository(session)
        await customers.set_sms_consent(business_id, customer.customer_id, False, withdrawn_at)
        await customers.set_sms_consent(
            business_id, customer.customer_id, True, withdrawn_at - timedelta(hours=1)
        )
        await session.commit()

    customer = await customer_row(session_factory, business_id)
    assert customer.sms_consent is False
    assert customer.sms_consent_at == withdrawn_at


@pytest.mark.asyncio
async def test_portal_start_answer_reaches_the_customer_immediately(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    application, _ = intake_app(session_factory)
    assert application.intake is not None
    await consent_to_texts(session_factory, business_id, EMAIL, consent=True)
    customer = await customer_row(session_factory, business_id)
    async with session_factory() as db_session:
        business_row = await SqlBusinessRepository(db_session).get(business_id)
    assert business_row is not None

    start = await application.intake.start_portal_conversation(
        business_row, customer, sms_consent=False
    )

    assert start.conversation.sms_consent is False
    assert (await customer_row(session_factory, business_id)).sms_consent is False
    inherited = await application.intake.start_portal_conversation(
        business_row, await customer_row(session_factory, business_id)
    )
    assert inherited.conversation.sms_consent is False


@pytest.mark.asyncio
async def test_a_no_is_accepted_after_approval_and_a_new_yes_is_too(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    availability = AvailabilityFake()
    application, _, business_id, reference = await reach_awaiting_owner(
        session_factory, availability=availability, sms_consent=True
    )
    assert application.intake is not None
    await application.ingest_service.ingest(
        inbound(business_id, f"approve booking {reference}", message_key="approve-withdraw")
    )
    for _ in range(6):
        await immediate_worker(application).drain()
    async with session_factory() as session:
        conversation = await SqlIntakeConversationRepository(session).find_by_reference(
            business_id, reference
        )
    assert conversation is not None and conversation.state is IntakeState.APPROVED

    withdrawn = await application.intake.record_sms_consent(conversation, False)

    assert withdrawn.sms_consent is False
    assert (await conversation_row(session_factory, business_id)).sms_consent is False
    assert (await customer_row(session_factory, business_id)).sms_consent is False
    again = await application.intake.record_sms_consent(withdrawn, True)
    assert again.sms_consent is True, "a booked chat stays open, so consent can still change"
    assert (await customer_row(session_factory, business_id)).sms_consent is True


@pytest.mark.asyncio
async def test_an_adopted_zone_is_seen_by_later_reads_in_the_same_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    async with session_factory() as session:
        businesses = SqlBusinessRepository(session)
        before = await businesses.get(business_id)
        assert before is not None and before.timezone is None

        assert await businesses.adopt_timezone(business_id, "America/Los_Angeles", NOW)
        after = await businesses.get(business_id)
        assert after is not None and after.timezone == "America/Los_Angeles"


@pytest.mark.asyncio
async def test_offer_line_and_visit_question_are_per_business_and_the_answer_is_saved(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    offer_line = "Let's schedule a walk-through so we can send you a quote."
    question = "Anything we should know before we stop by?"
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id, intake_offer_line=offer_line, intake_visit_question=question, now=NOW
        )
        await session.commit()
    offered = slot(datetime.now(UTC))
    agent = IntakeAgentFake(
        [
            collected_turn(
                name="Jane", email=EMAIL, phone="+15555550100", address="2 Elm St", problem="lawn"
            ),
            IntakeTurn(reply="Here you go.", ready_for_slots=True),
        ]
    )
    application, owner_replies = intake_app(
        session_factory, agent=agent, availability=AvailabilityFake((offered,))
    )
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        url = f"/v1/intake/conversations/{conversation_id}/messages"
        await client.post(url, json={"message": "new lawn"}, headers=headers)
        offer = await client.post(url, json={"message": "sounds good"}, headers=headers)
        picked = await client.post(
            url, json={"message": f"slot:{offered.start.isoformat()}"}, headers=headers
        )
        answered = await client.post(
            url, json={"message": "Gate code 4412,  dog in the back."}, headers=headers
        )
    await immediate_worker(application).drain()
    assert offer.json()["reply"] == offer_line
    assert picked.json()["state"] == "awaiting_owner"
    assert picked.json()["reply"].endswith(question)
    assert answered.json()["reply"] == VISIT_NOTED_REPLY
    row = await conversation_row(session_factory, business_id)
    assert "Before the visit: Gate code 4412, dog in the back." in str(row.collected["notes"])
    assert texts_of(owner_replies, "Jane added a note to request") == [
        f"Jane added a note to request {row.reference}: "
        "Before the visit: Gate code 4412, dog in the back."
    ]


@pytest.mark.asyncio
async def test_the_visit_answer_turn_keeps_the_notes_as_they_were_plus_the_answer(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    question = "Anything we should know before we stop by?"
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id, intake_visit_question=question, now=NOW
        )
        await session.commit()
    offered = slot(datetime.now(UTC))
    agent = IntakeAgentFake(
        [
            IntakeTurn(
                reply="Noted.",
                collected=IntakeCollected(
                    name="Jane",
                    email=EMAIL,
                    phone="+15555550100",
                    address="2 Elm St",
                    details="lawn",
                    notes="About 800 sq ft.",
                ),
            ),
            IntakeTurn(reply="Here you go.", ready_for_slots=True),
            IntakeTurn(
                reply="Got it.",
                collected=IntakeCollected(
                    property_type="house", notes="Customer will be home; dog in the back."
                ),
            ),
        ]
    )
    application, _ = intake_app(
        session_factory, agent=agent, availability=AvailabilityFake((offered,))
    )
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        url = f"/v1/intake/conversations/{conversation_id}/messages"
        await client.post(url, json={"message": "new lawn"}, headers=headers)
        await client.post(url, json={"message": "sounds good"}, headers=headers)
        await client.post(
            url, json={"message": f"slot:{offered.start.isoformat()}"}, headers=headers
        )
        await client.post(url, json={"message": "I'll be home, dog in the back."}, headers=headers)
    row = await conversation_row(session_factory, business_id)
    assert row.collected["notes"] == (
        "About 800 sq ft. Before the visit: I'll be home, dog in the back."
    )
    assert row.collected["propertyType"] == "house"


@pytest.mark.asyncio
async def test_a_booking_change_after_the_visit_question_is_not_saved_as_the_answer(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    question = "Anything we should know before we stop by?"
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id, intake_visit_question=question, now=NOW
        )
        await session.commit()
    offered = slot(datetime.now(UTC))
    agent = IntakeAgentFake(
        [
            collected_turn(
                name="Jane", email=EMAIL, phone="+15555550100", address="2 Elm St", problem="lawn"
            ),
            IntakeTurn(reply="Here you go.", ready_for_slots=True),
            IntakeTurn(reply="Let's find another time.", wants_reschedule=True),
        ]
    )
    application, _ = intake_app(
        session_factory, agent=agent, availability=AvailabilityFake((offered,))
    )
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        url = f"/v1/intake/conversations/{conversation_id}/messages"
        await client.post(url, json={"message": "new lawn"}, headers=headers)
        await client.post(url, json={"message": "sounds good"}, headers=headers)
        await client.post(
            url, json={"message": f"slot:{offered.start.isoformat()}"}, headers=headers
        )
        changed = await client.post(
            url, json={"message": "Actually, can we do a different day?"}, headers=headers
        )
    assert changed.json()["reply"] != VISIT_NOTED_REPLY
    row = await conversation_row(session_factory, business_id)
    assert "Before the visit" not in str(row.collected["notes"])


@pytest.mark.asyncio
async def test_the_visit_answer_is_saved_when_the_model_is_down(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = await intake_business(session_factory)
    async with session_factory() as session:
        await SqlBusinessRepository(session).configure_site(
            business_id, intake_visit_question="Anything we should know?", now=NOW
        )
        await session.commit()
    offered = slot(datetime.now(UTC))
    agent = IntakeAgentFake(
        [
            collected_turn(
                name="Jane", email=EMAIL, phone="+15555550100", address="2 Elm St", problem="lawn"
            ),
            IntakeTurn(reply="Here you go.", ready_for_slots=True),
        ]
    )
    application, owner_replies = intake_app(
        session_factory, agent=agent, availability=AvailabilityFake((offered,))
    )
    await seed_owner_thread(application, business_id)
    await immediate_worker(application).drain()
    async with http_client(application) as client:
        created = await client.post(f"/v1/businesses/{PUBLIC_KEY}/intake/conversations")
        conversation_id = created.json()["conversationId"]
        headers = {"Authorization": f"Bearer {created.json()['conversationToken']}"}
        url = f"/v1/intake/conversations/{conversation_id}/messages"
        await client.post(url, json={"message": "new lawn"}, headers=headers)
        await client.post(url, json={"message": "sounds good"}, headers=headers)
        await client.post(
            url, json={"message": f"slot:{offered.start.isoformat()}"}, headers=headers
        )
        agent.error = IntakeAgentError("openai down")
        answered = await client.post(url, json={"message": "Side gate."}, headers=headers)
    await immediate_worker(application).drain()
    assert answered.json()["reply"] == VISIT_NOTED_REPLY
    row = await conversation_row(session_factory, business_id)
    assert "Before the visit: Side gate." in str(row.collected["notes"])
    assert texts_of(owner_replies, "Jane added a note to request")


def test_a_visit_note_lands_in_full_when_the_notes_are_full() -> None:
    full = IntakeCollected(notes="x" * INTAKE_NOTES_MAX_CHARS)
    note = visit_note("Gate code 4412.")
    notes = with_visit_note(full, note).notes or ""
    assert notes.endswith("; Before the visit: Gate code 4412.")
    assert len(notes) <= INTAKE_NOTES_MAX_CHARS
