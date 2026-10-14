"""Visitor sandboxes: every demo visitor gets their own copy of the demo business."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.owner import OwnerAuthenticationError, OwnerService
from gvas.config import DemoSettings
from gvas.domain.intake import IntakeMessageRole
from gvas.infrastructure.demo import DemoBookedEvents
from gvas.infrastructure.intake_models import IntakeMessage
from gvas.infrastructure.models import Business, Customer, DemoSandbox, QuoteRecord
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory
from gvas.infrastructure.usage_models import UsageLedgerMonth
from gvas.interfaces.demo_sandboxes import (
    DAILY_LIMIT_REPLY,
    SAM,
    VISITOR_LIMIT_REPLY,
    DemoSandboxes,
    SandboxAuthenticationError,
    SandboxCapacityError,
    SandboxError,
)
from gvas.interfaces.http.sandbox import OWNER_TEXTS_USED_UP_REPLY, create_sandbox_router

NOW = datetime(2026, 10, 14, 18, 30, tzinfo=UTC)
SLUG = "larkspur"
TEMPLATE_KEY = "gvb_template"


def settings(**overrides: object) -> DemoSettings:
    values: dict[str, object] = {"mode": True, "sandbox_template_slug": SLUG}
    values.update(overrides)
    return DemoSettings.model_validate(values)


async def template(session_factory: async_sessionmaker[AsyncSession]) -> Business:
    business = Business(
        id=uuid4(),
        slug=SLUG,
        name="Larkspur Lawn & Garden",
        display_name="Larkspur Lawn & Garden",
        site_url="https://demo.example",
        public_key=TEMPLATE_KEY,
        intake_brief="Garden design and upkeep in the East Bay.",
        intake_questions="How big is the yard?",
        owner_email="owner@larkspur.example",
        notification_email="owner@larkspur.example",
        timezone="America/Los_Angeles",
        created_at=NOW,
        updated_at=NOW,
    )
    async with session_factory() as session:
        session.add(business)
        await session.commit()
    return business


def owner_service(session_factory: async_sessionmaker[AsyncSession]) -> OwnerService:
    return OwnerService(
        SqlUnitOfWorkFactory(session_factory),
        booked_events=DemoBookedEvents(session_factory),
        now=lambda: NOW,
    )


async def count(
    session_factory: async_sessionmaker[AsyncSession], model: Any, business_id: object
) -> int:
    async with session_factory() as session:
        value = await session.scalar(
            select(func.count()).select_from(model).where(model.business_id == business_id)
        )
    return int(value or 0)


def test_sandboxes_need_demo_mode_and_a_template(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(SandboxError):
        DemoSandboxes(settings(mode=False), session_factory)
    with pytest.raises(SandboxError):
        DemoSandboxes(settings(sandbox_template_slug=""), session_factory)


@pytest.mark.asyncio
async def test_each_visitor_gets_a_seeded_copy_of_the_template(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    original = await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)

    first = await sandboxes.create(NOW)
    second = await sandboxes.create(NOW)

    assert first.business_id != second.business_id
    assert len({first.public_key, second.public_key, TEMPLATE_KEY}) == 3
    async with session_factory() as session:
        copy = await session.get(Business, first.business_id)
        stored = await session.get(DemoSandbox, first.business_id)
    assert copy is not None and stored is not None
    assert copy.intake_questions == original.intake_questions
    assert copy.timezone == original.timezone
    assert copy.notification_email == original.notification_email
    assert copy.slug != SLUG
    assert first.sandbox_token not in stored.token_hash
    assert await count(session_factory, Customer, first.business_id) > 0
    assert await count(session_factory, QuoteRecord, first.business_id) > 0
    assert await count(session_factory, Customer, original.id) == 0


@pytest.mark.asyncio
async def test_the_demo_holds_at_most_the_live_cap(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await template(session_factory)
    sandboxes = DemoSandboxes(settings(sandbox_max_live=1), session_factory)
    await sandboxes.create(NOW)

    with pytest.raises(SandboxCapacityError):
        await sandboxes.create(NOW)
    # An idle sandbox no longer counts, even before the sweep deletes it.
    await sandboxes.create(NOW + timedelta(hours=3))


@pytest.mark.asyncio
async def test_sign_in_opens_only_the_visitors_own_dashboard(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    mine = await sandboxes.create(NOW)
    theirs = await sandboxes.create(NOW)
    owner = owner_service(session_factory)

    login = await sandboxes.sign_in(mine.sandbox_token, NOW)
    exchanged = await owner.exchange_login_token(login)

    assert exchanged is not None
    session_token, context = exchanged
    assert context.business.business_id == mine.business_id
    quotes = await owner.quotes(context)
    assert quotes
    assert {quote.business_id for quote in quotes} == {mine.business_id}
    assert (await owner.authenticate(session_token)).business.business_id != theirs.business_id
    with pytest.raises(OwnerAuthenticationError):
        await owner.exchange_login_token(login)


@pytest.mark.asyncio
async def test_a_wrong_or_expired_sandbox_token_signs_nobody_in(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    created = await sandboxes.create(NOW)

    for token in ("", "gvs_wrong", TEMPLATE_KEY, created.public_key):
        with pytest.raises(SandboxAuthenticationError):
            await sandboxes.sign_in(token, NOW)
    with pytest.raises(SandboxAuthenticationError):
        await sandboxes.sign_in(created.sandbox_token, NOW + timedelta(hours=2, minutes=1))


@pytest.mark.asyncio
async def test_the_sweep_deletes_idle_sandboxes_and_keeps_active_ones(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    original = await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    idle = await sandboxes.create(NOW)
    active = await sandboxes.create(NOW)
    await sandboxes.touch(active.business_id, NOW + timedelta(hours=1, minutes=30))
    async with session_factory() as session:
        for business_id in (idle.business_id, active.business_id):
            session.add(
                UsageLedgerMonth(
                    business_id=business_id,
                    kind="review_tokens",
                    month=NOW.date().replace(day=1),
                    units=120,
                    updated_at=NOW,
                )
            )
        await session.commit()

    deleted = await sandboxes.sweep(NOW + timedelta(hours=2, minutes=5))

    assert deleted == 1
    async with session_factory() as session:
        assert await session.get(Business, idle.business_id) is None
        assert await session.get(DemoSandbox, idle.business_id) is None
        assert await session.get(Business, active.business_id) is not None
        assert await session.get(Business, original.id) is not None
    assert await count(session_factory, Customer, idle.business_id) == 0
    assert await count(session_factory, QuoteRecord, idle.business_id) == 0
    assert await count(session_factory, IntakeMessage, idle.business_id) == 0
    assert await count(session_factory, UsageLedgerMonth, idle.business_id) == 0
    assert await count(session_factory, Customer, active.business_id) > 0
    assert await count(session_factory, UsageLedgerMonth, active.business_id) == 1
    assert await sandboxes.sweep(NOW + timedelta(hours=2, minutes=5)) == 0


@pytest.mark.asyncio
async def test_activity_is_written_at_most_once_a_minute(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    created = await sandboxes.create(NOW)

    await sandboxes.touch(created.business_id, NOW + timedelta(seconds=30))
    async with session_factory() as session:
        row = await session.get(DemoSandbox, created.business_id)
    assert row is not None
    assert row.last_active_at.replace(tzinfo=UTC) == NOW

    await sandboxes.touch(created.business_id, NOW + timedelta(minutes=5))
    async with session_factory() as session:
        row = await session.get(DemoSandbox, created.business_id)
    assert row is not None
    assert row.last_active_at.replace(tzinfo=UTC) == NOW + timedelta(minutes=5)


async def add_user_messages(
    session_factory: async_sessionmaker[AsyncSession], business_id: object, how_many: int
) -> None:
    async with session_factory() as session:
        conversation_id = uuid4()
        for index in range(how_many):
            session.add(
                IntakeMessage(
                    business_id=business_id,
                    conversation_id=conversation_id,
                    role=IntakeMessageRole.USER.value,
                    content=f"message {index}",
                    created_at=NOW + timedelta(seconds=index + 1),
                )
            )
        await session.commit()


@pytest.mark.asyncio
async def test_gus_stops_after_the_visitor_cap_and_the_daily_cap(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    original = await template(session_factory)
    sandboxes = DemoSandboxes(
        settings(sandbox_messages_per_visitor=3, sandbox_messages_per_day=5), session_factory
    )
    first = await sandboxes.create(NOW)
    second = await sandboxes.create(NOW)
    later = NOW + timedelta(minutes=5)

    # The budget is asked before the message being answered is committed:
    # with 2 stored, the 3rd still gets an answer; with 3 stored, the 4th doesn't.
    await add_user_messages(session_factory, first.business_id, 2)
    assert await sandboxes.refusal(first.business_id, later) is None
    await add_user_messages(session_factory, first.business_id, 1)
    assert await sandboxes.refusal(first.business_id, later) == VISITOR_LIMIT_REPLY

    await add_user_messages(session_factory, second.business_id, 1)
    assert await sandboxes.refusal(second.business_id, later) is None
    await add_user_messages(session_factory, second.business_id, 1)
    assert await sandboxes.refusal(second.business_id, later) == DAILY_LIMIT_REPLY
    # The template business itself is not a sandbox and has no visitor cap.
    assert await sandboxes.refusal(original.id, later) == DAILY_LIMIT_REPLY
    assert await sandboxes.refusal(second.business_id, later + timedelta(days=1)) is None


@pytest.mark.asyncio
async def test_every_sandbox_visitor_is_sam_rivera(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    original = await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    created = await sandboxes.create(NOW)

    assert await sandboxes.visitor(created.business_id) is SAM
    assert await sandboxes.visitor(original.id) is None
    assert SAM.collected.ready_for_slots is False
    assert SAM.collected.model_copy(update={"details": "A new patio"}).ready_for_slots


@pytest.mark.asyncio
async def test_payments_are_off_only_in_sandboxes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    original = await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    created = await sandboxes.create(NOW)

    assert await sandboxes.is_sandbox(created.business_id) is True
    assert await sandboxes.is_sandbox(original.id) is False


@pytest.mark.asyncio
async def test_one_address_opens_at_most_three_sandboxes_an_hour(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    app = FastAPI()
    app.include_router(create_sandbox_router(sandboxes, per_ip_per_hour=3))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://demo.test") as client:
        created = [await client.post("/v1/demo/sandboxes") for _ in range(4)]
        token = created[0].json()["sandboxToken"]
        signed_in = await client.post(
            "/v1/demo/sandboxes/sign-in", headers={"Authorization": f"Bearer {token}"}
        )
        refused = await client.post(
            "/v1/demo/sandboxes/sign-in", headers={"Authorization": "Bearer gvs_nope"}
        )

    assert [response.status_code for response in created] == [201, 201, 201, 429]
    body = created[0].json()
    assert set(body) == {"publicKey", "sandboxToken", "idleMinutes"}
    assert body["idleMinutes"] == 120
    assert signed_in.status_code == 200
    assert signed_in.json()["signInToken"]
    assert refused.status_code == 401


class _AssistantFake:
    def __init__(self) -> None:
        self.asked: list[tuple[object, str, tuple[tuple[str, str], ...]]] = []

    async def answer(
        self, business_id: object, text: str, history: tuple[tuple[str, str], ...] = ()
    ) -> str:
        self.asked.append((business_id, text, history))
        return "You have 2 bookings tomorrow."


@pytest.mark.asyncio
async def test_owner_texts_answer_only_for_the_visitors_own_sandbox(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await template(session_factory)
    sandboxes = DemoSandboxes(settings(), session_factory)
    assistant = _AssistantFake()
    app = FastAPI()
    app.include_router(
        create_sandbox_router(
            sandboxes,
            per_ip_per_hour=3,
            owner_assistant=assistant,  # type: ignore[arg-type]
            owner_texts_per_sandbox=2,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://demo.test") as client:
        token = (await client.post("/v1/demo/sandboxes")).json()["sandboxToken"]
        answered = await client.post(
            "/v1/demo/sandboxes/owner-texts",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "text": "what's on tomorrow?",
                "history": [
                    {"role": "owner", "text": "hello"},
                    {"role": "assistant", "text": "Hi!"},
                ],
            },
        )
        refused = await client.post(
            "/v1/demo/sandboxes/owner-texts",
            headers={"Authorization": "Bearer gvs_nope"},
            json={"text": "hello"},
        )
        second = await client.post(
            "/v1/demo/sandboxes/owner-texts",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "hello"},
        )
        used_up = await client.post(
            "/v1/demo/sandboxes/owner-texts",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "hello again"},
        )
        too_long = await client.post(
            "/v1/demo/sandboxes/owner-texts",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": "x" * 501},
        )

    assert answered.status_code == 200
    assert answered.json() == {"reply": "You have 2 bookings tomorrow."}
    async with session_factory() as session:
        sandbox_business = await session.scalar(select(DemoSandbox.business_id))
    assert [(business, text) for business, text, _ in assistant.asked] == [
        (sandbox_business, "what's on tomorrow?"),
        (sandbox_business, "hello"),
    ]
    assert second.json()["reply"] == "You have 2 bookings tomorrow."
    assert used_up.json() == {"reply": OWNER_TEXTS_USED_UP_REPLY}
    assert assistant.asked[0][2] == (("owner", "hello"), ("assistant", "Hi!"))
    assert refused.status_code == 401
    assert too_long.status_code == 422
