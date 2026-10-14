"""SMS is scoped to the quote workflow; everything else gets one scope reply."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    CustomerDeliveryFake,
    OwnerReplyFake,
    QuoteDraftingFake,
    ReportGenerationFake,
    TranscriptionFake,
    application_ports,
)
from gvas.application.channel_policy import CHANNEL_UNSUPPORTED_INTENT_PREFIX
from gvas.composition import Application, build_application
from gvas.domain.enums import OutboxStatus
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import TextPart
from gvas.infrastructure.field_note_models import FieldNoteCase
from gvas.infrastructure.models import OutboxMessage, QuoteRecord, WorkflowRun
from gvas.infrastructure.telnyx.composition import (
    SMS_QUOTES_ONLY_REPLY_PREFIX,
    sms_quotes_only_policy,
)
from telnyx_fixtures import message_payload, normalize, seed_business

POLICY = sms_quotes_only_policy("the team chat")


class Clock:
    def __init__(self) -> None:
        self._now = datetime.now(UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


def build(
    session_factory: async_sessionmaker[AsyncSession],
    owner_replies: OwnerReplyFake,
    quote_drafting: QuoteDraftingFake,
) -> Application:
    ports = replace(
        application_ports(
            owner_replies=owner_replies,
            quote_drafting=quote_drafting,
            quote_delivery=CustomerDeliveryFake(),
            transcription=TranscriptionFake({}),
            report_generation=ReportGenerationFake(),
        ),
        channel_policies=(POLICY,),
    )
    return build_application(ports, session_factory=session_factory, now=Clock())


def reply_texts(owner_replies: OwnerReplyFake) -> list[str]:
    return [
        part.text
        for _, message in owner_replies.sent
        for part in message.parts
        if isinstance(part, TextPart)
    ]


async def count(session_factory: async_sessionmaker[AsyncSession], model: type) -> int:
    async with session_factory() as session:
        return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def unsucceeded_outbox(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        return int(
            await session.scalar(
                select(func.count())
                .select_from(OutboxMessage)
                .where(OutboxMessage.status != OutboxStatus.SUCCEEDED.value)
            )
            or 0
        )


@pytest.mark.asyncio
async def test_field_notes_over_sms_get_the_quotes_only_reply_and_open_no_case(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = build(session_factory, owner_replies, QuoteDraftingFake())

    await application.ingest_service.ingest(
        normalize(message_payload(text="field notes: replaced the anode rod"), business_id)
    )
    await application.worker.drain()
    await application.worker.drain()

    assert reply_texts(owner_replies) == [
        f"{SMS_QUOTES_ONLY_REPLY_PREFIX} Field notes belong in the team chat."
    ]
    assert "quote:" in reply_texts(owner_replies)[0]
    assert await count(session_factory, FieldNoteCase) == 0
    assert await unsucceeded_outbox(session_factory) == 0
    async with session_factory() as session:
        runs = list((await session.scalars(select(WorkflowRun))).all())
    assert [run.intent for run in runs] == [f"{CHANNEL_UNSUPPORTED_INTENT_PREFIX}telnyx"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["approve report", "close notes", "hello?"])
async def test_other_non_quote_triggers_over_sms_get_the_same_single_reply(
    session_factory: async_sessionmaker[AsyncSession], text: str
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = build(session_factory, owner_replies, QuoteDraftingFake())

    await application.ingest_service.ingest(normalize(message_payload(text=text), business_id))
    await application.worker.drain()

    assert len(reply_texts(owner_replies)) == 1
    assert reply_texts(owner_replies)[0].startswith(SMS_QUOTES_ONLY_REPLY_PREFIX)
    assert await count(session_factory, FieldNoteCase) == 0
    assert await unsucceeded_outbox(session_factory) == 0


@pytest.mark.asyncio
async def test_quote_trigger_and_approval_over_sms_run_the_quote_workflow(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    quote_drafting = QuoteDraftingFake()
    application = build(session_factory, owner_replies, quote_drafting)

    await application.ingest_service.ingest(
        normalize(message_payload(text="quote: replace two gutters"), business_id)
    )
    await application.worker.drain()
    await application.ingest_service.ingest(
        normalize(message_payload(id="approve-1", text="approve"), business_id)
    )
    await application.worker.drain()

    assert len(quote_drafting.requests) == 1
    assert await count(session_factory, QuoteRecord) == 1
    assert not any(
        text.startswith("SMS supports quotes only") for text in reply_texts(owner_replies)
    )
    assert await unsucceeded_outbox(session_factory) == 0
