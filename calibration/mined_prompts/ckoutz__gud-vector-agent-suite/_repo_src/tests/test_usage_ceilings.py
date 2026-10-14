import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from composition_fakes import (
    CustomerDeliveryFake,
    OwnerReplyFake,
    QuoteDraftingFake,
    ReportGenerationFake,
    TranscriptionFake,
    application_ports,
)
from gvas.composition import Application, build_application
from gvas.composition.failure_notices import CEILING_GUIDANCE
from gvas.config import CostCeilingSettings, OpenAISettings
from gvas.domain.completeness import CompletenessReviewRequest
from gvas.domain.enums import MediaKind, OutboxStatus
from gvas.domain.field_notes import (
    FIELD_NOTE_REVIEW_COMMAND_TYPE,
    FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE,
    FieldNoteCaseStatus,
    TranscriptionStatus,
)
from gvas.domain.identifiers import BusinessId
from gvas.domain.messages import (
    AttachmentPayload,
    AttachmentReference,
    AudioReference,
    TextPart,
)
from gvas.domain.usage import UsageCeilingGuard, UsageCeilings, UsageKind, usage_month
from gvas.infrastructure.completeness_models import FieldNoteFollowUpQuestion
from gvas.infrastructure.field_note_models import FieldNoteCase as FieldNoteCaseRow
from gvas.infrastructure.field_note_models import FieldNotePartRow
from gvas.infrastructure.models import OutboxMessage
from gvas.infrastructure.openai_contradiction_guard import OpenAIContradictionGuard
from gvas.infrastructure.openai_transcription import OpenAITranscriber, TranscriptionError
from gvas.infrastructure.usage_ledger import SqlUsageLedger
from test_completeness import checklist
from test_composition import (
    Clock,
    audio_part,
    configure_checklist,
    drain,
    inbound,
    outbox_rows,
    seed_business,
)

MARCH = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
APRIL = datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
TRANSCRIPTION = UsageKind.TRANSCRIPTION_AUDIO_SECONDS
REVIEW = UsageKind.REVIEW_TOKENS


class MemoryLedger:
    def __init__(self) -> None:
        self.records: list[tuple[BusinessId, UsageKind, int]] = []

    async def record(
        self, business_id: BusinessId, kind: UsageKind, units: int, *, at: datetime
    ) -> None:
        self.records.append((business_id, kind, units))

    async def total(self, business_id: BusinessId, kind: UsageKind, *, month: object) -> int:
        return sum(units for b, k, units in self.records if b == business_id and k == kind)


def test_usage_month_is_the_utc_calendar_month() -> None:
    late_evening_west = datetime(2026, 3, 31, 20, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert usage_month(MARCH).isoformat() == "2026-03-01"
    assert usage_month(late_evening_west) == usage_month(APRIL)
    with pytest.raises(ValueError):
        usage_month(datetime(2026, 3, 1))


def test_ceiling_settings_default_to_unlimited() -> None:
    settings = CostCeilingSettings(_env_file=None)
    assert settings.transcription_seconds == 0
    assert settings.review_tokens == 0
    assert UsageCeilings().limit(TRANSCRIPTION) == 0
    assert UsageCeilings().limit(REVIEW) == 0
    with pytest.raises(ValueError):
        UsageCeilings(review_tokens=-1)


@pytest.mark.asyncio
async def test_unlimited_default_never_reaches_a_ceiling() -> None:
    ledger = MemoryLedger()
    business_id = BusinessId(uuid4())
    await ledger.record(business_id, TRANSCRIPTION, 10**9, at=MARCH)
    await ledger.record(business_id, REVIEW, 10**9, at=MARCH)
    guard = UsageCeilingGuard(ledger)
    assert not await guard.is_reached(business_id, TRANSCRIPTION, now=MARCH)
    assert not await guard.is_reached(business_id, REVIEW, now=MARCH)
    assert not await UsageCeilingGuard().is_reached(business_id, REVIEW, now=MARCH)


@pytest.mark.asyncio
async def test_sql_ledger_accumulates_per_business_kind_and_month(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ledger = SqlUsageLedger(session_factory)
    first = BusinessId(uuid4())
    second = BusinessId(uuid4())

    await ledger.record(first, TRANSCRIPTION, 30, at=MARCH)
    await ledger.record(first, TRANSCRIPTION, 45, at=MARCH + timedelta(days=1))
    await ledger.record(first, TRANSCRIPTION, 0, at=MARCH)
    await ledger.record(first, REVIEW, 1200, at=MARCH)
    await ledger.record(first, TRANSCRIPTION, 5, at=APRIL)
    await ledger.record(second, TRANSCRIPTION, 7, at=MARCH)

    march = usage_month(MARCH)
    assert await ledger.total(first, TRANSCRIPTION, month=march) == 75
    assert await ledger.total(first, REVIEW, month=march) == 1200
    assert await ledger.total(first, TRANSCRIPTION, month=usage_month(APRIL)) == 5
    assert await ledger.total(second, TRANSCRIPTION, month=march) == 7
    assert await ledger.total(second, REVIEW, month=march) == 0
    with pytest.raises(ValueError):
        await ledger.record(first, REVIEW, -1, at=MARCH)


@pytest.mark.asyncio
async def test_sql_ledger_ceiling_is_reached_at_the_limit_and_resets_next_month(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ledger = SqlUsageLedger(session_factory)
    business_id = BusinessId(uuid4())
    guard = UsageCeilingGuard(ledger, UsageCeilings(transcription_seconds=60, review_tokens=100))

    await ledger.record(business_id, TRANSCRIPTION, 59, at=MARCH)
    assert not await guard.is_reached(business_id, TRANSCRIPTION, now=MARCH)
    await ledger.record(business_id, TRANSCRIPTION, 1, at=MARCH)
    assert await guard.is_reached(business_id, TRANSCRIPTION, now=MARCH)
    assert not await guard.is_reached(business_id, REVIEW, now=MARCH)
    assert not await guard.is_reached(business_id, TRANSCRIPTION, now=APRIL)
    assert not await guard.is_reached(BusinessId(uuid4()), TRANSCRIPTION, now=MARCH)


class StubAttachmentAccess:
    async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
        return AttachmentPayload(content=b"audio-bytes", mime_type="audio/mp4", filename="n.m4a")


@pytest.mark.asyncio
async def test_openai_transcriber_records_whole_audio_seconds_for_the_business() -> None:
    ledger = MemoryLedger()
    business_id = BusinessId(uuid4())
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "job walk complete", "duration": 12.3})

    audio = AudioReference(
        attachment=AttachmentReference(
            attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="channel-file:F1"
        ),
        business_id=business_id,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transcriber = OpenAITranscriber(
            OpenAISettings(api_key="sk-test"),
            client,
            StubAttachmentAccess(),
            usage_ledger=ledger,
        )
        result = await transcriber.transcribe(audio)

    assert result.text == "job walk complete"
    assert result.duration_seconds == 12.3
    assert b"verbose_json" in seen[0].read()
    assert ledger.records == [(business_id, TRANSCRIPTION, 13)]


@pytest.mark.asyncio
async def test_openai_transcriber_failure_records_nothing() -> None:
    ledger = MemoryLedger()

    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    audio = AudioReference(
        attachment=AttachmentReference(
            attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="channel-file:F1"
        ),
        business_id=BusinessId(uuid4()),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        transcriber = OpenAITranscriber(
            OpenAISettings(api_key="sk-test"), client, StubAttachmentAccess(), usage_ledger=ledger
        )
        with pytest.raises(TranscriptionError):
            await transcriber.transcribe(audio)

    assert ledger.records == []


@pytest.mark.asyncio
async def test_openai_guard_records_prompt_plus_completion_tokens() -> None:
    ledger = MemoryLedger()
    business_id = BusinessId(uuid4())

    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"contradictions": []}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 410, "completion_tokens": 15, "total_tokens": 425},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        guard = OpenAIContradictionGuard(
            OpenAISettings(api_key="sk-test"), client, usage_ledger=ledger
        )
        outcome = await guard.detect(
            CompletenessReviewRequest(
                business_id=business_id,
                checklist=checklist(business_id),
                transcript_text="site: north work: inspection",
                answers=(),
                round_index=0,
            )
        )

    assert outcome.is_clear
    assert ledger.records == [(business_id, REVIEW, 425)]


def build(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    owner_replies: OwnerReplyFake,
    transcription: TranscriptionFake,
    ceilings: UsageCeilings | None,
) -> Application:
    return build_application(
        application_ports(
            owner_replies=owner_replies,
            quote_drafting=QuoteDraftingFake(),
            quote_delivery=CustomerDeliveryFake(),
            transcription=transcription,
            report_generation=ReportGenerationFake(),
        ),
        session_factory=session_factory,
        now=Clock(),
        ceilings=ceilings,
    )


def owner_texts(owner_replies: OwnerReplyFake) -> list[str]:
    return [
        part.text
        for _, message in owner_replies.sent
        for part in message.parts
        if isinstance(part, TextPart)
    ]


async def open_case_count(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        rows = (await session.scalars(select(FieldNoteCaseRow))).all()
    return sum(1 for row in rows if row.status == FieldNoteCaseStatus.OPEN.value)


async def replay(application: Application, row: OutboxMessage) -> None:
    async with application.session_factory() as session:
        await session.execute(
            update(OutboxMessage)
            .where(OutboxMessage.id == row.id)
            .values(
                status=OutboxStatus.PENDING.value,
                locked_by=None,
                locked_at=None,
                available_at=datetime.now(UTC) - timedelta(minutes=1),
            )
        )
        await session.commit()
    await drain(application)


@pytest.mark.asyncio
async def test_transcription_ceiling_reached_skips_the_provider_and_tells_the_owner_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    transcription = TranscriptionFake({"audio-1": "site: north work: inspection"})
    application = build(
        session_factory,
        owner_replies=owner_replies,
        transcription=transcription,
        ceilings=UsageCeilings(transcription_seconds=600),
    )
    await configure_checklist(application, business_id)
    await application.usage_ledger.record(business_id, TRANSCRIPTION, 600, at=datetime.now(UTC))

    await application.ingest_service.ingest(
        inbound(business_id, None, message_key="audio-notes", attachments=(audio_part("audio-1"),))
    )
    await drain(application)

    assert transcription.calls == []
    commands = await outbox_rows(session_factory, FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE)
    assert [row.status for row in commands] == [OutboxStatus.SUCCEEDED.value]
    assert commands[0].attempts == 1
    async with session_factory() as session:
        part = await session.scalar(
            select(FieldNotePartRow).where(
                FieldNotePartRow.transcription_status != TranscriptionStatus.NOT_REQUIRED.value
            )
        )
    assert part is not None
    assert part.transcription_status == TranscriptionStatus.PENDING.value
    assert part.attempts == 0
    assert await open_case_count(session_factory) == 1
    notices = [
        text
        for text in owner_texts(owner_replies)
        if text == CEILING_GUIDANCE[FIELD_NOTE_TRANSCRIBE_COMMAND_TYPE]
    ]
    assert len(notices) == 1
    assert "$" not in notices[0]

    await replay(application, commands[0])
    assert transcription.calls == []
    assert len([t for t in owner_texts(owner_replies) if "transcription limit" in t]) == 1


@pytest.mark.asyncio
async def test_review_ceiling_reached_skips_the_review_and_tells_the_owner_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        transcription=TranscriptionFake({}),
        ceilings=UsageCeilings(review_tokens=50_000),
    )
    await configure_checklist(application, business_id)
    await application.usage_ledger.record(business_id, REVIEW, 50_000, at=datetime.now(UTC))

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north tower", message_key="notes-1")
    )
    await drain(application)

    commands = await outbox_rows(session_factory, FIELD_NOTE_REVIEW_COMMAND_TYPE)
    assert [row.status for row in commands] == [OutboxStatus.SUCCEEDED.value]
    async with session_factory() as session:
        questions = (await session.scalars(select(FieldNoteFollowUpQuestion))).all()
    assert questions == []
    assert await open_case_count(session_factory) == 1
    review_notices = [t for t in owner_texts(owner_replies) if "review limit" in t]
    assert review_notices == [CEILING_GUIDANCE[FIELD_NOTE_REVIEW_COMMAND_TYPE]]

    await replay(application, commands[0])
    assert len([t for t in owner_texts(owner_replies) if "review limit" in t]) == 1


@pytest.mark.asyncio
async def test_unlimited_default_transcribes_and_reviews_regardless_of_usage(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    transcription = TranscriptionFake({"audio-1": "site: north work: inspection"})
    application = build(
        session_factory, owner_replies=owner_replies, transcription=transcription, ceilings=None
    )
    await configure_checklist(application, business_id)
    now = datetime.now(UTC)
    await application.usage_ledger.record(business_id, TRANSCRIPTION, 10**9, at=now)
    await application.usage_ledger.record(business_id, REVIEW, 10**9, at=now)

    await application.ingest_service.ingest(
        inbound(business_id, None, message_key="audio-notes", attachments=(audio_part("audio-1"),))
    )
    await drain(application)

    assert transcription.calls == ["audio-1"]
    reviews = await outbox_rows(session_factory, FIELD_NOTE_REVIEW_COMMAND_TYPE)
    assert reviews and all(row.status == OutboxStatus.SUCCEEDED.value for row in reviews)
    async with session_factory() as session:
        questions = (await session.scalars(select(FieldNoteFollowUpQuestion))).all()
    assert questions == []
    assert all("limit" not in text for text in owner_texts(owner_replies))
