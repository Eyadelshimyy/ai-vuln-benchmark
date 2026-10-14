import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.field_note_transcription import (
    FieldNoteMediaHandoff,
    TranscribeFieldNoteAudioService,
    TranscriptionOutcome,
)
from gvas.application.field_notes import FieldNoteIntakeHandler
from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.enums import MediaKind
from gvas.domain.field_note_repositories import (
    FieldNoteUnitOfWork,
    LostTranscriptionLeaseError,
    TranscriptionClaim,
    TranscriptionClaimResult,
)
from gvas.domain.field_notes import (
    FIELD_NOTE_INTENT,
    FieldNoteCaseId,
    FieldNotePart,
    FieldNotePartId,
    FieldNotePartKind,
    TranscriptionStatus,
)
from gvas.domain.identifiers import BusinessId, MessageKey
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentPayload,
    AttachmentReference,
    AudioReference,
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    TextPart,
    TranscriptResult,
)
from gvas.domain.workflows import WorkflowContext
from gvas.infrastructure.field_note_models import FieldNotePartRow
from gvas.infrastructure.field_note_repositories import SqlFieldNoteUnitOfWorkFactory
from gvas.infrastructure.models import Business
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory


async def seed_business(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> None:
    now = datetime.now(UTC)
    async with session_factory() as session:
        session.add(
            Business(
                id=business_id,
                slug=f"business-{business_id}",
                name="Business",
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


async def create_audio_part(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    *,
    message_key: str,
) -> FieldNotePartId:
    attachment = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator=f"opaque-{message_key}"
    )
    message = NormalizedOwnerMessage(
        message_key=MessageKey(message_key),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id=f"conversation-{message_key}"
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(TextPart(text="field notes:"), AttachmentPart(attachment=attachment)),
    )
    await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        InboundOwnerMessage(
            message=message,
            endpoint=ChannelEndpointRef(
                business_id=business_id,
                source_namespace="test",
                external_endpoint_id=f"endpoint-{message_key}",
            ),
            routing={},
        )
    )
    await FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory), now=lambda: datetime.now(UTC)
    ).handle(WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message))
    async with session_factory() as session:
        row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.kind == FieldNotePartKind.AUDIO.value)
        )
        assert row is not None
        return FieldNotePartId(row.id)


@pytest.mark.asyncio
async def test_transcription_lifecycle_and_retry(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    attachment = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
    )
    message = NormalizedOwnerMessage(
        message_key=MessageKey("audio"),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id="conversation"
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(TextPart(text="field notes: voice"), AttachmentPart(attachment=attachment)),
    )
    envelope = InboundOwnerMessage(
        message=message,
        endpoint=ChannelEndpointRef(
            business_id=business_id, source_namespace="test", external_endpoint_id="endpoint"
        ),
        routing={},
    )
    accepted = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        envelope
    )
    assert accepted.message_id is not None
    await FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory),
        now=lambda: datetime.now(UTC),
    ).handle(WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message))
    async with session_factory() as session:
        row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.kind == FieldNotePartKind.AUDIO.value)
        )
        assert row is not None
        part_id = FieldNotePartId(row.id)

    class FailingPort:
        async def transcribe(self, audio: AudioReference) -> TranscriptResult:
            raise RuntimeError("failed")

    service = TranscribeFieldNoteAudioService(
        SqlFieldNoteUnitOfWorkFactory(session_factory), FailingPort()
    )
    failed = await service.transcribe(
        business_id,
        part_id,
        now=datetime.now(UTC),
        stale_before=datetime.now(UTC) - timedelta(hours=1),
    )
    assert failed.outcome is TranscriptionOutcome.FAILED
    async with session_factory() as session:
        failed_row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.id == part_id)
        )
        assert failed_row is not None
        assert failed_row.attempts == 1
        assert failed_row.last_error is not None

    class SuccessfulPort:
        calls = 0

        async def transcribe(self, audio: AudioReference) -> TranscriptResult:
            self.calls += 1
            return TranscriptResult(text="voice transcript")

    port = SuccessfulPort()
    service = TranscribeFieldNoteAudioService(SqlFieldNoteUnitOfWorkFactory(session_factory), port)
    succeeded = await service.transcribe(
        business_id,
        part_id,
        now=datetime.now(UTC),
        stale_before=datetime.now(UTC) - timedelta(hours=1),
    )
    assert succeeded.outcome is TranscriptionOutcome.TRANSCRIBED
    duplicate = await service.transcribe(
        business_id,
        part_id,
        now=datetime.now(UTC),
        stale_before=datetime.now(UTC) - timedelta(hours=1),
    )
    assert duplicate.outcome is TranscriptionOutcome.ALREADY_TRANSCRIBED
    assert port.calls == 1


@pytest.mark.asyncio
async def test_media_handoff_rejects_non_audio() -> None:
    class Port:
        async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
            return AttachmentPayload(content=b"")

    image = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.IMAGE, locator="opaque-image"
    )
    part = FieldNotePart(
        part_id=FieldNotePartId(uuid4()),
        case_id=FieldNoteCaseId(uuid4()),
        business_id=BusinessId(uuid4()),
        sequence=1,
        kind=FieldNotePartKind.UNSUPPORTED,
        attachment=image,
        transcription_status=TranscriptionStatus.NOT_REQUIRED,
    )
    with pytest.raises(ValueError) as error:
        await FieldNoteMediaHandoff(Port()).open_audio(part)
    assert "opaque-image" not in str(error.value)
    assert "attachment" in str(error.value)


@pytest.mark.asyncio
async def test_media_handoff_fetches_audio_through_attachment_port() -> None:
    audio = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
    )
    part = FieldNotePart(
        part_id=FieldNotePartId(uuid4()),
        case_id=FieldNoteCaseId(uuid4()),
        business_id=BusinessId(uuid4()),
        sequence=1,
        kind=FieldNotePartKind.AUDIO,
        attachment=audio,
        transcription_status=TranscriptionStatus.PENDING,
    )

    class Port:
        calls: list[AttachmentReference] = []

        async def fetch(self, attachment: AttachmentReference) -> AttachmentPayload:
            self.calls.append(attachment)
            return AttachmentPayload(content=b"audio", mime_type="audio/mpeg")

    port = Port()
    payload = await FieldNoteMediaHandoff(port).open_audio(part)
    assert payload.content == b"audio"
    assert port.calls == [audio]


@pytest.mark.asyncio
async def test_transcription_lease_states_and_missing_parts(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    audio_message = NormalizedOwnerMessage(
        message_key=MessageKey("lease-audio"),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id="lease-conversation"
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(
            TextPart(text="field notes: audio"),
            AttachmentPart(
                attachment=AttachmentReference(
                    attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
                )
            ),
        ),
    )
    envelope = InboundOwnerMessage(
        message=audio_message,
        endpoint=ChannelEndpointRef(
            business_id=business_id, source_namespace="test", external_endpoint_id="lease-endpoint"
        ),
        routing={},
    )
    await ingest.ingest(envelope)
    await FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory), now=lambda: datetime.now(UTC)
    ).handle(WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=audio_message))
    async with session_factory() as session:
        audio_row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.kind == FieldNotePartKind.AUDIO.value)
        )
        assert audio_row is not None
        part_id = FieldNotePartId(audio_row.id)
        initial = datetime(2025, 1, 1, tzinfo=UTC)

    factory = SqlFieldNoteUnitOfWorkFactory(session_factory)
    async with factory() as unit_of_work:
        first = await unit_of_work.field_note_transcriptions.claim(
            business_id, part_id, now=initial, stale_before=initial - timedelta(hours=1)
        )
        await unit_of_work.commit()
    async with factory() as unit_of_work:
        busy = await unit_of_work.field_note_transcriptions.claim(
            business_id,
            part_id,
            now=initial + timedelta(minutes=1),
            stale_before=initial - timedelta(hours=1),
        )
        await unit_of_work.commit()
    assert busy.result.value == "busy"
    async with factory() as unit_of_work:
        stale = await unit_of_work.field_note_transcriptions.claim(
            business_id,
            part_id,
            now=initial + timedelta(hours=2),
            stale_before=initial + timedelta(hours=1),
        )
        await unit_of_work.commit()
    assert stale.result.value == "acquired"
    async with factory() as unit_of_work:
        with pytest.raises(ValueError, match="no longer active"):
            await unit_of_work.field_note_transcriptions.record_success(
                first, TranscriptResult(text="superseded")
            )
        await unit_of_work.rollback()
    async with factory() as unit_of_work:
        with pytest.raises(ValueError, match="no longer active"):
            await unit_of_work.field_note_transcriptions.record_failure(first, "superseded")
        await unit_of_work.rollback()
    async with session_factory() as session:
        persisted = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.id == part_id)
        )
        assert persisted is not None
        assert persisted.transcription_status == TranscriptionStatus.IN_PROGRESS.value
        assert persisted.attempts == 2

    async with factory() as unit_of_work:
        missing = await unit_of_work.field_note_transcriptions.claim(
            business_id,
            FieldNotePartId(uuid4()),
            now=initial,
            stale_before=initial - timedelta(hours=1),
        )
        await unit_of_work.commit()
    assert missing.result.value == "missing"
    async with session_factory() as session:
        text_row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.kind == FieldNotePartKind.TEXT.value)
        )
        assert text_row is not None
        text_id = FieldNotePartId(text_row.id)
    async with factory() as unit_of_work:
        non_audio = await unit_of_work.field_note_transcriptions.claim(
            business_id, text_id, now=initial, stale_before=initial - timedelta(hours=1)
        )
        await unit_of_work.commit()
    assert non_audio.result.value == "missing"


@pytest.mark.asyncio
async def test_transcription_port_runs_without_an_open_uow(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    attachment = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
    )
    message = NormalizedOwnerMessage(
        message_key=MessageKey("discipline"),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id="discipline-conversation"
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(TextPart(text="field notes:"), AttachmentPart(attachment=attachment)),
    )
    await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        InboundOwnerMessage(
            message=message,
            endpoint=ChannelEndpointRef(
                business_id=business_id,
                source_namespace="test",
                external_endpoint_id="discipline-endpoint",
            ),
            routing={},
        )
    )
    await FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory), now=lambda: datetime.now(UTC)
    ).handle(WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message))
    async with session_factory() as session:
        row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.kind == FieldNotePartKind.AUDIO.value)
        )
        assert row is not None
        part_id = FieldNotePartId(row.id)

    active_uows = 0
    inner_factory = SqlFieldNoteUnitOfWorkFactory(session_factory)

    class TrackingUow:
        def __init__(self) -> None:
            self.inner = inner_factory()

        async def __aenter__(self) -> FieldNoteUnitOfWork:
            nonlocal active_uows
            active_uows += 1
            try:
                return await self.inner.__aenter__()
            except Exception:
                active_uows -= 1
                raise

        async def __aexit__(
            self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: object
        ) -> None:
            nonlocal active_uows
            try:
                await self.inner.__aexit__(exc_type, exc, traceback)
            finally:
                active_uows -= 1

    class TrackingFactory:
        def __call__(self) -> FieldNoteUnitOfWork:
            return cast(FieldNoteUnitOfWork, TrackingUow())

    class Port:
        async def transcribe(self, audio: AudioReference) -> TranscriptResult:
            # The counter proves the provider port is called after claim UoW closure.
            assert active_uows == 0
            return TranscriptResult(text="discipline transcript")

    report = await TranscribeFieldNoteAudioService(TrackingFactory(), Port()).transcribe(
        business_id,
        part_id,
        now=datetime.now(UTC),
        stale_before=datetime.now(UTC) - timedelta(hours=1),
    )
    assert report.outcome is TranscriptionOutcome.TRANSCRIBED


@pytest.mark.asyncio
async def test_superseded_provider_results_report_lease_lost(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    message = NormalizedOwnerMessage(
        message_key=MessageKey("superseded"),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id="superseded-conversation"
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(
            TextPart(text="field notes:"),
            AttachmentPart(
                attachment=AttachmentReference(
                    attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
                )
            ),
        ),
    )
    await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        InboundOwnerMessage(
            message=message,
            endpoint=ChannelEndpointRef(
                business_id=business_id,
                source_namespace="test",
                external_endpoint_id="superseded-endpoint",
            ),
            routing={},
        )
    )
    factory = SqlFieldNoteUnitOfWorkFactory(session_factory)
    await FieldNoteIntakeHandler(factory, now=lambda: datetime(2025, 1, 1, tzinfo=UTC)).handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message)
    )
    async with session_factory() as session:
        row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.kind == FieldNotePartKind.AUDIO.value)
        )
        assert row is not None
        part_id = FieldNotePartId(row.id)

    class SupersedingPort:
        should_fail = False
        replacement_time = datetime(2025, 1, 3, tzinfo=UTC)

        async def transcribe(self, audio: AudioReference) -> TranscriptResult:
            async with factory() as unit_of_work:
                replacement = await unit_of_work.field_note_transcriptions.claim(
                    business_id,
                    part_id,
                    now=self.replacement_time,
                    stale_before=self.replacement_time - timedelta(days=1) + timedelta(minutes=1),
                )
                assert replacement.result.value == "acquired"
                await unit_of_work.commit()
            if self.should_fail:
                raise RuntimeError("superseded failure")
            return TranscriptResult(text="superseded transcript")

    port = SupersedingPort()
    service = TranscribeFieldNoteAudioService(factory, port)
    success_result = await service.transcribe(
        business_id,
        part_id,
        now=datetime(2025, 1, 1, tzinfo=UTC),
        stale_before=datetime(2024, 12, 1, tzinfo=UTC),
    )
    assert success_result.outcome is TranscriptionOutcome.LEASE_LOST
    port.should_fail = True
    port.replacement_time = datetime(2025, 1, 5, tzinfo=UTC)
    failure_result = await service.transcribe(
        business_id,
        part_id,
        now=datetime(2025, 1, 4, tzinfo=UTC),
        stale_before=datetime(2025, 1, 3, 0, 1, tzinfo=UTC),
    )
    assert failure_result.outcome is TranscriptionOutcome.LEASE_LOST


@pytest.mark.asyncio
async def test_cross_business_transcribe_is_missing_and_does_not_call_port(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_a = BusinessId(uuid4())
    business_b = BusinessId(uuid4())
    await seed_business(session_factory, business_a)
    await seed_business(session_factory, business_b)
    part_id = await create_audio_part(session_factory, business_a, message_key="cross-service")

    class Port:
        calls = 0

        async def transcribe(self, audio: AudioReference) -> TranscriptResult:
            self.calls += 1
            return TranscriptResult(text="must not run")

    port = Port()
    result = await TranscribeFieldNoteAudioService(
        SqlFieldNoteUnitOfWorkFactory(session_factory), port
    ).transcribe(
        business_b,
        part_id,
        now=datetime(2025, 1, 1, tzinfo=UTC),
        stale_before=datetime(2024, 12, 1, tzinfo=UTC),
    )
    assert result.outcome is TranscriptionOutcome.MISSING
    assert port.calls == 0
    async with session_factory() as session:
        row = await session.scalar(select(FieldNotePartRow).where(FieldNotePartRow.id == part_id))
        assert row is not None
        assert row.transcription_status == TranscriptionStatus.PENDING.value
        assert row.attempts == 0
        assert row.last_error is None


@pytest.mark.asyncio
async def test_repository_claim_requires_business_match(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_a = BusinessId(uuid4())
    business_b = BusinessId(uuid4())
    await seed_business(session_factory, business_a)
    await seed_business(session_factory, business_b)
    part_id = await create_audio_part(session_factory, business_a, message_key="cross-repository")
    factory = SqlFieldNoteUnitOfWorkFactory(session_factory)
    now = datetime(2025, 1, 1, tzinfo=UTC)
    async with factory() as unit_of_work:
        wrong = await unit_of_work.field_note_transcriptions.claim(
            business_b, part_id, now=now, stale_before=now - timedelta(hours=1)
        )
        await unit_of_work.commit()
    assert wrong.result is TranscriptionClaimResult.MISSING
    async with factory() as unit_of_work:
        right = await unit_of_work.field_note_transcriptions.claim(
            business_a, part_id, now=now, stale_before=now - timedelta(hours=1)
        )
        await unit_of_work.commit()
    assert right.result is TranscriptionClaimResult.ACQUIRED


@pytest.mark.asyncio
async def test_fenced_write_rejects_claim_with_wrong_business(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_a = BusinessId(uuid4())
    business_b = BusinessId(uuid4())
    await seed_business(session_factory, business_a)
    await seed_business(session_factory, business_b)
    part_id = await create_audio_part(session_factory, business_a, message_key="cross-fence")
    factory = SqlFieldNoteUnitOfWorkFactory(session_factory)
    now = datetime(2025, 1, 1, tzinfo=UTC)
    async with factory() as unit_of_work:
        claim = await unit_of_work.field_note_transcriptions.claim(
            business_a, part_id, now=now, stale_before=now - timedelta(hours=1)
        )
        await unit_of_work.commit()
    forged = claim.model_copy(update={"business_id": business_b})
    async with factory() as unit_of_work:
        with pytest.raises(LostTranscriptionLeaseError):
            await unit_of_work.field_note_transcriptions.record_success(
                forged, TranscriptResult(text="must not persist")
            )
        await unit_of_work.rollback()
    async with session_factory() as session:
        row = await session.scalar(select(FieldNotePartRow).where(FieldNotePartRow.id == part_id))
        assert row is not None
        assert row.transcription_status == TranscriptionStatus.IN_PROGRESS.value
        assert row.attempts == 1
        assert row.transcript_text is None


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_field_note_claim_is_exclusive(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(postgres_session_factory, business_id)
    attachment = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
    )
    message = NormalizedOwnerMessage(
        message_key=MessageKey("postgres-claim"),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id="postgres-conversation"
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(TextPart(text="field notes:"), AttachmentPart(attachment=attachment)),
    )
    await IngestOwnerMessageService(SqlUnitOfWorkFactory(postgres_session_factory)).ingest(
        InboundOwnerMessage(
            message=message,
            endpoint=ChannelEndpointRef(
                business_id=business_id,
                source_namespace="test",
                external_endpoint_id="postgres-endpoint",
            ),
            routing={},
        )
    )
    await FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(postgres_session_factory), now=lambda: datetime.now(UTC)
    ).handle(WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message))
    async with postgres_session_factory() as session:
        row = await session.scalar(
            select(FieldNotePartRow).where(FieldNotePartRow.kind == FieldNotePartKind.AUDIO.value)
        )
        assert row is not None
        part_id = FieldNotePartId(row.id)

    now = datetime(2025, 1, 1, tzinfo=UTC)

    async def claim() -> TranscriptionClaim:
        async with SqlFieldNoteUnitOfWorkFactory(postgres_session_factory)() as unit_of_work:
            result = await unit_of_work.field_note_transcriptions.claim(
                business_id, part_id, now=now, stale_before=now - timedelta(hours=1)
            )
            await unit_of_work.commit()
            return result

    results = await asyncio.gather(claim(), claim())
    assert sorted(result.result.value for result in results) == ["acquired", "busy"]
