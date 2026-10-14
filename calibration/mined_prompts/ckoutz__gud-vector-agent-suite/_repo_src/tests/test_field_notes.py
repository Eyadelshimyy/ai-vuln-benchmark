from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.field_note_transcription import (
    FieldNoteTranscriptService,
    TranscribeFieldNoteAudioService,
    TranscriptionOutcome,
)
from gvas.application.field_notes import FieldNoteIntakeHandler, FieldNoteIntentContribution
from gvas.application.ingestion import IngestOwnerMessageService
from gvas.domain.enums import MediaKind, WorkflowRunStatus
from gvas.domain.field_note_repositories import (
    AmbiguousFieldNoteMessageError,
    CrossBusinessFieldNoteError,
    FieldNotePartDraft,
)
from gvas.domain.field_notes import (
    FIELD_NOTE_INTENT,
    FieldNoteCase,
    FieldNoteCaseId,
    FieldNoteCaseNotFoundError,
    FieldNoteCaseStatus,
    FieldNotePart,
    FieldNotePartId,
    FieldNotePartKind,
    TranscriptionStatus,
    build_canonical_transcript,
    field_note_transcribe_command,
    match_field_note_trigger,
)
from gvas.domain.identifiers import BusinessId, ConversationId, MessageId, MessageKey
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentReference,
    AudioReference,
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    TextPart,
    TranscriptResult,
)
from gvas.domain.outbox import RESERVED_COMMAND_TYPES
from gvas.domain.workflows import WorkflowContext
from gvas.infrastructure.field_note_models import FieldNoteCase as FieldNoteCaseRow
from gvas.infrastructure.field_note_models import (
    FieldNoteConversationState,
    FieldNotePartRow,
)
from gvas.infrastructure.field_note_repositories import SqlFieldNoteUnitOfWorkFactory
from gvas.infrastructure.models import Business
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory


def make_message(
    business_id: BusinessId,
    *,
    message_key: str,
    conversation: str = "conversation",
    endpoint: str = "endpoint",
    parts: tuple[object, ...] = (TextPart(text="field notes: hello"),),
) -> InboundOwnerMessage:
    normalized = NormalizedOwnerMessage(
        message_key=MessageKey(message_key),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id=conversation
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime(2025, 1, 1, tzinfo=UTC),
        parts=parts,
    )
    return InboundOwnerMessage(
        message=normalized,
        endpoint=ChannelEndpointRef(
            business_id=business_id,
            source_namespace="test",
            external_endpoint_id=endpoint,
        ),
        routing={},
    )


async def seed_business(
    session_factory: "async_sessionmaker[AsyncSession]", business_id: BusinessId
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


@pytest.mark.asyncio
async def test_trigger_matching() -> None:
    business_id = BusinessId(uuid4())
    audio = AttachmentPart(
        attachment=AttachmentReference(
            attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
        )
    )
    message = make_message(
        business_id,
        message_key="trigger",
        parts=(TextPart(text="  FIELD NOTES:  "), audio),
    )
    match = match_field_note_trigger(message.message)
    assert match is not None
    assert match.parts == (audio,)
    audio_only_match = match_field_note_trigger(
        make_message(business_id, message_key="audio-only", parts=(audio,)).message
    )
    assert audio_only_match is not None
    assert not audio_only_match.is_new_case
    assert audio_only_match.parts == (audio,)
    assert (
        match_field_note_trigger(
            make_message(business_id, message_key="none", parts=(TextPart(text="notes:"),)).message
        )
        is None
    )
    assert (
        match_field_note_trigger(
            make_message(
                business_id,
                message_key="later",
                parts=(audio, TextPart(text="field notes: later")),
            ).message
        )
        is None
    )


@pytest.mark.asyncio
async def test_text_intake_and_audio_command(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    message = make_message(business_id, message_key="intake")
    accepted = await ingest.ingest(message)
    assert accepted.message_id is not None
    handler = FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory),
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
    )
    result = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message.message)
    )
    assert result.status is WorkflowRunStatus.SUCCEEDED
    assert len(result.replies) == 1
    assert result.commands == ()
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(FieldNoteCaseRow)) == 1
        assert await session.scalar(select(func.count()).select_from(FieldNotePartRow)) == 1
        assert (
            await session.scalar(select(func.count()).select_from(FieldNoteConversationState)) == 1
        )

    audio_message = make_message(
        business_id,
        message_key="audio",
        parts=(
            TextPart(text="field notes: voice"),
            AttachmentPart(
                attachment=AttachmentReference(
                    attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
                )
            ),
        ),
    )
    accepted = await ingest.ingest(audio_message)
    assert accepted.message_id is not None
    result = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=audio_message.message)
    )
    assert len(result.commands) == 1
    assert result.commands[0].command_type not in RESERVED_COMMAND_TYPES
    assert result.commands[0] == field_note_transcribe_command(
        business_id,
        FieldNotePartId(UUID(str(result.commands[0].payload["field_note_part_id"]))),
        FieldNoteCaseId(UUID(str(result.commands[0].payload["field_note_case_id"]))),
    )


@pytest.mark.asyncio
async def test_audio_only_message_extends_the_active_case(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    handler = FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory),
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
    )

    for key in ("audio-1", "audio-2"):
        message = make_message(
            business_id,
            message_key=key,
            parts=(
                AttachmentPart(
                    attachment=AttachmentReference(
                        attachment_id=uuid4(),
                        media_kind=MediaKind.AUDIO,
                        locator=key,
                    )
                ),
            ),
        )
        assert (await ingest.ingest(message)).message_id is not None
        result = await handler.handle(
            WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message.message)
        )
        assert result.status is WorkflowRunStatus.SUCCEEDED

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(FieldNoteCaseRow)) == 1
        assert await session.scalar(select(func.count()).select_from(FieldNotePartRow)) == 2


@pytest.mark.asyncio
async def test_reply_append_duplicate_and_canonical_output(
    session_factory: "async_sessionmaker[AsyncSession]",
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    first = make_message(business_id, message_key="first")
    await ingest.ingest(first)
    handler = FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory),
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
    )
    first_result = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=first.message)
    )
    replay = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=first.message)
    )
    assert replay.replies[0].correlation_id == first_result.replies[0].correlation_id

    second = make_message(business_id, message_key="second", parts=(TextPart(text="follow up"),))
    await ingest.ingest(second)
    contribution = FieldNoteIntentContribution(SqlFieldNoteUnitOfWorkFactory(session_factory))
    assert await contribution.contribute(second.message) == FIELD_NOTE_INTENT
    second_result = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=second.message)
    )
    assert second_result.status is WorkflowRunStatus.SUCCEEDED
    async with SqlFieldNoteUnitOfWorkFactory(session_factory)() as unit_of_work:
        case_id = await unit_of_work.field_note_conversation_states.get_active_case_id(
            business_id, ConversationId(uuid4())
        )
        await unit_of_work.rollback()
        assert case_id is None
    async with session_factory() as session:
        rows = list((await session.scalars(select(FieldNoteCaseRow))).all())
        parts = list(
            (
                await session.scalars(select(FieldNotePartRow).order_by(FieldNotePartRow.sequence))
            ).all()
        )
        assert len(rows) == 1
        assert [row.sequence for row in parts] == [1, 2]
        domain_case = FieldNoteCase(
            case_id=FieldNoteCaseId(rows[0].id),
            business_id=business_id,
            conversation_ref=first.message.conversation_ref,
            origin_inbound_message_id=MessageId(rows[0].origin_inbound_message_id),
            status=FieldNoteCaseStatus.OPEN,
            parts=tuple(
                FieldNotePart(
                    part_id=FieldNotePartId(row.id),
                    case_id=FieldNoteCaseId(row.case_id),
                    business_id=business_id,
                    sequence=row.sequence,
                    kind=FieldNotePartKind(row.kind),
                    text=row.text,
                    attachment=None,
                    transcription_status=TranscriptionStatus.NOT_REQUIRED,
                )
                for row in parts
            ),
        )
        canonical = build_canonical_transcript(domain_case)
        assert canonical.text == "hello\n\nfollow up"
        assert canonical.is_complete


@pytest.mark.asyncio
async def test_missing_message_and_ambiguous_endpoint_identity(
    session_factory: "async_sessionmaker[AsyncSession]",
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    handler = FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory),
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
    )
    missing = make_message(business_id, message_key="missing")
    missing_result = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=missing.message)
    )
    assert missing_result.status is WorkflowRunStatus.FAILED
    assert missing_result.detail == "field note message is not persisted"

    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    first = make_message(business_id, message_key="ambiguous", endpoint="one")
    second = make_message(business_id, message_key="ambiguous", endpoint="two")
    await ingest.ingest(first)
    await ingest.ingest(second)
    async with SqlFieldNoteUnitOfWorkFactory(session_factory)() as unit_of_work:
        with pytest.raises(AmbiguousFieldNoteMessageError):
            await unit_of_work.field_note_messages.locate(
                business_id,
                first.message.conversation_ref,
                first.message.message_key,
            )
        await unit_of_work.rollback()
    ambiguous_result = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=first.message)
    )
    assert ambiguous_result.status is WorkflowRunStatus.FAILED
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(FieldNoteCaseRow)) == 0


@pytest.mark.asyncio
async def test_unsupported_media_is_recorded_and_counted(
    session_factory: "async_sessionmaker[AsyncSession]",
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    message = make_message(
        business_id,
        message_key="unsupported",
        parts=(
            TextPart(text="field notes:"),
            AttachmentPart(
                attachment=AttachmentReference(
                    attachment_id=uuid4(), media_kind=MediaKind.IMAGE, locator="opaque-image"
                )
            ),
            AttachmentPart(
                attachment=AttachmentReference(
                    attachment_id=uuid4(), media_kind=MediaKind.DOCUMENT, locator="opaque-document"
                )
            ),
        ),
    )
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    await ingest.ingest(message)
    result = await FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory),
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
    ).handle(WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message.message))
    assert result.commands == ()
    assert result.replies
    assert isinstance(result.replies[0].parts[0], TextPart)
    assert "Skipped 2 unsupported attachment(s)." in result.replies[0].parts[0].text
    async with session_factory() as session:
        case_row = await session.scalar(select(FieldNoteCaseRow))
        assert case_row is not None
        parts = list((await session.scalars(select(FieldNotePartRow))).all())
        assert [part.kind for part in parts] == ["unsupported", "unsupported"]
        case_id = FieldNoteCaseId(case_row.id)
    transcript = await FieldNoteTranscriptService(
        SqlFieldNoteUnitOfWorkFactory(session_factory)
    ).canonical_transcript(business_id, case_id)
    assert transcript.segments == ()
    assert transcript.unsupported_parts == 2


@pytest.mark.asyncio
async def test_duplicate_audio_delivery_returns_pending_ids_and_same_commands(
    session_factory: "async_sessionmaker[AsyncSession]",
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    audio = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-audio"
    )
    message = make_message(
        business_id,
        message_key="duplicate-audio",
        parts=(TextPart(text="field notes:"), AttachmentPart(attachment=audio)),
    )
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    await ingest.ingest(message)
    factory = SqlFieldNoteUnitOfWorkFactory(session_factory)
    handler = FieldNoteIntakeHandler(factory, now=lambda: datetime(2025, 1, 1, tzinfo=UTC))
    first = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message.message)
    )
    async with session_factory() as session:
        before = await session.scalar(select(func.count()).select_from(FieldNotePartRow))
    async with factory() as unit_of_work:
        location = await unit_of_work.field_note_messages.locate(
            business_id, message.message.conversation_ref, message.message.message_key
        )
        assert location is not None
        case_id = await unit_of_work.field_note_conversation_states.get_active_case_id(
            business_id, location.conversation_id
        )
        assert case_id is not None
        replay = await unit_of_work.field_note_cases.record_intake(
            location=location,
            parts=(FieldNotePartDraft(kind=FieldNotePartKind.AUDIO, attachment=audio),),
            case_id=case_id,
        )
        await unit_of_work.commit()
    assert replay.created_part_ids == ()
    assert len(replay.audio_part_ids) == 1
    second = await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message.message)
    )
    async with session_factory() as session:
        after = await session.scalar(select(func.count()).select_from(FieldNotePartRow))
    assert before == after == 1
    assert first.commands == second.commands
    assert str(replay.audio_part_ids[0]) == first.commands[0].payload["field_note_part_id"]


@pytest.mark.asyncio
async def test_tenant_isolation_applies_to_cases_state_and_locator(
    session_factory: "async_sessionmaker[AsyncSession]",
) -> None:
    first_business = BusinessId(uuid4())
    second_business = BusinessId(uuid4())
    await seed_business(session_factory, first_business)
    await seed_business(session_factory, second_business)
    first = make_message(first_business, message_key="same", conversation="same-conversation")
    second = make_message(second_business, message_key="same", conversation="same-conversation")
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    await ingest.ingest(first)
    await ingest.ingest(second)
    handler = FieldNoteIntakeHandler(
        SqlFieldNoteUnitOfWorkFactory(session_factory),
        now=lambda: datetime(2025, 1, 1, tzinfo=UTC),
    )
    await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=first.message)
    )
    await handler.handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=second.message)
    )
    async with session_factory() as session:
        cases = list((await session.scalars(select(FieldNoteCaseRow))).all())
        states = list((await session.scalars(select(FieldNoteConversationState))).all())
        assert len(cases) == len(states) == 2
        case_id = FieldNoteCaseId(
            next(case.id for case in cases if case.business_id == first_business)
        )
    with pytest.raises(FieldNoteCaseNotFoundError):
        await FieldNoteTranscriptService(
            SqlFieldNoteUnitOfWorkFactory(session_factory)
        ).canonical_transcript(second_business, case_id)
    async with SqlFieldNoteUnitOfWorkFactory(session_factory)() as unit_of_work:
        with pytest.raises(CrossBusinessFieldNoteError):
            await unit_of_work.field_note_messages.locate(
                first_business,
                ConversationRef(
                    business_id=second_business, external_conversation_id="same-conversation"
                ),
                first.message.message_key,
            )
        await unit_of_work.rollback()


@pytest.mark.asyncio
async def test_mixed_parts_keep_order_through_partial_and_full_transcription(
    session_factory: "async_sessionmaker[AsyncSession]",
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    first_audio = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-first-audio"
    )
    second_audio = AttachmentReference(
        attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator="opaque-second-audio"
    )
    message = make_message(
        business_id,
        message_key="mixed",
        parts=(
            TextPart(text="field notes: first text"),
            AttachmentPart(attachment=first_audio),
            TextPart(text="second text"),
            AttachmentPart(attachment=second_audio),
        ),
    )
    ingest = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    await ingest.ingest(message)
    factory = SqlFieldNoteUnitOfWorkFactory(session_factory)
    await FieldNoteIntakeHandler(factory, now=lambda: datetime(2025, 1, 1, tzinfo=UTC)).handle(
        WorkflowContext(run_id=uuid4(), intent=FIELD_NOTE_INTENT, message=message.message)
    )
    async with session_factory() as session:
        case_row = await session.scalar(select(FieldNoteCaseRow))
        assert case_row is not None
        rows = list(
            (
                await session.scalars(select(FieldNotePartRow).order_by(FieldNotePartRow.sequence))
            ).all()
        )
        assert [row.sequence for row in rows] == [1, 2, 3, 4]
        case_id = FieldNoteCaseId(case_row.id)
        first_audio_id = FieldNotePartId(rows[1].id)
        second_audio_id = FieldNotePartId(rows[3].id)

    class FailingPort:
        async def transcribe(self, audio: AudioReference) -> TranscriptResult:
            raise RuntimeError("mixed failure")

    failed = await TranscribeFieldNoteAudioService(factory, FailingPort()).transcribe(
        business_id,
        first_audio_id,
        now=datetime(2025, 1, 1, tzinfo=UTC),
        stale_before=datetime(2024, 12, 1, tzinfo=UTC),
    )
    assert failed.outcome is TranscriptionOutcome.FAILED
    partial = await FieldNoteTranscriptService(factory).canonical_transcript(business_id, case_id)
    assert partial.pending_parts == 1
    assert partial.failed_parts == 1
    assert partial.is_complete is False
    assert partial.text == "first text\n\nsecond text"

    class SuccessfulPort:
        async def transcribe(self, audio: AudioReference) -> TranscriptResult:
            text = (
                "first audio"
                if audio.attachment.attachment_id == first_audio.attachment_id
                else "second audio"
            )
            return TranscriptResult(text=text)

    service = TranscribeFieldNoteAudioService(factory, SuccessfulPort())
    assert (
        await service.transcribe(
            business_id,
            first_audio_id,
            now=datetime(2025, 1, 2, tzinfo=UTC),
            stale_before=datetime(2025, 1, 1, tzinfo=UTC),
        )
    ).outcome is TranscriptionOutcome.TRANSCRIBED
    assert (
        await service.transcribe(
            business_id,
            second_audio_id,
            now=datetime(2025, 1, 2, tzinfo=UTC),
            stale_before=datetime(2025, 1, 1, tzinfo=UTC),
        )
    ).outcome is TranscriptionOutcome.TRANSCRIBED
    complete = await FieldNoteTranscriptService(factory).canonical_transcript(business_id, case_id)
    assert complete.is_complete is True
    assert complete.pending_parts == complete.failed_parts == 0
    assert complete.text == "first text\n\nfirst audio\n\nsecond text\n\nsecond audio"
    assert complete == await FieldNoteTranscriptService(factory).canonical_transcript(
        business_id, case_id
    )
