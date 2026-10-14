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
from gvas.application.field_notes import CLOSED_CASE_REPLY, NO_OPEN_CASE_REPLY
from gvas.application.templates import IndustryTemplateDefinition
from gvas.application.unmatched_messages import UNMATCHED_MESSAGE_REPLY
from gvas.application.workflow_conflicts import (
    FIELD_NOTE_CONFLICT_REPLY,
    QUOTE_CONFLICT_REPLY,
)
from gvas.composition import Application, build_application
from gvas.composition.dispatcher import UnknownCommandTypeError
from gvas.composition.review import (
    CoordinateFieldNoteReviewService,
    ReviewCoordinationStatus,
)
from gvas.domain.completeness import (
    ChecklistItem,
    ChecklistItemKey,
    ChecklistKey,
    CompletenessChecklist,
)
from gvas.domain.enums import MediaKind, OutboxStatus
from gvas.domain.field_notes import (
    FieldNoteCaseId,
    FieldNoteCaseStatus,
    FieldNoteReviewTrigger,
    TranscriptionStatus,
)
from gvas.domain.identifiers import BusinessId, MessageKey, OutboxCommandId
from gvas.domain.messages import (
    AttachmentPart,
    AttachmentReference,
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    SenderRef,
    TextPart,
)
from gvas.domain.outbox import OutboxCommand, OutboxRecord
from gvas.domain.repositories import UnitOfWork
from gvas.domain.templates import IndustryKey, ReportTemplateSection, TemplateSetKey
from gvas.infrastructure.completeness_models import FieldNoteFollowUpQuestion
from gvas.infrastructure.field_note_models import FieldNoteCase as FieldNoteCaseRow
from gvas.infrastructure.field_note_models import FieldNotePartRow
from gvas.infrastructure.models import (
    Business,
    FieldNoteReport,
    FieldNoteReportVersion,
    OutboxMessage,
    QuoteRecord,
)

NOW = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
CHECKLIST_KEY = ChecklistKey("field_notes")


class Clock:
    """Monotonic clock anchored to wall time so enqueued commands are claimable."""

    def __init__(self) -> None:
        self._now = datetime.now(UTC)

    def __call__(self) -> datetime:
        self._now += timedelta(seconds=1)
        return self._now


def inbound(
    business_id: BusinessId,
    text: str | None,
    *,
    message_key: str,
    conversation: str = "conversation",
    attachments: tuple[AttachmentPart, ...] = (),
) -> InboundOwnerMessage:
    normalized = NormalizedOwnerMessage(
        message_key=MessageKey(message_key),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id=conversation
        ),
        sender=SenderRef(external_id="owner", role="owner"),
        received_at=NOW,
        parts=(() if text is None else (TextPart(text=text),)) + attachments,
    )
    return InboundOwnerMessage(
        message=normalized,
        endpoint=ChannelEndpointRef(
            business_id=business_id,
            source_namespace="test",
            external_endpoint_id="endpoint",
        ),
        routing={"thread_root": conversation},
    )


def audio_part(locator: str) -> AttachmentPart:
    return AttachmentPart(
        attachment=AttachmentReference(
            attachment_id=uuid4(), media_kind=MediaKind.AUDIO, locator=locator
        )
    )


async def seed_business(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> None:
    async with session_factory() as session:
        session.add(
            Business(
                id=business_id,
                slug=f"business-{business_id}",
                name="Business",
                created_at=NOW,
                updated_at=NOW,
            )
        )
        await session.commit()


def checklist(business_id: BusinessId) -> CompletenessChecklist:
    return CompletenessChecklist(
        business_id=business_id,
        checklist_key=CHECKLIST_KEY,
        version=1,
        items=(
            ChecklistItem(
                key=ChecklistItemKey("site"),
                prompt="Which site was visited?",
                evidence_markers=("site:",),
            ),
            ChecklistItem(
                key=ChecklistItemKey("work"),
                prompt="What work was performed?",
                evidence_markers=("work:",),
            ),
        ),
    )


async def configure_checklist(application: Application, business_id: BusinessId) -> None:
    definition = checklist(business_id)
    await application.template_publisher.seed_industry(
        business_id,
        IndustryTemplateDefinition(
            industry_key=IndustryKey("environmental_testing"),
            template_set_key=TemplateSetKey(CHECKLIST_KEY),
            checklist_key=CHECKLIST_KEY,
            version=definition.version,
            items=definition.items,
            report_template_key="field_notes_report",
            report_title="Field Notes Report",
            report_sections=(
                ReportTemplateSection(
                    section_key="site_and_work",
                    heading="Site and Work",
                    checklist_item_keys=tuple(item.key for item in definition.items),
                ),
            ),
        ),
    )


def build(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    owner_replies: OwnerReplyFake,
    quote_drafting: QuoteDraftingFake,
    quote_delivery: CustomerDeliveryFake,
    transcription: TranscriptionFake,
    report_generation: ReportGenerationFake,
) -> Application:
    return build_application(
        application_ports(
            owner_replies=owner_replies,
            quote_drafting=quote_drafting,
            quote_delivery=quote_delivery,
            transcription=transcription,
            report_generation=report_generation,
        ),
        session_factory=session_factory,
        now=Clock(),
    )


async def drain(application: Application) -> None:
    await application.worker.drain()


async def outbox_rows(
    session_factory: async_sessionmaker[AsyncSession], command_type: str
) -> list[OutboxMessage]:
    async with session_factory() as session:
        return list(
            (
                await session.scalars(
                    select(OutboxMessage).where(OutboxMessage.command_type == command_type)
                )
            ).all()
        )


@pytest.mark.asyncio
async def test_quote_path_delivers_once_under_replay(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    quote_drafting = QuoteDraftingFake()
    quote_delivery = CustomerDeliveryFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=quote_drafting,
        quote_delivery=quote_delivery,
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )

    request = inbound(business_id, "quote: replace two gutters", message_key="quote-1")
    await application.ingest_service.ingest(request)
    await application.ingest_service.ingest(request)
    await drain(application)

    assert len(quote_drafting.requests) == 1
    approval = [
        message
        for _, message in owner_replies.sent
        if any(isinstance(part, TextPart) and "approve" in part.text for part in message.parts)
    ]
    assert len(approval) == 1

    approve = inbound(business_id, "approve", message_key="quote-1-approve")
    await application.ingest_service.ingest(approve)
    await application.ingest_service.ingest(approve)
    await drain(application)

    assert len(quote_delivery.requests) == 1
    assert quote_delivery.requests[0].idempotency_key.startswith("quote-delivery:")
    delivery_commands = await outbox_rows(session_factory, "customer_quote.deliver")
    assert len(delivery_commands) == 1
    assert delivery_commands[0].status == OutboxStatus.SUCCEEDED.value
    async with session_factory() as session:
        statuses = list(
            (await session.scalars(select(func.count()).select_from(FieldNoteCaseRow))).all()
        )
    assert statuses == [0]


@pytest.mark.asyncio
async def test_text_field_note_path_reaches_report_with_one_follow_up(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    reports = ReportGenerationFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=reports,
    )
    await configure_checklist(application, business_id)

    intake = inbound(business_id, "field notes: site: north tower", message_key="notes-1")
    await application.ingest_service.ingest(intake)
    await application.ingest_service.ingest(intake)
    await drain(application)

    async with session_factory() as session:
        questions = (
            await session.scalars(
                select(FieldNoteFollowUpQuestion).order_by(FieldNoteFollowUpQuestion.asked_at)
            )
        ).all()
    assert [question.item_key for question in questions] == ["work"]
    assert reports.requests == []

    answer_key = questions[0].correlation_id
    answer = inbound(business_id, "Replaced the gutter run", message_key="notes-1-answer")
    await application.ingest_service.ingest(answer)
    await application.ingest_service.ingest(answer)
    await drain(application)

    assert len(reports.requests) == 1
    snapshot = reports.requests[0].source
    assert snapshot.canonical_transcript.startswith("site: north tower")
    assert {answer.question_key for answer in snapshot.correlated_answers} == {"work"}
    assert {item.item_key for item in snapshot.checklist_evidence} == {"site", "work"}
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(FieldNoteReport)) == 1
        assert (
            await session.scalar(select(func.count()).select_from(FieldNoteFollowUpQuestion)) == 1
        )
    assert answer_key is not None


class CrashingUnitOfWorkFactory:
    """Stands in for a crash between the review commit and the report enqueue."""

    def __call__(self) -> UnitOfWork:
        raise RuntimeError("crashed before the report command was enqueued")


@pytest.mark.asyncio
async def test_completed_review_recovers_a_lost_report_request(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    reports = ReportGenerationFake()
    application = build(
        session_factory,
        owner_replies=OwnerReplyFake(),
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=reports,
    )
    await configure_checklist(application, business_id)
    await application.ingest_service.ingest(
        inbound(
            business_id,
            "field notes: site: north work: inspection",
            message_key="recovered-notes",
        )
    )
    await application.worker.run_once()

    async with session_factory() as session:
        case = await session.scalar(select(FieldNoteCaseRow))
    assert case is not None
    crashing = CoordinateFieldNoteReviewService(
        application.field_note_unit_of_work_factory,
        CrashingUnitOfWorkFactory(),
        application.transcript_service,
        application.completeness_service,
    )
    with pytest.raises(RuntimeError):
        await crashing.coordinate(
            business_id,
            FieldNoteCaseId(case.id),
            FieldNoteReviewTrigger.INTAKE,
            now=NOW,
        )
    assert await outbox_rows(session_factory, "field_notes_report.generate") == []

    recovered = await application.review_service.coordinate(
        business_id,
        FieldNoteCaseId(case.id),
        FieldNoteReviewTrigger.INTAKE,
        now=NOW,
    )
    assert recovered.status is ReviewCoordinationStatus.ALREADY_COMPLETE
    assert recovered.report_requested
    await drain(application)
    await application.review_service.coordinate(
        business_id,
        FieldNoteCaseId(case.id),
        FieldNoteReviewTrigger.INTAKE,
        now=NOW,
    )
    await drain(application)

    assert len(reports.requests) == 1
    report_commands = await outbox_rows(session_factory, "field_notes_report.generate")
    assert [row.status for row in report_commands] == [OutboxStatus.SUCCEEDED.value]
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(FieldNoteReport)) == 1


@pytest.mark.asyncio
async def test_audio_field_note_transcribes_outside_the_claim_transaction(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)

    async def observe() -> str | None:
        async with session_factory() as session:
            part = await session.scalar(
                select(FieldNotePartRow).where(
                    FieldNotePartRow.transcription_status != TranscriptionStatus.NOT_REQUIRED.value
                )
            )
            return None if part is None else part.transcription_status

    transcription = TranscriptionFake(
        {
            "audio-1": "site: north work: inspection",
            "audio-2": "finding: dry reading: 12%",
        },
        observe,
    )
    application = build(
        session_factory,
        owner_replies=OwnerReplyFake(),
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=transcription,
        report_generation=ReportGenerationFake(),
    )
    await configure_checklist(application, business_id)

    intake = inbound(
        business_id,
        None,
        message_key="audio-notes",
        attachments=(audio_part("audio-1"),),
    )
    await application.ingest_service.ingest(intake)
    await drain(application)
    await application.ingest_service.ingest(
        inbound(
            business_id,
            None,
            message_key="audio-notes-2",
            attachments=(audio_part("audio-2"),),
        )
    )
    await drain(application)

    assert transcription.calls == ["audio-1", "audio-2"]
    assert transcription.observed_states == [
        TranscriptionStatus.IN_PROGRESS.value,
        TranscriptionStatus.SUCCEEDED.value,
    ]
    transcribe_commands = await outbox_rows(session_factory, "field_note.transcribe")
    assert [row.status for row in transcribe_commands] == [
        OutboxStatus.SUCCEEDED.value,
        OutboxStatus.SUCCEEDED.value,
    ]
    async with session_factory() as session:
        cases = (await session.scalars(select(FieldNoteCaseRow))).all()
    assert len(cases) == 1
    case = cases[0]
    assert case is not None
    transcript = await application.transcript_service.canonical_transcript(
        business_id, FieldNoteCaseId(case.id)
    )
    assert transcript.is_complete
    assert "site: north work: inspection" in transcript.text
    assert "finding: dry reading: 12%" in transcript.text


@pytest.mark.asyncio
async def test_tenant_isolation_and_thread_root_correlation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first = BusinessId(uuid4())
    second = BusinessId(uuid4())
    await seed_business(session_factory, first)
    await seed_business(session_factory, second)
    owner_replies = OwnerReplyFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    await configure_checklist(application, first)
    await configure_checklist(application, second)

    await application.ingest_service.ingest(
        inbound(first, "field notes: site: north", message_key="a", conversation="thread-a")
    )
    await application.ingest_service.ingest(
        inbound(second, "field notes: site: south", message_key="b", conversation="thread-b")
    )
    await drain(application)

    async with session_factory() as session:
        cases = (await session.scalars(select(FieldNoteCaseRow))).all()
    assert {case.business_id for case in cases} == {first, second}
    conversations = {
        conversation_ref.external_conversation_id for conversation_ref, _ in owner_replies.sent
    }
    assert conversations == {"thread-a", "thread-b"}
    for conversation_ref, message in owner_replies.sent:
        assert isinstance(message, OutboundOwnerMessage)
        assert message.business_id == conversation_ref.business_id


@pytest.mark.asyncio
async def test_unknown_command_fails_explicitly_and_stays_retryable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    application = build(
        session_factory,
        owner_replies=OwnerReplyFake(),
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    command = OutboxCommand(
        command_id=OutboxCommandId(uuid4()),
        business_id=business_id,
        command_type="unsupported.command",
        payload={},
        dedup_key="unsupported",
    )
    await application.outbox.enqueue(command)
    with pytest.raises(UnknownCommandTypeError):
        await application.dispatcher.dispatch(
            OutboxRecord(command=command, status=OutboxStatus.PENDING, available_at=NOW)
        )

    report = await application.worker.run_once()
    assert report.failed == 1
    rows = await outbox_rows(session_factory, "unsupported.command")
    assert [row.status for row in rows] == [OutboxStatus.FAILED.value]
    assert rows[0].attempts == 1
    assert rows[0].last_error is not None


@pytest.mark.asyncio
async def test_retired_owner_email_reply_command_is_dropped_not_dead_lettered(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    application = build(
        session_factory,
        owner_replies=OwnerReplyFake(),
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    await application.outbox.enqueue(
        OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=business_id,
            command_type="owner_email.reply",
            payload={"text": "approve"},
            dedup_key="queued-before-retirement",
        )
    )

    report = await application.worker.run_once()
    assert report.failed == 0
    rows = await outbox_rows(session_factory, "owner_email.reply")
    assert [row.status for row in rows] == [OutboxStatus.SUCCEEDED.value]


@pytest.mark.asyncio
async def test_postgres_backed_quote_and_field_note_paths(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(postgres_session_factory, business_id)
    quote_delivery = CustomerDeliveryFake()
    reports = ReportGenerationFake()
    application = build(
        postgres_session_factory,
        owner_replies=OwnerReplyFake(),
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=quote_delivery,
        transcription=TranscriptionFake({"audio-pg": "site: north work: inspection"}),
        report_generation=reports,
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "quote: install a gate", message_key="pg-quote", conversation="pg-q")
    )
    await drain(application)
    await application.ingest_service.ingest(
        inbound(business_id, "approve", message_key="pg-quote-approve", conversation="pg-q")
    )
    await drain(application)
    assert len(quote_delivery.requests) == 1

    await application.ingest_service.ingest(
        inbound(
            business_id,
            "field notes: voice memo",
            message_key="pg-notes",
            conversation="pg-n",
            attachments=(audio_part("audio-pg"),),
        )
    )
    await drain(application)
    assert len(reports.requests) == 1

    async with postgres_session_factory() as session:
        pending = await session.scalar(
            select(func.count())
            .select_from(OutboxMessage)
            .where(OutboxMessage.status != OutboxStatus.SUCCEEDED.value)
        )
        quotes = (await session.scalars(select(func.count()).select_from(FieldNoteCaseRow))).all()
    assert pending == 0
    assert quotes == [1]


def reply_texts(owner_replies: OwnerReplyFake) -> list[str]:
    return [
        part.text
        for _, message in owner_replies.sent
        for part in message.parts
        if isinstance(part, TextPart)
    ]


async def case_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[FieldNoteCaseRow]:
    async with session_factory() as session:
        return list(
            (
                await session.scalars(
                    select(FieldNoteCaseRow).order_by(FieldNoteCaseRow.created_at)
                )
            ).all()
        )


async def report_versions(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[FieldNoteReportVersion]:
    async with session_factory() as session:
        return list(
            (
                await session.scalars(
                    select(FieldNoteReportVersion).order_by(FieldNoteReportVersion.version)
                )
            ).all()
        )


async def unsucceeded_outbox(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(OutboxMessage)
            .where(OutboxMessage.status != OutboxStatus.SUCCEEDED.value)
        )
    return int(count or 0)


def field_note_application(
    session_factory: async_sessionmaker[AsyncSession], owner_replies: OwnerReplyFake
) -> Application:
    return build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )


@pytest.mark.asyncio
async def test_close_notes_closes_the_case_once_under_replay(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    reports = ReportGenerationFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=reports,
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    assert len(reports.requests) == 1
    open_cases = await case_rows(session_factory)
    assert [case.status for case in open_cases] == [FieldNoteCaseStatus.OPEN.value]
    assert open_cases[0].closed_at is None

    close = inbound(business_id, "  Close Notes  ", message_key="close-1")
    await application.ingest_service.ingest(close)
    await application.ingest_service.ingest(close)
    await drain(application)

    closed = await case_rows(session_factory)
    assert [case.status for case in closed] == [FieldNoteCaseStatus.CLOSED.value]
    assert closed[0].closed_at is not None
    assert reply_texts(owner_replies).count(CLOSED_CASE_REPLY) == 1

    await application.ingest_service.ingest(
        inbound(business_id, "close notes", message_key="close-2")
    )
    await drain(application)

    assert [case.status for case in await case_rows(session_factory)] == [
        FieldNoteCaseStatus.CLOSED.value
    ]
    assert reply_texts(owner_replies).count(CLOSED_CASE_REPLY) == 1
    assert reply_texts(owner_replies).count(NO_OPEN_CASE_REPLY) == 1


@pytest.mark.asyncio
async def test_close_notes_without_an_active_case_replies_without_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = field_note_application(session_factory, owner_replies)
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "close notes", message_key="close-only")
    )
    await drain(application)

    assert await unsucceeded_outbox(session_factory) == 0
    assert await case_rows(session_factory) == []
    assert reply_texts(owner_replies) == [NO_OPEN_CASE_REPLY]


@pytest.mark.asyncio
async def test_unmatched_first_message_gets_the_triggers_once_without_retrying(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = field_note_application(session_factory, owner_replies)
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "hey, are you there?", message_key="unmatched-1")
    )
    await drain(application)
    await drain(application)

    assert await unsucceeded_outbox(session_factory) == 0
    assert await case_rows(session_factory) == []
    assert reply_texts(owner_replies) == [UNMATCHED_MESSAGE_REPLY]
    assert "field notes:" in UNMATCHED_MESSAGE_REPLY
    assert "quote:" in UNMATCHED_MESSAGE_REPLY


@pytest.mark.asyncio
async def test_field_notes_after_closure_start_a_distinct_case(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    application = field_note_application(session_factory, OwnerReplyFake())
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    await application.ingest_service.ingest(
        inbound(business_id, "close notes", message_key="close-1")
    )
    await drain(application)
    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: south work: repair", message_key="notes-2")
    )
    await drain(application)

    cases = await case_rows(session_factory)
    assert [case.status for case in cases] == [
        FieldNoteCaseStatus.CLOSED.value,
        FieldNoteCaseStatus.OPEN.value,
    ]
    assert cases[0].id != cases[1].id
    async with session_factory() as session:
        parts = (
            await session.scalars(
                select(FieldNotePartRow).where(FieldNotePartRow.case_id == cases[1].id)
            )
        ).all()
    assert [part.text for part in parts] == ["site: south work: repair"]


@pytest.mark.asyncio
async def test_notes_after_a_report_extend_the_open_case(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Extending an open case reviews the updated transcript and versions the report."""

    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    reports = ReportGenerationFake()
    application = build(
        session_factory,
        owner_replies=OwnerReplyFake(),
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=reports,
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)
    assert len(reports.requests) == 1

    follow_up = inbound(business_id, "work: replaced the downpipe", message_key="notes-1-more")
    await application.ingest_service.ingest(follow_up)
    await application.ingest_service.ingest(follow_up)
    await drain(application)

    cases = await case_rows(session_factory)
    assert [case.status for case in cases] == [FieldNoteCaseStatus.OPEN.value]
    async with session_factory() as session:
        parts = (
            await session.scalars(select(FieldNotePartRow).order_by(FieldNotePartRow.sequence))
        ).all()
    assert [part.case_id for part in parts] == [cases[0].id, cases[0].id]
    assert len(reports.requests) == 2
    assert "replaced the downpipe" in reports.requests[1].source.canonical_transcript
    assert [version.version for version in await report_versions(session_factory)] == [1, 2]
    assert len(await outbox_rows(session_factory, "field_notes_report.generate")) == 2

    await application.ingest_service.ingest(follow_up)
    await drain(application)

    assert len(reports.requests) == 2
    assert [version.version for version in await report_versions(session_factory)] == [1, 2]


@pytest.mark.asyncio
async def test_field_notes_trigger_during_an_active_quote_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    quote_drafting = QuoteDraftingFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=quote_drafting,
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "quote: replace two gutters", message_key="quote-1")
    )
    await drain(application)

    notes = inbound(business_id, "field notes: site: north", message_key="notes-during-quote")
    await application.ingest_service.ingest(notes)
    await application.ingest_service.ingest(notes)
    await drain(application)

    assert await unsucceeded_outbox(session_factory) == 0
    assert await case_rows(session_factory) == []
    assert len(quote_drafting.requests) == 1
    assert reply_texts(owner_replies).count(FIELD_NOTE_CONFLICT_REPLY) == 1
    assert "separate thread or conversation" in FIELD_NOTE_CONFLICT_REPLY


@pytest.mark.asyncio
async def test_quote_trigger_during_an_active_field_note_case_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    quote_drafting = QuoteDraftingFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=quote_drafting,
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "field notes: site: north work: inspection", message_key="notes-1")
    )
    await drain(application)

    quote = inbound(business_id, "quote: replace two gutters", message_key="quote-during-notes")
    await application.ingest_service.ingest(quote)
    await application.ingest_service.ingest(quote)
    await drain(application)

    assert await unsucceeded_outbox(session_factory) == 0
    assert quote_drafting.requests == []
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(QuoteRecord)) == 0
    assert [case.status for case in await case_rows(session_factory)] == [
        FieldNoteCaseStatus.OPEN.value
    ]
    assert reply_texts(owner_replies).count(QUOTE_CONFLICT_REPLY) == 1


@pytest.mark.asyncio
async def test_close_notes_during_a_quote_only_conversation_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    owner_replies = OwnerReplyFake()
    application = build(
        session_factory,
        owner_replies=owner_replies,
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=ReportGenerationFake(),
    )
    await configure_checklist(application, business_id)

    await application.ingest_service.ingest(
        inbound(business_id, "quote: replace two gutters", message_key="quote-1")
    )
    await drain(application)
    await application.ingest_service.ingest(
        inbound(business_id, "close notes", message_key="close-during-quote")
    )
    await drain(application)

    assert await unsucceeded_outbox(session_factory) == 0
    assert await case_rows(session_factory) == []
    assert reply_texts(owner_replies).count(FIELD_NOTE_CONFLICT_REPLY) == 1
    assert NO_OPEN_CASE_REPLY not in reply_texts(owner_replies)


@pytest.mark.asyncio
async def test_close_notes_is_scoped_to_one_business(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first = BusinessId(uuid4())
    second = BusinessId(uuid4())
    await seed_business(session_factory, first)
    await seed_business(session_factory, second)
    owner_replies = OwnerReplyFake()
    application = field_note_application(session_factory, owner_replies)
    await configure_checklist(application, first)
    await configure_checklist(application, second)

    for business_id in (first, second):
        await application.ingest_service.ingest(
            inbound(
                business_id,
                "field notes: site: north work: inspection",
                message_key="notes-1",
                conversation="shared",
            )
        )
    await drain(application)
    await application.ingest_service.ingest(
        inbound(first, "close notes", message_key="close-1", conversation="shared")
    )
    await drain(application)

    cases = await case_rows(session_factory)
    statuses = {case.business_id: case.status for case in cases}
    assert statuses == {
        first: FieldNoteCaseStatus.CLOSED.value,
        second: FieldNoteCaseStatus.OPEN.value,
    }
    closing = [
        conversation_ref.business_id
        for conversation_ref, message in owner_replies.sent
        if any(
            isinstance(part, TextPart) and part.text == CLOSED_CASE_REPLY for part in message.parts
        )
    ]
    assert closing == [first]


@pytest.mark.asyncio
async def test_postgres_backed_report_versions_follow_case_revisions(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(postgres_session_factory, business_id)
    reports = ReportGenerationFake()
    application = build(
        postgres_session_factory,
        owner_replies=OwnerReplyFake(),
        quote_drafting=QuoteDraftingFake(),
        quote_delivery=CustomerDeliveryFake(),
        transcription=TranscriptionFake({}),
        report_generation=reports,
    )
    await configure_checklist(application, business_id)

    first = inbound(
        business_id,
        "field notes: site: north work: inspection",
        message_key="pg-rev-notes",
        conversation="pg-rev",
    )
    await application.ingest_service.ingest(first)
    await drain(application)
    assert [version.version for version in await report_versions(postgres_session_factory)] == [1]

    more = inbound(
        business_id,
        "work: replaced the downpipe",
        message_key="pg-rev-more",
        conversation="pg-rev",
    )
    await application.ingest_service.ingest(more)
    await drain(application)

    versions = await report_versions(postgres_session_factory)
    assert [version.version for version in versions] == [1, 2]
    assert versions[0].source_fingerprint != versions[1].source_fingerprint
    assert "replaced the downpipe" in reports.requests[1].source.canonical_transcript

    await application.ingest_service.ingest(first)
    await application.ingest_service.ingest(more)
    await drain(application)

    assert [version.version for version in await report_versions(postgres_session_factory)] == [
        1,
        2,
    ]
    assert len(reports.requests) == 2
    assert [case.status for case in await case_rows(postgres_session_factory)] == [
        FieldNoteCaseStatus.OPEN.value
    ]
    assert await unsucceeded_outbox(postgres_session_factory) == 0
