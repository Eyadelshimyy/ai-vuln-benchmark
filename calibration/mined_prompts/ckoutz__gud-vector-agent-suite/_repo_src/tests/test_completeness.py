from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.completeness import (
    CompletenessStatus,
    FieldNoteCompletenessService,
)
from gvas.application.completeness_review import MarkerCompletenessReviewer
from gvas.application.templates import (
    IndustryTemplateDefinition,
    PublishTemplateSetService,
    TemplateResolver,
)
from gvas.domain.completeness import (
    ChecklistItem,
    ChecklistItemKey,
    ChecklistKey,
    ChecklistVersionConflictError,
    CompletenessChecklist,
    CompletenessReviewOutcome,
    CompletenessReviewPort,
    CompletenessReviewRequest,
    CorrelatedAnswer,
    MissingChecklistItem,
    UnknownChecklistItemError,
)
from gvas.domain.identifiers import BusinessId, ConversationId, MessageId, MessageKey
from gvas.domain.messages import (
    ConversationRef,
    NormalizedOwnerMessage,
    ReplyRef,
    SenderRef,
    TextPart,
)
from gvas.domain.templates import IndustryKey, ReportTemplateSection, TemplateSetKey
from gvas.infrastructure.completeness_models import (
    FieldNoteFollowUpQuestion,
    FieldNoteReview,
    FieldNoteReviewAnswer,
)
from gvas.infrastructure.models import (
    Business,
    Conversation,
    InboundMessage,
    OutboundMessage,
    OutboxMessage,
    OwnerChannelEndpoint,
)
from gvas.infrastructure.unit_of_work import SqlCompletenessUnitOfWorkFactory

NOW = datetime(2025, 1, 1, tzinfo=UTC)


class UnitOfWorkAwareReviewer:
    def __init__(
        self,
        uow_factory: SqlCompletenessUnitOfWorkFactory,
        fail_on_call: int | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._fail_on_call = fail_on_call
        self.calls = 0

    async def review(self, request: CompletenessReviewRequest) -> CompletenessReviewOutcome:
        self.calls += 1
        async with self._uow_factory() as uow:
            assert (
                await uow.checklists.get(
                    request.business_id, request.checklist.checklist_key, request.checklist.version
                )
                is not None
            )
            await uow.rollback()
        if self.calls == self._fail_on_call:
            raise RuntimeError("review failed")
        answered = {answer.item_key for answer in request.answers}
        for item in request.checklist.items:
            if item.key not in answered:
                return CompletenessReviewOutcome(
                    missing_items=(MissingChecklistItem(item_key=item.key, prompt=item.prompt),)
                )
        return CompletenessReviewOutcome()


async def seed_context(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> tuple[ConversationId, MessageId]:
    endpoint_id = uuid4()
    conversation_id = ConversationId(uuid4())
    inbound_id = MessageId(uuid4())
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
        session.add(
            OwnerChannelEndpoint(
                id=endpoint_id,
                business_id=business_id,
                source_namespace="test",
                external_endpoint_id=f"endpoint-{business_id}",
                routing={},
            )
        )
        session.add(
            Conversation(
                id=conversation_id,
                business_id=business_id,
                endpoint_id=endpoint_id,
                external_conversation_id=f"conversation-{business_id}",
                routing={},
            )
        )
        session.add(
            InboundMessage(
                id=inbound_id,
                business_id=business_id,
                endpoint_id=endpoint_id,
                conversation_id=conversation_id,
                message_key="transcript",
                sender_external_id="owner",
                sender_role="owner",
                received_at=NOW,
                parts=[{"kind": "text", "text": "field notes"}],
                reply_to=None,
                routing={},
            )
        )
        await session.commit()
    return conversation_id, inbound_id


async def seed_reply(
    session_factory: async_sessionmaker[AsyncSession],
    business_id: BusinessId,
    conversation_id: ConversationId,
    message_key: str,
) -> MessageId:
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation is not None
        inbound_id = MessageId(uuid4())
        session.add(
            InboundMessage(
                id=inbound_id,
                business_id=business_id,
                endpoint_id=conversation.endpoint_id,
                conversation_id=conversation_id,
                message_key=message_key,
                sender_external_id="owner",
                sender_role="owner",
                received_at=NOW,
                parts=[{"kind": "text", "text": "answer"}],
                reply_to=None,
                routing={},
            )
        )
        await session.commit()
    return inbound_id


def checklist(business_id: BusinessId, version: int = 1) -> CompletenessChecklist:
    return CompletenessChecklist(
        business_id=business_id,
        checklist_key=ChecklistKey("configured-review"),
        version=version,
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


async def configure(
    session_factory: async_sessionmaker[AsyncSession], definition: CompletenessChecklist
) -> None:
    await PublishTemplateSetService(
        SqlCompletenessUnitOfWorkFactory(session_factory)
    ).seed_industry(definition.business_id, industry(definition))


def industry(definition: CompletenessChecklist) -> IndustryTemplateDefinition:
    return IndustryTemplateDefinition(
        industry_key=IndustryKey("configured"),
        template_set_key=TemplateSetKey(definition.checklist_key),
        checklist_key=definition.checklist_key,
        version=definition.version,
        items=definition.items,
        report_template_key="configured-report",
        report_title="Configured report",
        report_sections=(
            ReportTemplateSection(
                section_key="findings",
                heading="Findings",
                checklist_item_keys=tuple(item.key for item in definition.items),
            ),
        ),
    )


def owner_reply(
    business_id: BusinessId, correlation_id: str | None, text: str = "answer"
) -> NormalizedOwnerMessage:
    return NormalizedOwnerMessage(
        message_key=MessageKey(str(uuid4())),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id=f"conversation-{business_id}"
        ),
        sender=SenderRef(external_id="owner", role="owner"),
        received_at=NOW,
        parts=(TextPart(text=text),),
        reply_to=ReplyRef(correlation_id=correlation_id) if correlation_id else None,
    )


def service(
    session_factory: async_sessionmaker[AsyncSession],
    reviewer: CompletenessReviewPort | None = None,
) -> FieldNoteCompletenessService:
    return FieldNoteCompletenessService(
        SqlCompletenessUnitOfWorkFactory(session_factory),
        reviewer or MarkerCompletenessReviewer(),
        TemplateResolver(SqlCompletenessUnitOfWorkFactory(session_factory)),
    )


async def outgoing_messages(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[OutboundMessage]:
    async with session_factory() as session:
        return list(
            (
                await session.scalars(
                    select(OutboundMessage).order_by(OutboundMessage.correlation_id)
                )
            ).all()
        )


@pytest.mark.asyncio
async def test_no_missing_items_completes_without_questions(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)
    result = await service(session_factory).start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "site: north work: inspection",
    )

    assert result.status is CompletenessStatus.COMPLETE
    assert result.round_index == 0
    assert await outgoing_messages(session_factory) == []


@pytest.mark.asyncio
async def test_multiple_rounds_complete_only_after_all_requirements_are_satisfied(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)
    reviewer = UnitOfWorkAwareReviewer(SqlCompletenessUnitOfWorkFactory(session_factory))
    completeness = service(session_factory, reviewer)

    started = await completeness.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "No configured evidence markers are present.",
    )
    assert started.status is CompletenessStatus.QUESTIONS_SENT
    assert started.round_index == 1
    assert started.questions_sent == 1
    first = (await outgoing_messages(session_factory))[0]
    assert first.reply_to is not None
    assert first.reply_to["correlation_id"] == f"field_note:{inbound_id}"

    first_reply_id = await seed_reply(session_factory, business_id, conversation_id, "first-reply")
    partial = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        first_reply_id,
        owner_reply(business_id, first.correlation_id, "North depot"),
    )
    assert partial.status is CompletenessStatus.QUESTIONS_SENT
    assert partial.round_index == 2
    assert partial.questions_sent == 1
    first, second = await outgoing_messages(session_factory)
    assert first.reply_to == second.reply_to

    second_reply_id = await seed_reply(
        session_factory, business_id, conversation_id, "second-reply"
    )
    completed = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        second_reply_id,
        owner_reply(business_id, second.correlation_id, "Inspection"),
    )
    assert completed.status is CompletenessStatus.COMPLETE
    assert completed.round_index == 2
    assert reviewer.calls == 3


@pytest.mark.asyncio
async def test_duplicate_start_and_duplicate_reply_do_not_duplicate_questions_or_answers(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)
    completeness = service(session_factory)

    first = await completeness.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "missing",
    )
    repeated = await completeness.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "missing",
    )
    assert first.questions_sent == 1
    assert repeated.questions_sent == 0
    messages = await outgoing_messages(session_factory)
    assert len(messages) == 1

    reply_id = await seed_reply(session_factory, business_id, conversation_id, "reply")
    first_reply = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        reply_id,
        owner_reply(business_id, f"field_note:{inbound_id}"),
    )
    duplicate = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        reply_id,
        owner_reply(business_id, f"field_note:{inbound_id}"),
    )
    assert first_reply.status is CompletenessStatus.QUESTIONS_SENT
    assert first_reply.questions_sent == 1
    assert duplicate.status is CompletenessStatus.DUPLICATE_REPLY
    assert len(await outgoing_messages(session_factory)) == 2
    async with session_factory() as session:
        assert await session.scalar(select(func.count(FieldNoteReviewAnswer.id))) == 1
        assert await session.scalar(select(func.count(FieldNoteFollowUpQuestion.id))) == 2
        assert await session.scalar(select(func.count(OutboxMessage.id))) == 2


@pytest.mark.asyncio
async def test_retry_after_review_failure_resumes_from_persisted_answer(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)
    reviewer = UnitOfWorkAwareReviewer(
        SqlCompletenessUnitOfWorkFactory(session_factory), fail_on_call=2
    )
    completeness = service(session_factory, reviewer)
    await completeness.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "missing",
    )
    first = (await outgoing_messages(session_factory))[0]
    reply_id = await seed_reply(session_factory, business_id, conversation_id, "retry-reply")
    message = owner_reply(business_id, first.correlation_id, "North depot")

    with pytest.raises(RuntimeError, match="review failed"):
        await completeness.record_owner_reply(business_id, conversation_id, reply_id, message)
    resumed = await completeness.record_owner_reply(business_id, conversation_id, reply_id, message)

    assert resumed.status is CompletenessStatus.QUESTIONS_SENT
    assert resumed.round_index == 2
    assert resumed.questions_sent == 1
    assert len(await outgoing_messages(session_factory)) == 2
    async with session_factory() as session:
        assert await session.scalar(select(func.count(FieldNoteReviewAnswer.id))) == 1


@pytest.mark.asyncio
async def test_thread_root_reply_advances_questions_sequentially(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)
    completeness = service(session_factory)
    await completeness.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "missing",
    )
    messages = await outgoing_messages(session_factory)
    assert len(messages) == 1
    first = messages[0]
    thread_correlation = f"field_note:{inbound_id}"
    assert first.correlation_id != thread_correlation

    first_reply_id = await seed_reply(
        session_factory, business_id, conversation_id, "thread-root-first"
    )
    advanced = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        first_reply_id,
        owner_reply(business_id, thread_correlation, "North depot"),
    )
    assert advanced.status is CompletenessStatus.QUESTIONS_SENT
    assert advanced.questions_sent == 1
    first, second = await outgoing_messages(session_factory)
    assert first.reply_to == second.reply_to

    second_reply_id = await seed_reply(
        session_factory, business_id, conversation_id, "thread-root-second"
    )
    completed = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        second_reply_id,
        owner_reply(business_id, thread_correlation, "Inspection"),
    )
    assert completed.status is CompletenessStatus.COMPLETE
    assert completed.round_index == 1
    async with session_factory() as session:
        answers = (
            await session.scalars(
                select(FieldNoteReviewAnswer.item_key).order_by(FieldNoteReviewAnswer.item_key)
            )
        ).all()
        assert answers == ["site", "work"]


@pytest.mark.asyncio
async def test_reply_is_rejected_when_no_question_is_currently_asked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)
    reviewer = UnitOfWorkAwareReviewer(
        SqlCompletenessUnitOfWorkFactory(session_factory), fail_on_call=2
    )
    completeness = service(session_factory, reviewer)
    await completeness.start_review(
        business_id,
        conversation_id,
        f"conversation-{business_id}",
        inbound_id,
        "missing",
    )
    answered_id = await seed_reply(session_factory, business_id, conversation_id, "answered")
    with pytest.raises(RuntimeError, match="review failed"):
        await completeness.record_owner_reply(
            business_id,
            conversation_id,
            answered_id,
            owner_reply(business_id, f"field_note:{inbound_id}"),
        )

    extra_reply_id = await seed_reply(session_factory, business_id, conversation_id, "extra")
    result = await completeness.record_owner_reply(
        business_id,
        conversation_id,
        extra_reply_id,
        owner_reply(business_id, f"field_note:{inbound_id}"),
    )

    assert result.status is CompletenessStatus.UNCORRELATED_REPLY
    async with session_factory() as session:
        assert await session.scalar(select(func.count(FieldNoteReviewAnswer.id))) == 1


@pytest.mark.asyncio
async def test_tenant_isolation_prevents_cross_business_correlation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first_business = BusinessId(uuid4())
    second_business = BusinessId(uuid4())
    first_conversation, first_inbound = await seed_context(session_factory, first_business)
    second_conversation, _ = await seed_context(session_factory, second_business)
    definition = checklist(first_business)
    await configure(session_factory, definition)
    completeness = service(session_factory)
    await completeness.start_review(
        first_business,
        first_conversation,
        f"conversation-{first_business}",
        first_inbound,
        "missing",
    )
    first_question = (await outgoing_messages(session_factory))[0]
    second_reply_id = await seed_reply(
        session_factory, second_business, second_conversation, "cross-tenant"
    )

    result = await completeness.record_owner_reply(
        second_business,
        second_conversation,
        second_reply_id,
        owner_reply(second_business, first_question.correlation_id),
    )
    assert result.status is CompletenessStatus.NO_ACTIVE_REVIEW
    async with session_factory() as session:
        assert await session.scalar(select(func.count(FieldNoteReviewAnswer.id))) == 0
        assert await session.scalar(select(func.count(FieldNoteReview.id))) == 1


@pytest.mark.asyncio
async def test_versioned_checklists_are_tenant_scoped_and_latest_is_selected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    other_business = BusinessId(uuid4())
    await seed_context(session_factory, business_id)
    await seed_context(session_factory, other_business)
    await configure(session_factory, checklist(business_id, version=1))
    await configure(session_factory, checklist(business_id, version=2))
    await configure(session_factory, checklist(other_business, version=1))

    async with SqlCompletenessUnitOfWorkFactory(session_factory)() as uow:
        latest = await uow.checklists.get(business_id, ChecklistKey("configured-review"))
        other = await uow.checklists.get(other_business, ChecklistKey("configured-review"))
        await uow.rollback()
    assert latest is not None and latest.version == 2
    assert other is not None and other.business_id == other_business


@pytest.mark.asyncio
async def test_checklist_versions_are_immutable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)
    changed = definition.model_copy(
        update={
            "items": (
                ChecklistItem(
                    key=ChecklistItemKey("different"),
                    prompt="Different requirement",
                ),
            )
        }
    )

    async with SqlCompletenessUnitOfWorkFactory(session_factory)() as uow:
        with pytest.raises(ChecklistVersionConflictError):
            await uow.checklists.upsert(changed)
        await uow.rollback()


@pytest.mark.asyncio
async def test_marker_reviewer_uses_configured_markers_and_correlated_answers() -> None:
    business_id = BusinessId(uuid4())
    definition = checklist(business_id)
    result = await MarkerCompletenessReviewer().review(
        CompletenessReviewRequest(
            business_id=business_id,
            checklist=definition,
            transcript_text="site: north",
            answers=(
                CorrelatedAnswer(
                    item_key=ChecklistItemKey("work"),
                    text="Inspection",
                    received_at=NOW,
                ),
            ),
            round_index=1,
        )
    )
    assert result.is_complete

    missing = await MarkerCompletenessReviewer().review(
        CompletenessReviewRequest(
            business_id=business_id,
            checklist=definition,
            transcript_text="none",
            round_index=0,
        )
    )
    assert missing == CompletenessReviewOutcome(
        missing_items=(
            MissingChecklistItem(
                item_key=ChecklistItemKey("site"), prompt="Which site was visited?"
            ),
            MissingChecklistItem(
                item_key=ChecklistItemKey("work"), prompt="What work was performed?"
            ),
        )
    )


@pytest.mark.asyncio
async def test_unknown_review_item_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    conversation_id, inbound_id = await seed_context(session_factory, business_id)
    definition = checklist(business_id)
    await configure(session_factory, definition)

    class UnknownItemReviewer:
        async def review(self, request: CompletenessReviewRequest) -> CompletenessReviewOutcome:
            return CompletenessReviewOutcome(
                missing_items=(
                    MissingChecklistItem(
                        item_key=ChecklistItemKey("unknown"),
                        prompt="Unknown requirement",
                    ),
                )
            )

    with pytest.raises(UnknownChecklistItemError):
        await service(session_factory, UnknownItemReviewer()).start_review(
            business_id,
            conversation_id,
            f"conversation-{business_id}",
            inbound_id,
            "missing",
        )
    assert await outgoing_messages(session_factory) == []
