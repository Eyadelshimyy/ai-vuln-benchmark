from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.ingestion import IngestionStatus, IngestOwnerMessageService
from gvas.application.intents import UnconfiguredIntentResolver
from gvas.application.processing import ProcessingStatus, ProcessOwnerMessageService
from gvas.domain.enums import WorkflowRunStatus
from gvas.domain.identifiers import (
    BusinessId,
    MessageId,
    MessageKey,
    OutboxCommandId,
    WorkflowIntent,
)
from gvas.domain.intents import IntentResolution
from gvas.domain.messages import (
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.outbox import (
    OWNER_MESSAGE_PROCESS_COMMAND_TYPE,
    OWNER_REPLY_COMMAND_TYPE,
    OutboxCommand,
    ReservedOutboxCommandTypeError,
    owner_message_process_command,
    owner_reply_command,
)
from gvas.domain.repositories import CrossBusinessReferenceError, LostWorkflowLeaseError
from gvas.domain.workflows import WorkflowContext, WorkflowResult, WorkflowRouter
from gvas.infrastructure.models import (
    Business,
    InboundMessage,
    OutboundMessage,
    OutboxMessage,
    WorkflowRun,
)
from gvas.infrastructure.repositories import SqlOutboxRepository
from gvas.infrastructure.unit_of_work import SqlUnitOfWork, SqlUnitOfWorkFactory

TEST_NOW = datetime(2025, 1, 1, tzinfo=UTC)
TEST_STALE_BEFORE = datetime(2024, 1, 1, tzinfo=UTC)


class EchoResolver:
    async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
        return IntentResolution(intent=WorkflowIntent("echo"))


class ReplyHandler:
    intent = WorkflowIntent("echo")

    def __init__(self) -> None:
        self.calls = 0

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        self.calls += 1
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            replies=(
                OutboundOwnerMessage(
                    business_id=context.message.business_id,
                    conversation_ref=context.message.conversation_ref,
                    parts=(TextPart(text="reply"),),
                    correlation_id="deterministic-reply",
                ),
            ),
        )


class FailingHandler:
    intent = WorkflowIntent("echo")

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        raise RuntimeError("handler failed")


class ReservedHandler:
    intent = WorkflowIntent("echo")

    async def handle(self, context: WorkflowContext) -> WorkflowResult:
        return WorkflowResult(
            status=WorkflowRunStatus.SUCCEEDED,
            commands=(
                OutboxCommand(
                    command_id=OutboxCommandId(uuid4()),
                    business_id=context.message.business_id,
                    command_type=OWNER_REPLY_COMMAND_TYPE,
                    payload={"outbound_message_id": str(uuid4())},
                    outbound_message_id=MessageId(uuid4()),
                ),
            ),
        )


def make_message(
    business_id: BusinessId,
    *,
    endpoint: str = "endpoint",
    conversation: str = "conversation",
    message_key: str = "message",
) -> InboundOwnerMessage:
    normalized = NormalizedOwnerMessage(
        message_key=MessageKey(message_key),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id=conversation
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(TextPart(text="hello"),),
    )
    return InboundOwnerMessage(
        message=normalized,
        endpoint=ChannelEndpointRef(
            business_id=business_id,
            source_namespace="test",
            external_endpoint_id=endpoint,
        ),
        routing={"endpoint": endpoint},
    )


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


@pytest.mark.asyncio
async def test_ingress_persists_process_command_without_processing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    outcome = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert outcome.status is IngestionStatus.ACCEPTED
    assert outcome.process_command_id is not None
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(WorkflowRun)) == 0
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0
        command = await session.scalar(select(OutboxMessage))
        assert command is not None
        assert command.command_type == OWNER_MESSAGE_PROCESS_COMMAND_TYPE
        assert command.inbound_message_id == outcome.message_id
        claim_now = datetime.now(UTC) + timedelta(days=1)
        claimed = await SqlOutboxRepository(session).claim_batch(
            1, claim_now, "test", stale_before=claim_now - timedelta(days=1)
        )
        assert len(claimed) == 1
        assert claimed[0].command.inbound_message_id == outcome.message_id


@pytest.mark.asyncio
async def test_duplicate_ingress_has_one_process_command(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    service = IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory))
    message = make_message(business_id)
    first = await service.ingest(message)
    second = await service.ingest(message)
    assert first.status is IngestionStatus.ACCEPTED
    assert second.status is IngestionStatus.DUPLICATE
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(InboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 1


@pytest.mark.asyncio
async def test_processing_success_is_terminal_and_replay_safe(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    message = make_message(business_id)
    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        message
    )
    assert ingestion.message_id is not None
    handler = ReplyHandler()
    processing = ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(session_factory), WorkflowRouter([handler]), EchoResolver()
    )
    first = await processing.process(
        ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE
    )
    second = await processing.process(
        ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE
    )
    assert first.status is ProcessingStatus.COMPLETED
    assert second.status is ProcessingStatus.ALREADY_PROCESSED
    assert handler.calls == 1
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(WorkflowRun)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 2


@pytest.mark.asyncio
async def test_processing_reclaim_does_not_duplicate_reply(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert ingestion.message_id is not None
    handler = ReplyHandler()
    processing = ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(session_factory), WorkflowRouter([handler]), EchoResolver()
    )
    await processing.process(ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE)
    async with session_factory() as session:
        await session.execute(
            update(WorkflowRun)
            .values(status=WorkflowRunStatus.FAILED.value)
            .where(WorkflowRun.inbound_message_id == ingestion.message_id)
        )
        await session.commit()
    replay = await processing.process(
        ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE
    )
    assert replay.status is ProcessingStatus.COMPLETED
    assert handler.calls == 2
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboxMessage)
                .where(OutboxMessage.command_type == OWNER_REPLY_COMMAND_TYPE)
            )
            == 1
        )


@pytest.mark.asyncio
async def test_stale_workflow_claim_cannot_write_after_reclaim(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert ingestion.message_id is not None

    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        first = await unit_of_work.workflow_runs.claim(
            business_id,
            ingestion.message_id,
            now=TEST_NOW,
            stale_before=TEST_STALE_BEFORE,
        )
        await unit_of_work.commit()

    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        busy = await unit_of_work.workflow_runs.claim(
            business_id,
            ingestion.message_id,
            now=TEST_NOW,
            stale_before=TEST_STALE_BEFORE,
        )
        await unit_of_work.commit()
    assert busy.lease_token is None

    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        second = await unit_of_work.workflow_runs.claim(
            business_id,
            ingestion.message_id,
            now=TEST_NOW + timedelta(days=1),
            stale_before=TEST_NOW + timedelta(hours=1),
        )
        await unit_of_work.commit()

    assert first.lease_token is not None
    assert second.lease_token is not None
    assert second.lease_token != first.lease_token
    with pytest.raises(LostWorkflowLeaseError):
        async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
            await unit_of_work.workflow_runs.set_intent(first, WorkflowIntent("echo"))


def test_workflow_result_rejects_non_terminal_status() -> None:
    for status in (WorkflowRunStatus.PENDING, WorkflowRunStatus.RUNNING):
        with pytest.raises(ValueError, match="terminal"):
            WorkflowResult(status=status)


@pytest.mark.asyncio
async def test_handler_failure_is_durable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert ingestion.message_id is not None
    outcome = await ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(session_factory),
        WorkflowRouter([FailingHandler()]),
        EchoResolver(),
    ).process(ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE)
    assert outcome.status is ProcessingStatus.HANDLER_FAILED
    assert outcome.detail == "RuntimeError('handler failed')"
    async with session_factory() as session:
        run = await session.scalar(select(WorkflowRun))
        assert run is not None
        assert run.status == WorkflowRunStatus.FAILED.value
        assert run.error == outcome.detail
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0


@pytest.mark.asyncio
async def test_unresolved_intent_is_resumable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert ingestion.message_id is not None
    unresolved = await ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(session_factory),
        WorkflowRouter([]),
        UnconfiguredIntentResolver(),
    ).process(ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE)
    assert unresolved.status is ProcessingStatus.INTENT_UNRESOLVED
    handler = ReplyHandler()
    completed = await ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(session_factory), WorkflowRouter([handler]), EchoResolver()
    ).process(
        ingestion.message_id,
        now=TEST_NOW,
        stale_before=TEST_NOW + timedelta(days=1),
    )
    assert completed.status is ProcessingStatus.COMPLETED
    async with session_factory() as session:
        run = await session.scalar(select(WorkflowRun))
        assert run is not None
        assert run.status == WorkflowRunStatus.SUCCEEDED.value
        assert run.attempts == 2
        assert run.error is None


@pytest.mark.asyncio
async def test_unknown_intent_is_failed(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert ingestion.message_id is not None
    outcome = await ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(session_factory), WorkflowRouter([]), EchoResolver()
    ).process(ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE)
    assert outcome.status is ProcessingStatus.UNKNOWN_INTENT
    async with session_factory() as session:
        run = await session.scalar(select(WorkflowRun))
        assert run is not None
        assert run.intent == "echo"
        assert run.status == WorkflowRunStatus.FAILED.value


@pytest.mark.asyncio
async def test_provider_calls_happen_without_open_uow(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    state: dict[str, bool | str | None] = {"open": False, "violation": None}

    class TrackingUnitOfWork(SqlUnitOfWork):
        async def __aenter__(self) -> "TrackingUnitOfWork":
            state["open"] = True
            await super().__aenter__()
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            try:
                await super().__aexit__(exc_type, exc, traceback)
            finally:
                state["open"] = False

    class TrackingFactory:
        def __call__(self) -> TrackingUnitOfWork:
            return TrackingUnitOfWork(session_factory)

    class GuardResolver:
        async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
            if state["open"] is True:
                state["violation"] = "resolver called with an open unit of work"
                raise AssertionError("resolver called with an open unit of work")
            return IntentResolution(intent=WorkflowIntent("echo"))

    class GuardHandler:
        intent = WorkflowIntent("echo")

        async def handle(self, context: WorkflowContext) -> WorkflowResult:
            if state["open"] is True:
                state["violation"] = "handler called with an open unit of work"
                raise AssertionError("handler called with an open unit of work")
            return WorkflowResult(status=WorkflowRunStatus.SUCCEEDED)

    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert ingestion.message_id is not None
    outcome = await ProcessOwnerMessageService(
        TrackingFactory(), WorkflowRouter([GuardHandler()]), GuardResolver()
    ).process(ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE)
    assert outcome.status is ProcessingStatus.COMPLETED
    assert state["violation"] is None


@pytest.mark.asyncio
async def test_reserved_commands_are_rejected_during_processing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    ingestion = await IngestOwnerMessageService(SqlUnitOfWorkFactory(session_factory)).ingest(
        make_message(business_id)
    )
    assert ingestion.message_id is not None
    with pytest.raises(ReservedOutboxCommandTypeError):
        await ProcessOwnerMessageService(
            SqlUnitOfWorkFactory(session_factory),
            WorkflowRouter([ReservedHandler()]),
            EchoResolver(),
        ).process(ingestion.message_id, now=TEST_NOW, stale_before=TEST_STALE_BEFORE)
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 0


@pytest.mark.asyncio
async def test_cross_tenant_references_are_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first_business = BusinessId(uuid4())
    second_business = BusinessId(uuid4())
    await seed_business(session_factory, first_business)
    await seed_business(session_factory, second_business)
    first_message = make_message(first_business)
    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        endpoint_id = await unit_of_work.owner_channel_endpoints.get_or_create(
            first_message.endpoint, first_message.routing
        )
        conversation_id = await unit_of_work.conversations.get_or_create(
            first_message.message.conversation_ref, endpoint_id, first_message.routing
        )
        inbound_id = await unit_of_work.inbound_messages.create(
            first_message, conversation_id, endpoint_id
        )
        assert inbound_id is not None
        await unit_of_work.commit()
    mismatched = make_message(second_business)
    with pytest.raises(CrossBusinessReferenceError):
        async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
            await unit_of_work.inbound_messages.create(mismatched, conversation_id, endpoint_id)
    with pytest.raises(CrossBusinessReferenceError):
        async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
            await unit_of_work.outbound_messages.create(
                OutboundOwnerMessage(
                    business_id=second_business,
                    conversation_ref=ConversationRef(
                        business_id=second_business,
                        external_conversation_id="conversation",
                    ),
                    parts=(TextPart(text="bad"),),
                    correlation_id="bad",
                ),
                conversation_id,
                inbound_id,
            )
    async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
        outbound_id = await unit_of_work.outbound_messages.create(
            OutboundOwnerMessage(
                business_id=first_business,
                conversation_ref=first_message.message.conversation_ref,
                parts=(TextPart(text="valid"),),
                correlation_id="valid",
            ),
            conversation_id,
            inbound_id,
        )
        await unit_of_work.commit()
    with pytest.raises(CrossBusinessReferenceError):
        async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
            await unit_of_work.outbox.enqueue(owner_reply_command(second_business, outbound_id))
    with pytest.raises(CrossBusinessReferenceError):
        async with SqlUnitOfWorkFactory(session_factory)() as unit_of_work:
            await unit_of_work.outbox.enqueue(
                owner_message_process_command(second_business, inbound_id)
            )
    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
