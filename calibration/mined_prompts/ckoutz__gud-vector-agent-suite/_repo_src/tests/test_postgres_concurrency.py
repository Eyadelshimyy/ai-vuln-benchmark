import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.ingestion import IngestOwnerMessageService
from gvas.application.outbox_service import OutboxService
from gvas.application.processing import ProcessingStatus, ProcessOwnerMessageService
from gvas.domain.enums import OutboxStatus, WorkflowRunStatus
from gvas.domain.identifiers import BusinessId, MessageKey, OutboxCommandId, WorkflowIntent
from gvas.domain.intents import IntentResolution
from gvas.domain.messages import (
    ChannelEndpointRef,
    ConversationRef,
    InboundOwnerMessage,
    NormalizedOwnerMessage,
    OutboundOwnerMessage,
    TextPart,
)
from gvas.domain.outbox import LostOutboxLeaseError, OutboxCommand
from gvas.domain.workflows import WorkflowContext, WorkflowResult, WorkflowRouter
from gvas.infrastructure.models import (
    Business,
    InboundMessage,
    OutboundMessage,
    OutboxMessage,
    WorkflowRun,
)
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_processing_claim_is_exclusive(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(postgres_session_factory, business_id)
    ingestion = await IngestOwnerMessageService(
        SqlUnitOfWorkFactory(postgres_session_factory)
    ).ingest(make_message(business_id))
    assert ingestion.message_id is not None

    entered = asyncio.Event()
    release = asyncio.Event()

    class Resolver:
        async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
            return IntentResolution(intent=WorkflowIntent("echo"))

    class Handler:
        intent = WorkflowIntent("echo")
        calls = 0

        async def handle(self, context: WorkflowContext) -> WorkflowResult:
            self.calls += 1
            entered.set()
            await release.wait()
            return WorkflowResult(
                status=WorkflowRunStatus.SUCCEEDED,
                replies=(
                    OutboundOwnerMessage(
                        business_id=context.message.business_id,
                        conversation_ref=context.message.conversation_ref,
                        parts=(TextPart(text="reply"),),
                        correlation_id="concurrent",
                    ),
                ),
            )

    handler = Handler()
    service = ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(postgres_session_factory), WorkflowRouter([handler]), Resolver()
    )
    now = datetime(2025, 1, 1, tzinfo=UTC)
    first_task = asyncio.create_task(
        service.process(
            ingestion.message_id,
            now=now,
            stale_before=now - timedelta(days=1),
        )
    )
    await entered.wait()
    second = await service.process(
        ingestion.message_id,
        now=now,
        stale_before=now - timedelta(days=1),
    )
    release.set()
    first = await first_task

    assert first.status is ProcessingStatus.COMPLETED
    assert second.status is ProcessingStatus.BUSY
    assert handler.calls == 1
    async with postgres_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(WorkflowRun)) == 1
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboxMessage)
                .where(OutboxMessage.outbound_message_id.is_not(None))
            )
            == 1
        )
        process_command = await session.scalar(
            select(OutboxMessage).where(OutboxMessage.inbound_message_id == ingestion.message_id)
        )
        assert process_command is not None

        await session.execute(
            delete(InboundMessage).where(InboundMessage.id == ingestion.message_id)
        )
        await session.commit()
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboxMessage)
                .where(OutboxMessage.inbound_message_id == ingestion.message_id)
            )
            == 0
        )


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_workflow_lease_reclaims_only_stale_runs(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(postgres_session_factory, business_id)
    ingress = IngestOwnerMessageService(SqlUnitOfWorkFactory(postgres_session_factory))
    stale_ingestion = await ingress.ingest(make_message(business_id, message_key="stale"))
    recent_ingestion = await ingress.ingest(make_message(business_id, message_key="recent"))
    assert stale_ingestion.message_id is not None
    assert recent_ingestion.message_id is not None

    old_now = datetime(2024, 1, 1, tzinfo=UTC)
    recent_now = datetime(2025, 1, 1, tzinfo=UTC)
    uow_factory = SqlUnitOfWorkFactory(postgres_session_factory)
    async with uow_factory() as unit_of_work:
        await unit_of_work.workflow_runs.claim(
            business_id,
            stale_ingestion.message_id,
            now=old_now,
            stale_before=old_now - timedelta(days=1),
        )
        await unit_of_work.workflow_runs.claim(
            business_id,
            recent_ingestion.message_id,
            now=recent_now,
            stale_before=recent_now - timedelta(days=1),
        )
        await unit_of_work.commit()

    class Resolver:
        async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
            return IntentResolution(intent=WorkflowIntent("echo"))

    class Handler:
        intent = WorkflowIntent("echo")

        async def handle(self, context: WorkflowContext) -> WorkflowResult:
            return WorkflowResult(status=WorkflowRunStatus.SUCCEEDED)

    service = ProcessOwnerMessageService(uow_factory, WorkflowRouter([Handler()]), Resolver())
    stale = await service.process(
        stale_ingestion.message_id,
        now=recent_now,
        stale_before=old_now + timedelta(days=1),
    )
    recent = await service.process(
        recent_ingestion.message_id,
        now=recent_now,
        stale_before=recent_now - timedelta(days=1),
    )
    assert stale.status is ProcessingStatus.COMPLETED
    assert recent.status is ProcessingStatus.BUSY


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_stale_workflow_cannot_finish_after_reclaim(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(postgres_session_factory, business_id)
    ingestion = await IngestOwnerMessageService(
        SqlUnitOfWorkFactory(postgres_session_factory)
    ).ingest(make_message(business_id))
    assert ingestion.message_id is not None

    entered = asyncio.Event()
    release_first = asyncio.Event()
    entered_second = asyncio.Event()
    release_second = asyncio.Event()

    class Resolver:
        async def resolve(self, message: NormalizedOwnerMessage) -> IntentResolution:
            return IntentResolution(intent=WorkflowIntent("echo"))

    class Handler:
        intent = WorkflowIntent("echo")
        calls = 0

        async def handle(self, context: WorkflowContext) -> WorkflowResult:
            self.calls += 1
            if self.calls == 1:
                entered.set()
                await release_first.wait()
            else:
                entered_second.set()
                await release_second.wait()
            return WorkflowResult(
                status=WorkflowRunStatus.SUCCEEDED,
                replies=(
                    OutboundOwnerMessage(
                        business_id=context.message.business_id,
                        conversation_ref=context.message.conversation_ref,
                        parts=(TextPart(text="reply"),),
                        correlation_id="fenced",
                    ),
                ),
            )

    handler = Handler()
    service = ProcessOwnerMessageService(
        SqlUnitOfWorkFactory(postgres_session_factory), WorkflowRouter([handler]), Resolver()
    )
    first_task = asyncio.create_task(
        service.process(
            ingestion.message_id,
            now=datetime(2025, 1, 1, tzinfo=UTC),
            stale_before=datetime(2024, 1, 1, tzinfo=UTC),
        )
    )
    await entered.wait()
    second_task = asyncio.create_task(
        service.process(
            ingestion.message_id,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            stale_before=datetime(2025, 1, 2, tzinfo=UTC),
        )
    )
    await entered_second.wait()
    release_first.set()
    first = await first_task
    release_second.set()
    second = await second_task

    assert first.status is ProcessingStatus.LEASE_LOST
    assert second.status is ProcessingStatus.COMPLETED
    assert handler.calls == 2
    async with postgres_session_factory() as session:
        run = await session.scalar(select(WorkflowRun))
        assert run is not None
        assert run.attempts == 2
        assert run.status == WorkflowRunStatus.SUCCEEDED.value
        assert await session.scalar(select(func.count()).select_from(OutboundMessage)) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(OutboxMessage)
                .where(OutboxMessage.outbound_message_id.is_not(None))
            )
            == 1
        )


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_stale_outbox_cannot_update_after_reclaim(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(postgres_session_factory, business_id)
    service = OutboxService(SqlUnitOfWorkFactory(postgres_session_factory))
    await service.enqueue(
        OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=business_id,
            command_type="custom",
            payload={"value": "x"},
            dedup_key="fenced",
        )
    )
    first_now = datetime.now(UTC) + timedelta(days=1)
    first = (
        await service.claim_batch(
            1,
            first_now,
            "worker-a",
            stale_before=first_now - timedelta(days=1),
        )
    )[0]
    second_now = first_now + timedelta(days=1)
    second = (
        await service.claim_batch(
            1,
            second_now,
            "worker-b",
            stale_before=first_now + timedelta(hours=1),
        )
    )[0]

    with pytest.raises(LostOutboxLeaseError):
        await service.mark_succeeded(first)

    assert second.attempts == 2
    async with postgres_session_factory() as session:
        row = await session.scalar(select(OutboxMessage))
        assert row is not None
        assert row.status == OutboxStatus.IN_PROGRESS.value
        assert row.attempts == 2
        assert row.locked_by == "worker-b"


async def seed_business(
    session_factory: async_sessionmaker[AsyncSession], business_id: BusinessId
) -> None:
    now = datetime.now(UTC)
    async with session_factory() as session:
        session.add(
            Business(
                id=business_id,
                slug=f"postgres-{business_id}",
                name="Postgres",
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


def make_message(business_id: BusinessId, *, message_key: str = "message") -> InboundOwnerMessage:
    normalized = NormalizedOwnerMessage(
        message_key=MessageKey(message_key),
        business_id=business_id,
        conversation_ref=ConversationRef(
            business_id=business_id, external_conversation_id="conversation"
        ),
        sender={"external_id": "owner", "role": "owner"},
        received_at=datetime.now(UTC),
        parts=(TextPart(text="hello"),),
    )
    return InboundOwnerMessage(
        message=normalized,
        endpoint=ChannelEndpointRef(
            business_id=business_id,
            source_namespace="postgres",
            external_endpoint_id="endpoint",
        ),
        routing={"endpoint": "endpoint"},
    )
