from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.application.outbox_service import OutboxService
from gvas.domain.enums import OutboxStatus
from gvas.domain.identifiers import BusinessId, OutboxCommandId
from gvas.domain.outbox import (
    InvalidOutboxTransitionError,
    LostOutboxLeaseError,
    OutboxCommand,
    OutboxRecord,
)
from gvas.infrastructure.models import Business, OutboxMessage
from gvas.infrastructure.repositories import SqlOutboxRepository
from gvas.infrastructure.unit_of_work import SqlUnitOfWorkFactory


def command(business_id: BusinessId, dedup_key: str) -> OutboxCommand:
    return OutboxCommand(
        command_id=OutboxCommandId(uuid4()),
        business_id=business_id,
        command_type="custom",
        payload={"value": "x"},
        dedup_key=dedup_key,
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
async def test_claim_deduplicates_and_records_lock(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    now = datetime.now(UTC)
    async with session_factory() as session:
        repository = SqlOutboxRepository(session)
        item = command(business_id, "dedup")
        await repository.enqueue(item)
        await repository.enqueue(item)
        claimed = await repository.claim_batch(
            10,
            now + timedelta(seconds=1),
            "worker",
            stale_before=now,
        )
        assert len(claimed) == 1
        assert claimed[0].attempts == 1
        row = await session.scalar(select(OutboxMessage))
        assert row is not None
        assert row.locked_by == "worker"
        assert row.locked_at == (now + timedelta(seconds=1)).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_claim_reclaims_stale_in_progress_lease(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    service = OutboxService(SqlUnitOfWorkFactory(session_factory))
    now = datetime.now(UTC)
    await service.enqueue(command(business_id, "lease"))

    first = (
        await service.claim_batch(1, now + timedelta(seconds=1), "worker-a", stale_before=now)
    )[0]
    fresh = await service.claim_batch(
        1,
        now + timedelta(seconds=2),
        "worker-b",
        stale_before=now,
    )
    assert fresh == []

    reclaimed = (
        await service.claim_batch(
            1, now + timedelta(seconds=2), "worker-b", stale_before=now + timedelta(seconds=1)
        )
    )[0]
    assert reclaimed.command.command_id == first.command.command_id
    assert reclaimed.attempts == 2
    async with session_factory() as session:
        row = await session.scalar(select(OutboxMessage))
        assert row is not None
        assert row.locked_by == "worker-b"
        assert row.locked_at == (now + timedelta(seconds=2)).replace(tzinfo=None)


@pytest.mark.asyncio
async def test_deduplication_is_scoped_to_business(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    first_business = BusinessId(uuid4())
    second_business = BusinessId(uuid4())
    await seed_business(session_factory, first_business)
    await seed_business(session_factory, second_business)
    service = OutboxService(SqlUnitOfWorkFactory(session_factory))

    await service.enqueue(command(first_business, "shared"))
    await service.enqueue(command(second_business, "shared"))
    await service.enqueue(command(first_business, "shared"))

    async with session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(OutboxMessage)) == 2


@pytest.mark.asyncio
async def test_service_retry_backoff_counts_attempts_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    service = OutboxService(SqlUnitOfWorkFactory(session_factory))
    now = datetime.now(UTC)
    await service.enqueue(command(business_id, "retry"))

    first_claim = (
        await service.claim_batch(1, now + timedelta(seconds=1), "worker", stale_before=now)
    )[0]
    first = await service.mark_failed(
        first_claim, timedelta(seconds=10), "temporary", now + timedelta(seconds=1)
    )
    assert first.status is OutboxStatus.FAILED
    assert first.attempts == 1
    assert first.available_at == now + timedelta(seconds=11)

    second_claim = (
        await service.claim_batch(
            1, now + timedelta(seconds=11), "worker", stale_before=now + timedelta(seconds=1)
        )
    )[0]
    second = await service.mark_failed(
        second_claim, timedelta(seconds=10), "temporary", now + timedelta(seconds=11)
    )
    assert second.status is OutboxStatus.FAILED
    assert second.attempts == 2

    third_claim = (
        await service.claim_batch(
            1, now + timedelta(seconds=21), "worker", stale_before=now + timedelta(seconds=11)
        )
    )[0]
    third = await service.mark_failed(
        third_claim, timedelta(seconds=10), "permanent", now + timedelta(seconds=21)
    )
    assert third.status is OutboxStatus.DEAD
    assert third.attempts == 3


@pytest.mark.asyncio
async def test_service_success_dead_and_invalid_transitions(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    service = OutboxService(SqlUnitOfWorkFactory(session_factory))
    now = datetime.now(UTC)
    await service.enqueue(command(business_id, "success"))
    claimed = (
        await service.claim_batch(1, now + timedelta(seconds=1), "worker", stale_before=now)
    )[0]
    succeeded = await service.mark_succeeded(claimed)
    assert succeeded.status is OutboxStatus.SUCCEEDED
    with pytest.raises(InvalidOutboxTransitionError):
        await service.mark_succeeded(succeeded)

    await service.enqueue(command(business_id, "manual-dead"))
    claimed = (
        await service.claim_batch(1, now + timedelta(seconds=1), "worker", stale_before=now)
    )[0]
    dead = await service.mark_dead(claimed, "cancelled")
    assert dead.status is OutboxStatus.DEAD


@pytest.mark.asyncio
async def test_stale_outbox_claim_cannot_update_reclaimed_row(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    business_id = BusinessId(uuid4())
    await seed_business(session_factory, business_id)
    service = OutboxService(SqlUnitOfWorkFactory(session_factory))
    now = datetime.now(UTC)
    await service.enqueue(command(business_id, "fenced"))

    first = (
        await service.claim_batch(1, now + timedelta(seconds=1), "worker-a", stale_before=now)
    )[0]
    reclaimed = (
        await service.claim_batch(
            1,
            now + timedelta(seconds=2),
            "worker-b",
            stale_before=now + timedelta(seconds=1),
        )
    )[0]

    with pytest.raises(LostOutboxLeaseError):
        await service.mark_succeeded(first)

    async with session_factory() as session:
        row = await session.scalar(select(OutboxMessage))
        assert row is not None
        assert row.status == OutboxStatus.IN_PROGRESS.value
        assert row.attempts == reclaimed.attempts == 2
        assert row.locked_by == "worker-b"


def test_outbox_domain_rejects_invalid_transition() -> None:
    record = OutboxRecord(
        command=OutboxCommand(
            command_id=OutboxCommandId(uuid4()),
            business_id=BusinessId(uuid4()),
            command_type="custom",
            payload={},
        ),
        available_at=datetime.now(UTC),
    )
    with pytest.raises(InvalidOutboxTransitionError):
        record.transition(OutboxStatus.SUCCEEDED)
