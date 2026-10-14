"""Durable delivery ledger shared by the web and worker processes."""

import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.enums import DeliveryStatus
from gvas.domain.messages import DeliveryReceipt
from gvas.infrastructure.delivery_ledger import SqlChannelDeliveryLedger
from gvas.infrastructure.delivery_models import ChannelDeliveryReceipt

KEY = "owner-reply:1"
NOW = datetime(2026, 3, 4, 5, 6, tzinfo=UTC)


def receipt(provider_message_id: str) -> DeliveryReceipt:
    return DeliveryReceipt(
        status=DeliveryStatus.DELIVERED,
        provider_message_id=provider_message_id,
        occurred_at=NOW,
    )


@pytest.mark.asyncio
async def test_recorded_receipt_is_visible_to_another_process(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await SqlChannelDeliveryLedger(session_factory).record(KEY, receipt("m-1"))

    found = await SqlChannelDeliveryLedger(session_factory).find(KEY)

    assert found is not None
    assert found.provider_message_id == "m-1"
    assert found.occurred_at == NOW
    assert found.status is DeliveryStatus.DELIVERED


@pytest.mark.asyncio
async def test_missing_key_has_no_receipt(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    assert await SqlChannelDeliveryLedger(session_factory).find(KEY) is None


@pytest.mark.asyncio
async def test_replayed_record_keeps_the_first_receipt(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ledger = SqlChannelDeliveryLedger(session_factory)

    await ledger.record(KEY, receipt("m-1"))
    await ledger.record(KEY, receipt("m-2"))

    found = await ledger.find(KEY)
    assert found is not None
    assert found.provider_message_id == "m-1"


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_concurrent_writers_keep_the_first_receipt(
    postgres_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    session_factory = postgres_session_factory
    first = SqlChannelDeliveryLedger(session_factory)
    second = SqlChannelDeliveryLedger(session_factory)

    await asyncio.gather(first.record(KEY, receipt("m-1")), second.record(KEY, receipt("m-2")))

    async with session_factory() as session:
        rows = list((await session.scalars(select(ChannelDeliveryReceipt))).all())
        assert await session.scalar(select(func.count()).select_from(ChannelDeliveryReceipt)) == 1
    assert rows[0].provider_message_id in {"m-1", "m-2"}
