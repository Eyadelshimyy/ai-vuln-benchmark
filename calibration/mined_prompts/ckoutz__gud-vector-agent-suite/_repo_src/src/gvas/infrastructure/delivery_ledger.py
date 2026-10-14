"""Durable delivery ledger shared by the web and worker processes.

Outbound delivery is at-least-once, not exactly-once, and this module does not
pretend otherwise. The channel APIs used here accept no idempotency key, so a
process that dies after the provider accepts a post but before the receipt
commits will post again on replay. The ledger closes that window for every
crash outside this single gap; adapters additionally attach the delivery key to
the provider message where the provider supports it, so a duplicate can be
identified afterwards.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.enums import DeliveryStatus
from gvas.domain.messages import DeliveryReceipt
from gvas.infrastructure.delivery_models import ChannelDeliveryReceipt


class SqlChannelDeliveryLedger:
    """Records receipts in their own transaction so a caller rollback cannot lose one."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def find(self, key: str) -> DeliveryReceipt | None:
        async with self._session_factory() as session:
            row = await session.get(ChannelDeliveryReceipt, key)
            if row is None:
                return None
            return _receipt_of(row)

    async def record(self, key: str, receipt: DeliveryReceipt) -> None:
        async with self._session_factory() as session:
            session.add(
                ChannelDeliveryReceipt(
                    delivery_key=key,
                    status=receipt.status.value,
                    provider_message_id=receipt.provider_message_id,
                    occurred_at=receipt.occurred_at,
                    detail=receipt.detail,
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await session.scalar(
                    select(ChannelDeliveryReceipt).where(ChannelDeliveryReceipt.delivery_key == key)
                )
                if existing is None:
                    raise


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _receipt_of(row: ChannelDeliveryReceipt) -> DeliveryReceipt:
    return DeliveryReceipt(
        status=DeliveryStatus(row.status),
        provider_message_id=row.provider_message_id,
        occurred_at=_as_utc(row.occurred_at),
        detail=row.detail,
    )
