from datetime import datetime, timedelta
from typing import Protocol

from gvas.domain.enums import OutboxStatus
from gvas.domain.outbox import OutboxCommand, OutboxRecord
from gvas.domain.repositories import UnitOfWork


class OutboxService:
    def __init__(self, unit_of_work_factory: "OutboxUnitOfWorkFactory") -> None:
        self._unit_of_work_factory = unit_of_work_factory

    async def enqueue(self, command: OutboxCommand) -> None:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.outbox.enqueue(command)
            await unit_of_work.commit()

    async def claim_batch(
        self, limit: int, now: datetime, claimed_by: str, *, stale_before: datetime
    ) -> list[OutboxRecord]:
        async with self._unit_of_work_factory() as unit_of_work:
            records = await unit_of_work.outbox.claim_batch(
                limit, now, claimed_by, stale_before=stale_before
            )
            await unit_of_work.commit()
            return records

    async def mark_succeeded(self, record: OutboxRecord) -> OutboxRecord:
        return await self._update(record.transition(OutboxStatus.SUCCEEDED))

    async def mark_failed(
        self, record: OutboxRecord, retry_in: timedelta, error: str, now: datetime
    ) -> OutboxRecord:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        status = (
            OutboxStatus.DEAD if record.attempts >= record.max_attempts else OutboxStatus.FAILED
        )
        updated = record.model_copy(
            update={
                "available_at": now + retry_in,
            }
        ).transition(status, error=error)
        return await self._update(updated)

    async def mark_dead(self, record: OutboxRecord, error: str | None = None) -> OutboxRecord:
        updated = record.model_copy(
            update={"attempts": max(record.attempts, record.max_attempts)}
        ).transition(OutboxStatus.DEAD, error=error)
        return await self._update(updated)

    async def _update(self, record: OutboxRecord) -> OutboxRecord:
        async with self._unit_of_work_factory() as unit_of_work:
            await unit_of_work.outbox.update(record)
            await unit_of_work.commit()
        return record


class OutboxUnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...
