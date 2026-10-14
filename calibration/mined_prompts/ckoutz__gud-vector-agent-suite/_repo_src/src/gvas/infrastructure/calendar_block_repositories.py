from datetime import UTC, date, datetime
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gvas.domain.calendar_blocks import CalendarBlock, CalendarBlockState, DayHours
from gvas.domain.identifiers import BusinessId, JsonValue
from gvas.infrastructure.calendar_block_models import CalendarBlockRecord


class SqlCalendarBlockRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, block: CalendarBlock) -> None:
        row = CalendarBlockRecord(id=block.block_id, business_id=block.business_id)
        _write(row, block)
        self.session.add(row)
        await self.session.flush()

    async def save(self, block: CalendarBlock) -> None:
        row = await self.session.get(CalendarBlockRecord, block.block_id)
        if row is None or row.business_id != block.business_id:
            raise LookupError("calendar block is not persisted")
        _write(row, block)
        await self.session.flush()

    async def latest_proposed(self, business_id: BusinessId) -> CalendarBlock | None:
        row = await self.session.scalar(
            select(CalendarBlockRecord)
            .where(
                CalendarBlockRecord.business_id == business_id,
                CalendarBlockRecord.state == CalendarBlockState.PROPOSED.value,
            )
            .order_by(CalendarBlockRecord.created_at.desc())
            .limit(1)
        )
        return None if row is None else _read(row)

    async def applied_on(self, business_id: BusinessId, day: date) -> tuple[CalendarBlock, ...]:
        rows = await self.session.scalars(
            select(CalendarBlockRecord)
            .where(
                CalendarBlockRecord.business_id == business_id,
                CalendarBlockRecord.day == day,
                CalendarBlockRecord.state == CalendarBlockState.APPLIED.value,
            )
            .order_by(CalendarBlockRecord.created_at, CalendarBlockRecord.id)
        )
        return tuple(_read(row) for row in rows.all())


def _write(row: CalendarBlockRecord, block: CalendarBlock) -> None:
    row.day = block.day
    row.start_minute = block.start_minute
    row.end_minute = block.end_minute
    row.reference = block.reference
    row.state = block.state.value
    row.previous = (
        None
        if block.previous is None
        else cast(dict[str, JsonValue], block.previous.model_dump(mode="json"))
    )
    row.created_at = block.created_at
    row.expires_at = block.expires_at
    row.decided_at = block.decided_at


def _read(row: CalendarBlockRecord) -> CalendarBlock:
    return CalendarBlock(
        block_id=row.id,
        business_id=BusinessId(row.business_id),
        day=row.day,
        start_minute=row.start_minute,
        end_minute=row.end_minute,
        reference=row.reference,
        state=CalendarBlockState(row.state),
        previous=None if row.previous is None else DayHours.model_validate(row.previous),
        created_at=_aware(row.created_at),
        expires_at=_aware(row.expires_at),
        decided_at=None if row.decided_at is None else _aware(row.decided_at),
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
