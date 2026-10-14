"""Durable monthly usage totals shared by the web and worker processes.

Each recording runs in its own short transaction so a caller's rollback cannot
lose it and a provider call that already happened is always counted. Two
adapters recording the same month concurrently serialize on the row lock (or on
the primary key when both insert), so the running total never loses an update.
"""

from datetime import date, datetime
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult, Result
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.identifiers import BusinessId
from gvas.domain.usage import UsageKind, usage_month
from gvas.infrastructure.usage_models import UsageLedgerMonth


class SqlUsageLedger:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def record(
        self, business_id: BusinessId, kind: UsageKind, units: int, *, at: datetime
    ) -> None:
        if units < 0:
            raise ValueError("usage units must not be negative")
        if units == 0:
            return
        month = usage_month(at)
        async with self._session_factory() as session:
            result: Result[Any] = await session.execute(
                update(UsageLedgerMonth)
                .where(
                    UsageLedgerMonth.business_id == business_id,
                    UsageLedgerMonth.kind == kind.value,
                    UsageLedgerMonth.month == month,
                )
                .values(units=UsageLedgerMonth.units + units, updated_at=at)
            )
            if cast(CursorResult[Any], result).rowcount == 1:
                await session.commit()
                return
            session.add(
                UsageLedgerMonth(
                    business_id=business_id,
                    kind=kind.value,
                    month=month,
                    units=units,
                    updated_at=at,
                )
            )
            try:
                await session.commit()
                return
            except IntegrityError:
                await session.rollback()
            await session.execute(
                update(UsageLedgerMonth)
                .where(
                    UsageLedgerMonth.business_id == business_id,
                    UsageLedgerMonth.kind == kind.value,
                    UsageLedgerMonth.month == month,
                )
                .values(units=UsageLedgerMonth.units + units, updated_at=at)
            )
            await session.commit()

    async def total(self, business_id: BusinessId, kind: UsageKind, *, month: date) -> int:
        async with self._session_factory() as session:
            units = await session.scalar(
                select(UsageLedgerMonth.units).where(
                    UsageLedgerMonth.business_id == business_id,
                    UsageLedgerMonth.kind == kind.value,
                    UsageLedgerMonth.month == month,
                )
            )
        return int(units or 0)
