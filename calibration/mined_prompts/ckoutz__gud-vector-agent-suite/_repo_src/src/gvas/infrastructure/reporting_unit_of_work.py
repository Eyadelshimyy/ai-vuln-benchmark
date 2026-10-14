from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gvas.domain.reporting import FieldNotesReportRepository, ReportUnitOfWork
from gvas.infrastructure.reporting_repositories import SqlFieldNotesReportRepository


class SqlReportUnitOfWork:
    reports: FieldNotesReportRepository

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> "SqlReportUnitOfWork":
        self._session = self._session_factory()
        self.reports = SqlFieldNotesReportRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            await self._session.rollback()
        await self._session.close()

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work is not active")
        await self._session.commit()


class SqlReportUnitOfWorkFactory:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> ReportUnitOfWork:
        return SqlReportUnitOfWork(self._session_factory)
