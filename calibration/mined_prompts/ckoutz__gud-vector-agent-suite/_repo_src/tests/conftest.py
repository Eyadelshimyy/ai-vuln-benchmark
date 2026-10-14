import os
from collections.abc import AsyncIterator
from typing import Protocol

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from gvas.infrastructure.models import Base


class _SQLiteConnection(Protocol):
    isolation_level: str | None


class _SQLAlchemyConnection(Protocol):
    def exec_driver_sql(self, statement: str) -> object: ...


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine: AsyncEngine = create_async_engine("sqlite+aiosqlite:///:memory:", isolation_level=None)

    @event.listens_for(engine.sync_engine, "connect")
    def do_connect(dbapi_connection: _SQLiteConnection, connection_record: object) -> None:
        dbapi_connection.isolation_level = None

    @event.listens_for(engine.sync_engine, "begin")
    def do_begin(connection: _SQLAlchemyConnection) -> None:
        connection.exec_driver_sql("BEGIN")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def postgres_session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    database_url = os.getenv("GVAS_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("GVAS_TEST_DATABASE_URL is not set")
    engine: AsyncEngine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        yield factory
    finally:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)
        await engine.dispose()
