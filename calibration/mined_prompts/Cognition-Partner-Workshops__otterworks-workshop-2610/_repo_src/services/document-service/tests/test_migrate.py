"""Boot-time migrations only adopt a legacy schema that is actually complete."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from app.db import migrate
from app.db.base import Base
from app.models.document import Document


@pytest.fixture
def alembic_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(migrate.command, "stamp", lambda cfg, rev: calls.append(f"stamp {rev}"))
    monkeypatch.setattr(migrate.command, "upgrade", lambda cfg, rev: calls.append(f"upgrade {rev}"))
    return calls


@pytest.mark.asyncio
async def test_partial_unversioned_schema_refuses_to_stamp(alembic_calls: list[str]) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync: Document.__table__.create(sync))

    with pytest.raises(RuntimeError, match="partial schema"):
        await migrate.upgrade_to_head(engine)
    assert alembic_calls == []


@pytest.mark.asyncio
async def test_complete_legacy_schema_is_stamped_then_upgraded(alembic_calls: list[str]) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await migrate.upgrade_to_head(engine)
    assert alembic_calls == [f"stamp {migrate.LEGACY_BASELINE}", "upgrade head"]


@pytest.mark.asyncio
async def test_empty_database_just_upgrades(alembic_calls: list[str]) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    await migrate.upgrade_to_head(engine)
    assert alembic_calls == ["upgrade head"]
