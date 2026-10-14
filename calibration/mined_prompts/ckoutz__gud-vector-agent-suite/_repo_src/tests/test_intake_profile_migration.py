"""Migration 0017 round trip on PostgreSQL: intake ``problem`` becomes
``details`` on upgrade and comes back (with notes folded in) on downgrade."""

import asyncio
import importlib.util
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.ext.asyncio import create_async_engine

import gvas.infrastructure as infrastructure_package
from gvas.interfaces.migrate import build_config

BEFORE = "0016_intake_booking_event_type"
AFTER = "0017_business_intake_profile"
NOW = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)

businesses = sa.table(
    "businesses",
    sa.column("id", sa.Uuid()),
    sa.column("slug", sa.String()),
    sa.column("name", sa.String()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)
conversations = sa.table(
    "intake_conversations",
    sa.column("id", sa.Uuid()),
    sa.column("business_id", sa.Uuid()),
    sa.column("reference", sa.String()),
    sa.column("token_hash", sa.String()),
    sa.column("channel", sa.String()),
    sa.column("state", sa.String()),
    sa.column("collected", sa.JSON()),
    sa.column("expires_at", sa.DateTime(timezone=True)),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("updated_at", sa.DateTime(timezone=True)),
)


async def _run(database_url: str, statement: Any) -> list[Any]:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            result = await connection.execute(statement)
            return list(result.all()) if result.returns_rows else []
    finally:
        await engine.dispose()


def _seed(database_url: str, business_id: UUID, conversation_id: UUID) -> None:
    asyncio.run(
        _run(
            database_url,
            businesses.insert().values(
                id=business_id,
                slug=f"migration-{business_id}",
                name="Migration Co",
                created_at=NOW,
                updated_at=NOW,
            ),
        )
    )
    asyncio.run(
        _run(
            database_url,
            conversations.insert().values(
                id=conversation_id,
                business_id=business_id,
                reference="mig0017",
                token_hash="0" * 64,
                channel="web",
                state="collecting",
                collected={"name": "Jane", "address": "2 Elm St", "problem": "ants"},
                expires_at=NOW + timedelta(days=1),
                created_at=NOW,
                updated_at=NOW,
            ),
        )
    )


def _collected(database_url: str, conversation_id: UUID) -> dict[str, Any]:
    rows = asyncio.run(
        _run(
            database_url,
            sa.select(conversations.c.collected).where(conversations.c.id == conversation_id),
        )
    )
    value = rows[0][0]
    assert isinstance(value, dict)
    return value


def test_intake_profile_migration_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = os.getenv("GVAS_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("GVAS_TEST_DATABASE_URL is not set")
    monkeypatch.setenv("GVAS_DATABASE_URL", database_url)
    config = build_config()
    business_id, conversation_id = uuid4(), uuid4()
    command.upgrade(config, BEFORE)
    try:
        _seed(database_url, business_id, conversation_id)

        command.upgrade(config, AFTER)
        upgraded = _collected(database_url, conversation_id)
        assert upgraded == {"name": "Jane", "address": "2 Elm St", "details": "ants", "notes": None}
        asyncio.run(
            _run(
                database_url,
                sa.text(
                    "UPDATE businesses SET intake_brief = 'b', intake_questions = 'q', "
                    "intake_opening = 'o' WHERE id = :id"
                ).bindparams(id=business_id),
            )
        )
        asyncio.run(
            _run(
                database_url,
                conversations.update()
                .where(conversations.c.id == conversation_id)
                .values(collected={**upgraded, "notes": "basement too"}),
            )
        )

        command.downgrade(config, BEFORE)
        downgraded = _collected(database_url, conversation_id)
        assert downgraded == {
            "name": "Jane",
            "address": "2 Elm St",
            "problem": "ants — basement too",
        }

        command.upgrade(config, AFTER)
        assert _collected(database_url, conversation_id)["details"] == "ants — basement too"
    finally:
        command.downgrade(config, "base")


def _migration_module() -> Any:
    path = Path(infrastructure_package.__file__).parent / "migrations" / "versions" / f"{AFTER}.py"
    spec = importlib.util.spec_from_file_location(AFTER, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downgrade_keeps_notes_when_details_fill_the_old_limit() -> None:
    migration = _migration_module()
    limit = migration.TEXT_MAX_CHARS
    folded = migration._details_to_problem({"details": "d" * limit, "notes": "n" * 50})
    problem = folded["problem"]
    assert len(problem) == limit
    assert problem.endswith(" — " + "n" * 50)
    long_notes = migration._details_to_problem({"details": "d" * 10, "notes": "n" * limit})
    assert long_notes["problem"].startswith("d" * 10 + " — n")
    assert len(long_notes["problem"]) == limit
    assert migration._details_to_problem({"details": None, "notes": "n"})["problem"] == "n"
    assert migration._details_to_problem({"details": None, "notes": None})["problem"] is None
