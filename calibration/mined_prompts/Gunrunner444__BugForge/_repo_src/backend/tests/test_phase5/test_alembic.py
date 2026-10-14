"""Alembic model discovery and upgrade checks.

Full upgrade/downgrade of historical revisions uses PostgreSQL. SQLite cannot
ALTER constraints (revision 008) so the complete chain is skipped there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

import app.models  # noqa: F401 — register HackerOne and agent tables
from app.models.base import Base


def _alembic_config(url: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_alembic_env_discovers_hackerone_and_agent_models() -> None:
    table_names = set(Base.metadata.tables)
    assert "hackerone_programs" in table_names
    assert "hackerone_report_intents" in table_names
    assert "hackerone_attachments" in table_names
    assert "hackerone_report_drafts" in table_names
    assert "security_research_sessions" in table_names
    assert "research_evidence_node" in table_names
    assert "research_evidence_edge" in table_names
    assert "security_research_hypotheses" in table_names
    assert "security_research_tool_calls" in table_names
    assert "security_reproduction_plans" in table_names
    assert "research_findings" in table_names
    assert "research_memory" in table_names
    assert "research_checkpoints" in table_names
    assert "research_identities" in table_names
    assert "security_research_projects" in table_names
    intent = Base.metadata.tables["hackerone_report_intents"]
    assert "local_status" in intent.c
    assert "remote_intent_id" in intent.c
    attachment = Base.metadata.tables["hackerone_attachments"]
    assert "review_state" in attachment.c
    assert "sha256" in attachment.c
    program = Base.metadata.tables["hackerone_programs"]
    assert "scope_content_hash" in program.c


def test_alembic_revision_chain_includes_020() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("020")
    assert revision is not None
    assert revision.down_revision == "019"


def test_alembic_revision_chain_includes_021() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("021")
    assert revision is not None
    assert revision.down_revision == "020"


def test_alembic_revision_chain_includes_022() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("022")
    assert revision is not None
    assert revision.down_revision == "021"


def test_alembic_revision_chain_includes_023() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("023")
    assert revision is not None
    assert revision.down_revision == "022"


def test_alembic_revision_chain_includes_025() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("025")
    assert revision is not None
    assert revision.down_revision == "024"


def test_alembic_revision_chain_includes_024() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert heads == ["028"]
    revision = script.get_revision("024")
    assert revision is not None
    assert revision.down_revision == "023"


def test_alembic_revision_chain_includes_019() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("019")
    assert revision is not None
    assert revision.down_revision == "018"


def test_alembic_revision_chain_includes_018() -> None:
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("018")
    assert revision is not None
    assert revision.down_revision == "017"


@pytest.mark.asyncio
async def test_sqlite_create_all_includes_phase6_tables(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'models.sqlite'}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        names = await conn.run_sync(lambda sync_conn: set(inspect(sync_conn).get_table_names()))
    await engine.dispose()
    assert "hackerone_report_intents" in names
    assert "security_research_sessions" in names
    assert "research_evidence_node" in names
    assert "research_evidence_edge" in names
    assert "research_findings" in names
    assert "research_identities" in names
    assert "security_research_projects" in names


def test_alembic_upgrade_downgrade_upgrade_postgres() -> None:
    url = os.environ.get("BUGFORGE_ALEMBIC_TEST_URL") or os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        pytest.skip("PostgreSQL is required for the full Alembic upgrade/downgrade chain")
    from alembic import command

    cfg = _alembic_config(url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "015")
    command.upgrade(cfg, "head")
    if hasattr(command, "check"):
        command.check(cfg)
