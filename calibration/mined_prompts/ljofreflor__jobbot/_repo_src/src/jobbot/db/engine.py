"""SQLAlchemy engine and session helpers."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from jobbot.db import models as orm

_JOB_EXTRA_COLUMNS: dict[str, str] = {
    "ats_url": "TEXT",
    "ats_kind": "VARCHAR(64)",
    "posted_at": "DATETIME",
    "invalid_reason": "TEXT",
}


def make_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, future=True)
    orm.Base.metadata.create_all(engine)
    _migrate_sqlite(engine)
    return engine


def _migrate_sqlite(engine: Engine) -> None:
    """Add columns introduced after first create_all (SQLite has no ALTER sync)."""
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(jobs)")).fetchall()
        existing = {r[1] for r in rows}
        for name, col_type in _JOB_EXTRA_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE jobs ADD COLUMN {name} {col_type}"))


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
