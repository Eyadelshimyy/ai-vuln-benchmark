"""Database initialization and connection management."""
from __future__ import annotations

import sqlite3
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .config import get_db_path_from_config

# Retry settings for SQLITE_BUSY on network/shared paths
_BUSY_RETRY_COUNT = 5
_BUSY_RETRY_DELAY = 0.2  # seconds between retries


def get_db_path() -> Path:
    path = get_db_path_from_config()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_network_path(path: Path) -> bool:
    """Heuristic: flag paths that look like a network mount or cloud sync folder."""
    network_hints = ("/mnt/", "/media/", "/net/", "/Volumes/", "Dropbox", "OneDrive", "Google Drive", "iCloud")
    path_str = str(path)
    return any(hint in path_str for hint in network_hints)


def get_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    # Longer timeout on network paths to survive transient SQLITE_BUSY spikes
    timeout = 30.0 if _is_network_path(db_path) else 5.0
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager with automatic retry on SQLITE_BUSY (network shares)."""
    last_exc: Exception | None = None
    for attempt in range(_BUSY_RETRY_COUNT):
        try:
            conn = get_connection()
            try:
                yield conn
            finally:
                conn.close()
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc).lower():
                last_exc = exc
                if attempt < _BUSY_RETRY_COUNT - 1:
                    print(
                        f"[project-hub] DB locked (attempt {attempt + 1}/{_BUSY_RETRY_COUNT}), retrying…",
                        file=sys.stderr,
                    )
                    time.sleep(_BUSY_RETRY_DELAY * (attempt + 1))
            else:
                raise
    raise last_exc  # type: ignore[misc]


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    conn = get_connection()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slug        TEXT    NOT NULL UNIQUE,
                name        TEXT    NOT NULL,
                type        TEXT    NOT NULL DEFAULT 'generic',
                status      TEXT    NOT NULL DEFAULT 'active',
                description TEXT    NOT NULL DEFAULT '',
                market      TEXT    NOT NULL DEFAULT '',
                products    TEXT    NOT NULL DEFAULT '',
                phase       TEXT    NOT NULL DEFAULT '',
                go_live     TEXT    NOT NULL DEFAULT '',
                budget      TEXT    NOT NULL DEFAULT '',
                notes       TEXT    NOT NULL DEFAULT '',
                docs_path   TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS contacts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name        TEXT    NOT NULL,
                role        TEXT    NOT NULL DEFAULT '',
                type        TEXT    NOT NULL DEFAULT 'internal',
                email       TEXT    NOT NULL DEFAULT '',
                phone       TEXT    NOT NULL DEFAULT '',
                company     TEXT    NOT NULL DEFAULT '',
                notes       TEXT    NOT NULL DEFAULT '',
                is_shared   INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                title       TEXT    NOT NULL,
                type        TEXT    NOT NULL DEFAULT 'note',
                content     TEXT    NOT NULL DEFAULT '',
                agenda      TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS project_links (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                related_project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                relation_type       TEXT    NOT NULL CHECK (relation_type IN ('successor', 'related')),
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                CHECK (project_id != related_project_id),
                UNIQUE (project_id, related_project_id, relation_type)
            );

            CREATE TABLE IF NOT EXISTS session (
                id              INTEGER PRIMARY KEY CHECK (id = 1),
                project_id      INTEGER REFERENCES projects(id),
                last_skill      TEXT    NOT NULL DEFAULT '',
                updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            INSERT OR IGNORE INTO session (id) VALUES (1);
        """)
        # Safe migrations: no-op if column already exists.
        # updated_at uses a constant default ('') because ALTER TABLE ADD COLUMN
        # rejects non-constant expressions like datetime('now') in SQLite.
        # Existing rows are backfilled from created_at so ORDER BY updated_at works.
        for alter_sql, backfill_sql in (
            (
                "ALTER TABLE notes ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''",
                "UPDATE notes SET updated_at = created_at WHERE updated_at = ''",
            ),
            ("ALTER TABLE notes ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'", None),
            ("ALTER TABLE notes ADD COLUMN file_path TEXT NOT NULL DEFAULT ''", None),
            ("ALTER TABLE contacts ADD COLUMN is_shared INTEGER NOT NULL DEFAULT 0", None),
        ):
            try:
                conn.execute(alter_sql)
                if backfill_sql:
                    conn.execute(backfill_sql)
                conn.commit()
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
    conn.close()
