"""Session management — track the currently active project."""
from __future__ import annotations

from .db import db_connection
from .projects import get_project


def get_session() -> dict:
    """Return current session state including active project details."""
    with db_connection() as conn:
        row = conn.execute("""
            SELECT s.id, s.project_id, s.last_skill, s.updated_at,
                   p.name as project_name, p.type as project_type,
                   p.status as project_status, p.phase as project_phase,
                   p.description as project_description
            FROM session s
            LEFT JOIN projects p ON p.id = s.project_id
            WHERE s.id = 1
        """).fetchone()
        return dict(row) if row else {}


def set_session(identifier: str, last_skill: str = "") -> dict:
    """Set the active project in the session by slug or name."""
    project = get_project(identifier)
    if not project:
        return {"error": f"Project not found: {identifier}"}
    with db_connection() as conn:
        with conn:
            conn.execute(
                """UPDATE session
                   SET project_id = ?, last_skill = ?, updated_at = datetime('now')
                   WHERE id = 1""",
                (project["id"], last_skill),
            )
    return get_session()


def clear_session() -> dict:
    """Clear the active project from the session."""
    with db_connection() as conn:
        with conn:
            conn.execute(
                "UPDATE session SET project_id = NULL, last_skill = '', updated_at = datetime('now') WHERE id = 1"
            )
    return get_session()
