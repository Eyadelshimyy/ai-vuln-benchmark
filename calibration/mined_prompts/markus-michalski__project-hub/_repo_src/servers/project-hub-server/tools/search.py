"""Full-text search across notes and contacts."""
from __future__ import annotations

from .db import db_connection


def search_notes(query: str, project_id: int = 0) -> list[dict]:
    """Search notes by title or content (case-insensitive).

    project_id: limit to a specific project (0 = all projects).
    Returns matching notes ordered by created_at DESC.
    """
    if not query.strip():
        return []
    pattern = f"%{query}%"
    with db_connection() as conn:
        if project_id:
            rows = conn.execute(
                """SELECT n.*, p.name as project_name, p.slug as project_slug
                   FROM notes n
                   JOIN projects p ON p.id = n.project_id
                   WHERE n.project_id = ?
                     AND (n.title LIKE ? OR n.content LIKE ?)
                   ORDER BY n.created_at DESC""",
                (project_id, pattern, pattern),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT n.*, p.name as project_name, p.slug as project_slug
                   FROM notes n
                   JOIN projects p ON p.id = n.project_id
                   WHERE n.title LIKE ? OR n.content LIKE ?
                   ORDER BY n.created_at DESC""",
                (pattern, pattern),
            ).fetchall()
        return [dict(r) for r in rows]


def search_contacts(query: str, project_id: int = 0) -> list[dict]:
    """Search contacts by name, role, email, or company (case-insensitive).

    project_id: limit to a specific project (0 = all projects).
    Returns matching contacts ordered by name.
    """
    if not query.strip():
        return []
    pattern = f"%{query}%"
    with db_connection() as conn:
        if project_id:
            rows = conn.execute(
                """SELECT c.*, p.name as project_name, p.slug as project_slug
                   FROM contacts c
                   JOIN projects p ON p.id = c.project_id
                   WHERE (c.project_id = ? OR c.is_shared = 1)
                     AND (c.name LIKE ? OR c.role LIKE ? OR c.email LIKE ? OR c.company LIKE ?)
                   ORDER BY c.name""",
                (project_id, pattern, pattern, pattern, pattern),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT c.*, p.name as project_name, p.slug as project_slug
                   FROM contacts c
                   JOIN projects p ON p.id = c.project_id
                   WHERE c.name LIKE ? OR c.role LIKE ? OR c.email LIKE ? OR c.company LIKE ?
                   ORDER BY c.name""",
                (pattern, pattern, pattern, pattern),
            ).fetchall()
        return [dict(r) for r in rows]
