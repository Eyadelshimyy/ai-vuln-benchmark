"""Project CRUD operations."""
from __future__ import annotations

import re
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from .config import get_docs_root
from .db import db_connection
from .project_links import get_links_for_project


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug.strip("-")


def _row_to_dict(row) -> dict:
    return dict(row) if row else {}


def _ensure_docs_path(slug: str) -> str:
    docs_path = get_docs_root() / slug / "docs"
    docs_path.mkdir(parents=True, exist_ok=True)
    # Create standard subdirectories
    for sub in ["emails", "meeting-notes", "misc"]:
        (docs_path / sub).mkdir(exist_ok=True)
    return str(docs_path)


def list_projects(status: str = "", limit: int = 50, offset: int = 0) -> dict:
    """List projects with pagination, optionally filtered by status.

    Each item includes its "links" list of related projects (see link_project).
    Returns {"items": [...], "total": N, "limit": L, "offset": O}.
    """
    with db_connection() as conn:
        where = "WHERE status = ?" if status else ""
        count_params: list = [status] if status else []
        total: int = conn.execute(
            f"SELECT COUNT(*) FROM projects {where}", count_params
        ).fetchone()[0]

        query = f"SELECT * FROM projects {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        rows = conn.execute(query, count_params + [limit, offset]).fetchall()
        items = [_row_to_dict(r) for r in rows]

    for item in items:
        item["links"] = get_links_for_project(item["id"])
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


def get_project(identifier: str) -> Optional[dict]:
    """Get a project by slug or name (case-insensitive). Includes linked projects (see link_project)."""
    with db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE slug = ? OR LOWER(name) = LOWER(?)",
            (identifier, identifier),
        ).fetchone()
    if not row:
        return None
    project = _row_to_dict(row)
    project["links"] = get_links_for_project(project["id"])
    return project


def get_project_by_id(project_id: int) -> Optional[dict]:
    """Get a project by its numeric ID. Includes linked projects (see link_project)."""
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not row:
        return None
    project = _row_to_dict(row)
    project["links"] = get_links_for_project(project["id"])
    return project


def create_project(
    name: str,
    project_type: str = "generic",
    description: str = "",
    market: str = "",
    products: str = "",
    phase: str = "",
    go_live: str = "",
    budget: str = "",
    notes: str = "",
) -> dict:
    """Create a new project and its docs folder."""
    slug = _slugify(name)
    docs_path = _ensure_docs_path(slug)

    with db_connection() as conn:
        try:
            with conn:
                conn.execute(
                    """INSERT INTO projects
                       (slug, name, type, description, market, products, phase, go_live, budget, notes, docs_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (slug, name, project_type, description, market, products, phase, go_live, budget, notes, docs_path),
                )
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorname != "SQLITE_CONSTRAINT_UNIQUE":
                raise
            existing = conn.execute("SELECT name FROM projects WHERE slug = ?", (slug,)).fetchone()
            existing_name = existing["name"] if existing else name
            return {"error": f"A project with slug '{slug}' already exists (existing project: '{existing_name}')"}
        row = conn.execute("SELECT * FROM projects WHERE slug = ?", (slug,)).fetchone()
        return _row_to_dict(row)


def update_project(identifier: str, **fields) -> Optional[dict]:
    """Update arbitrary fields on a project."""
    allowed = {
        "name", "type", "status", "description", "market",
        "products", "phase", "go_live", "budget", "notes",
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_project(identifier)

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values())
    values.append(identifier)
    values.append(identifier)

    with db_connection() as conn:
        with conn:
            conn.execute(
                f"UPDATE projects SET {set_clause}, updated_at = datetime('now') "
                f"WHERE slug = ? OR LOWER(name) = LOWER(?)",
                values,
            )
        return get_project(identifier)


def delete_project(identifier: str) -> bool:
    """Delete a project by slug or name (case-insensitive).

    Cascades to its non-shared contacts, notes, and project_links via ON DELETE
    CASCADE. Shared contacts (is_shared=True) are re-parented to another
    surviving project first, since they are explicitly cross-project — only if
    this is the last remaining project (nothing to re-parent to, sharing is
    moot with nothing else to share to) are they cascade-deleted too. Clears
    the active session first if it points at this project — the session
    table's FK has no ON DELETE action, so deleting the active project would
    otherwise fail with a FOREIGN KEY constraint error. Also removes the
    project's docs folder from disk (DB delete alone would orphan it, since
    contacts/notes are cascade-deleted from the DB but their .md files on disk
    are not).
    """
    with db_connection() as conn:
        with conn:
            row = conn.execute(
                "SELECT id, docs_path FROM projects WHERE slug = ? OR LOWER(name) = LOWER(?)",
                (identifier, identifier),
            ).fetchone()
            if not row:
                return False
            project_id, docs_path = row["id"], row["docs_path"]
            other_project = conn.execute(
                "SELECT id FROM projects WHERE id != ? ORDER BY id LIMIT 1", (project_id,)
            ).fetchone()
            if other_project:
                conn.execute(
                    "UPDATE contacts SET project_id = ? WHERE project_id = ? AND is_shared = 1",
                    (other_project["id"], project_id),
                )
            conn.execute("UPDATE session SET project_id = NULL WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    if docs_path:
        shutil.rmtree(docs_path, ignore_errors=True)
    return True


def list_docs(project_id: int) -> dict:
    """Return docs path and list of files for a project."""
    project = get_project_by_id(project_id)
    if not project:
        return {"error": "Project not found"}

    docs_path = Path(project["docs_path"])
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)

    result: dict = {"docs_path": str(docs_path), "files": {}}
    for sub in docs_path.iterdir():
        if sub.is_dir():
            result["files"][sub.name] = [f.name for f in sub.iterdir() if f.is_file()]
    return result
