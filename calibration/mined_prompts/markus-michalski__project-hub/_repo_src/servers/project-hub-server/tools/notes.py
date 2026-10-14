"""Notes CRUD operations."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .db import db_connection
from .docs_writer import write_note_to_disk
from .projects import get_project_by_id


def list_notes(project_id: int, note_type: str = "", limit: int = 50, offset: int = 0) -> dict:
    """List notes for a project with pagination.

    Returns {"items": [...], "total": N, "limit": L, "offset": O}, newest first.
    """
    with db_connection() as conn:
        where = "WHERE project_id = ?"
        params: list = [project_id]
        if note_type:
            where += " AND type = ?"
            params.append(note_type)

        total: int = conn.execute(
            f"SELECT COUNT(*) FROM notes {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"SELECT * FROM notes {where} ORDER BY updated_at DESC, created_at DESC, id DESC"
            f" LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return {
            "items": [dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }


def get_note(note_id: int) -> Optional[dict]:
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return dict(row) if row else None


def add_note(
    project_id: int,
    title: str,
    content: str,
    note_type: str = "note",
    agenda: str = "",
    source_paths: Optional[list[str]] = None,
    home_override: Optional[Path] = None,
) -> dict:
    """Add a note to a project.

    note_type options: note, meeting-notes, email, decision, action-item
    agenda: optional agenda text to compare against during meeting-notes summarization
    source_paths: absolute paths of local files `content` was extracted/imported from
        (project-hub#134) — when given, each is preserved via attach_files() right after
        the note is created, so the original is never lost, only its extracted text kept.
        A bad individual path never aborts note creation; failures are surfaced on the
        returned note as "attachment_failures" instead of raising.
    home_override: used in tests to simulate a different home directory for
        source_paths' path-traversal checking (see attach_files()).
    """
    file_path: Optional[str] = None
    project = get_project_by_id(project_id)
    if project and project.get("docs_path"):
        file_path = write_note_to_disk(project["docs_path"], title, content, note_type)

    with db_connection() as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO notes (project_id, title, type, content, agenda, file_path)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, title, note_type, content, agenda, file_path or ""),
            )
            note_id = cursor.lastrowid
            if note_id is None:
                raise RuntimeError("INSERT did not return a rowid")  # pragma: no cover
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        note = dict(row)

    note["file_path"] = file_path

    if source_paths:
        # Deferred: attachments.py imports get_note from this module at module load
        # time, so a top-level import here would be circular.
        from .attachments import attach_files

        try:
            result = attach_files(note_id, source_paths, home_override=home_override)
        except ValueError as exc:
            # attach_files() raises for note-level problems (e.g. the project has no
            # docs_path) rather than per-file ones. The note row above is already
            # committed, so this must never propagate past add_note() — that would
            # recreate #134's failure mode: a hard tool error that looks like "nothing
            # was saved" when the note in fact exists, just without its attachment(s).
            result = {
                "attached": [],
                "failed": [{"path": p, "error": str(exc)} for p in source_paths],
            }

        note["attachments_added"] = result["attached"]
        if result["failed"]:
            note["attachment_failures"] = result["failed"]
        if result["attached"]:
            # attach_files() already persisted these to the DB; the `note` dict above
            # was read before that write, so refresh it to keep the returned
            # "attachments" field in sync with "attachments_added" instead of
            # returning a stale "[]" next to a populated attachments_added.
            refreshed = get_note(note_id)
            if refreshed:
                note = refreshed
                note["file_path"] = file_path
                note["attachments_added"] = result["attached"]
                if result["failed"]:
                    note["attachment_failures"] = result["failed"]

    return note


def update_note(
    note_id: int,
    title: str = "",
    content: str = "",
    note_type: str = "",
    agenda: str = "",
) -> Optional[dict]:
    """Update an existing note. Only non-empty values are updated."""
    allowed = {"title": title, "type": note_type, "content": content, "agenda": agenda}
    updates = {k: v for k, v in allowed.items() if v}
    if not updates:
        return get_note(note_id)

    updates["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [note_id]

    with db_connection() as conn:
        with conn:
            conn.execute(f"UPDATE notes SET {set_clause} WHERE id = ?", values)
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return dict(row) if row else None


def delete_note(note_id: int) -> bool:
    note = get_note(note_id)
    if not note:
        return False

    with db_connection() as conn:
        with conn:
            result = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        deleted = result.rowcount > 0

    # Remove the companion .md file — skip action-items (shared todo.md, not per-note)
    if deleted and note.get("type") != "action-item":
        fp = note.get("file_path", "")
        if fp:
            try:
                path = Path(fp)
                if path.exists():
                    path.unlink()
            except OSError:
                pass  # best-effort: don't fail the delete if disk cleanup fails

    return deleted
