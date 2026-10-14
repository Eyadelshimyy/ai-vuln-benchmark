"""Tests for notes CRUD operations."""
import json
import tempfile
from pathlib import Path

import pytest
from tools.attachments import list_attachments
from tools.notes import add_note, delete_note, get_note, list_notes, update_note
from tools.projects import create_project

# See tests/test_attachments.py for the .resolve() / OS-temp-root rationale.
_HOME = Path(tempfile.gettempdir()).resolve()


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    return create_project("Test Project")


def test_add_note(project):
    note = add_note(project["id"], "My Note", "Some content")

    assert note["title"] == "My Note"
    assert note["content"] == "Some content"
    assert note["type"] == "note"
    assert note["project_id"] == project["id"]


def test_add_note_with_type(project):
    note = add_note(project["id"], "Kickoff", "Notes here", note_type="meeting-notes", agenda="1. Intro")

    assert note["type"] == "meeting-notes"
    assert note["agenda"] == "1. Intro"


def test_get_note(project):
    note = add_note(project["id"], "Fetch Me", "Content")

    found = get_note(note["id"])
    assert found is not None
    assert found["title"] == "Fetch Me"


def test_get_note_not_found():
    assert get_note(99999) is None


def test_list_notes_empty(project):
    result = list_notes(project["id"])
    assert result["items"] == []
    assert result["total"] == 0


def test_list_notes(project):
    add_note(project["id"], "Note 1", "A")
    add_note(project["id"], "Note 2", "B")

    result = list_notes(project["id"])
    assert len(result["items"]) == 2
    assert result["total"] == 2


def test_list_notes_filter_by_type(project):
    add_note(project["id"], "Meeting", "...", note_type="meeting-notes")
    add_note(project["id"], "Decision", "...", note_type="decision")

    meetings = list_notes(project["id"], note_type="meeting-notes")
    assert len(meetings["items"]) == 1
    assert meetings["total"] == 1
    assert meetings["items"][0]["type"] == "meeting-notes"


def test_list_notes_limit(project):
    for i in range(5):
        add_note(project["id"], f"Note {i}", "content")

    result = list_notes(project["id"], limit=3)
    assert len(result["items"]) == 3
    assert result["total"] == 5


def test_list_notes_pagination(project):
    for i in range(5):
        add_note(project["id"], f"Note {i}", "content")

    page1 = list_notes(project["id"], limit=3, offset=0)
    page2 = list_notes(project["id"], limit=3, offset=3)

    assert len(page1["items"]) == 3
    assert len(page2["items"]) == 2
    assert page1["total"] == 5


def test_list_notes_ordered_newest_first(project):
    add_note(project["id"], "First", "a")
    add_note(project["id"], "Second", "b")

    result = list_notes(project["id"])
    assert result["items"][0]["title"] == "Second"


def test_update_note_title(project):
    note = add_note(project["id"], "Old Title", "Content")

    updated = update_note(note["id"], title="New Title")

    assert updated["title"] == "New Title"
    assert updated["content"] == "Content"  # Unchanged


def test_update_note_content(project):
    note = add_note(project["id"], "Title", "Old content")

    updated = update_note(note["id"], content="New content")

    assert updated["content"] == "New content"
    assert updated["title"] == "Title"  # Unchanged


def test_update_note_type(project):
    note = add_note(project["id"], "Title", "Content", note_type="note")

    updated = update_note(note["id"], note_type="decision")

    assert updated["type"] == "decision"


def test_update_note_no_fields_returns_unchanged(project):
    note = add_note(project["id"], "Unchanged", "Content")

    result = update_note(note["id"])

    assert result["title"] == "Unchanged"


def test_update_note_not_found():
    result = update_note(99999, title="Ghost")
    assert result is None


def test_delete_note(project):
    note = add_note(project["id"], "Delete Me", "content")

    assert delete_note(note["id"]) is True
    assert get_note(note["id"]) is None


def test_delete_note_not_found():
    assert delete_note(99999) is False


# --- Bug #75 regression tests ---


@pytest.fixture
def project_with_docs(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p = create_project("Docs Project")
    return p


def test_add_note_unicode_content_roundtrips(project_with_docs):
    """Bug #75 / Bug 1: note with arrow must be saved and returned verbatim."""
    content = "Go-Live 15.09.2026 → ca. 2 Monate"
    note = add_note(project_with_docs["id"], "Meeting", content, note_type="meeting-notes")

    assert note["content"] == content
    assert note["file_path"] is not None

    written = Path(note["file_path"]).read_text(encoding="utf-8")
    assert "→" in written


def test_add_note_no_orphan_on_write_failure(project_with_docs, monkeypatch):
    """Bug #75 / Bug 2: if write_note_to_disk raises, no DB row must be created.

    The file write happens before the DB insert, so a write failure never
    creates a row in the first place — no rollback needed.
    """
    import tools.notes as notes_module

    def _fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(notes_module, "write_note_to_disk", _fail)

    with pytest.raises(OSError):
        add_note(project_with_docs["id"], "Ghost Note", "content")

    result = list_notes(project_with_docs["id"])
    assert result["total"] == 0, "no row must be created when the disk write fails"


def test_note_has_updated_at(project):
    note = add_note(project["id"], "Timestamped", "Content")
    assert "updated_at" in note
    assert note["updated_at"] is not None


def test_update_note_sets_updated_at(project):
    note = add_note(project["id"], "Track Me", "Initial")
    original_updated_at = note["updated_at"]

    import time
    time.sleep(0.01)

    updated = update_note(note["id"], content="Changed")
    assert updated["updated_at"] >= original_updated_at


def test_list_notes_includes_updated_at(project):
    add_note(project["id"], "Note A", "Content A")
    result = list_notes(project["id"])
    assert "updated_at" in result["items"][0]


def test_migration_adds_updated_at_to_legacy_db(tmp_path, monkeypatch):
    """Regression test for #42: updated_at missing from notes in pre-migration DBs.

    Simulates a DB created before the updated_at column was added and verifies
    that init_db() correctly migrates it so update_note/list_notes work.
    """
    import sqlite3

    import tools.db as db_module

    db_file = tmp_path / "legacy.db"
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_file)

    # Build legacy schema without updated_at
    conn = sqlite3.connect(db_file)
    conn.executescript("""
        CREATE TABLE projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            slug        TEXT NOT NULL UNIQUE,
            name        TEXT NOT NULL,
            type        TEXT NOT NULL DEFAULT 'generic',
            status      TEXT NOT NULL DEFAULT 'active',
            description TEXT NOT NULL DEFAULT '',
            docs_path   TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE notes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            title       TEXT NOT NULL,
            type        TEXT NOT NULL DEFAULT 'note',
            content     TEXT NOT NULL DEFAULT '',
            agenda      TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE session (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            project_id INTEGER REFERENCES projects(id),
            last_skill TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT OR IGNORE INTO session (id) VALUES (1);
        INSERT INTO projects (slug, name) VALUES ('legacy-proj', 'Legacy Project');
        INSERT INTO notes (project_id, title, content)
            VALUES (1, 'Old Note', 'Pre-migration content');
    """)
    conn.close()

    # Confirm legacy schema has no updated_at
    conn = sqlite3.connect(db_file)
    cols_before = [row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()]
    conn.close()
    assert "updated_at" not in cols_before

    # init_db() must apply the migration
    from tools.db import init_db
    init_db()

    conn = sqlite3.connect(db_file)
    cols_after = [row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()]
    conn.close()
    assert "updated_at" in cols_after, "Migration must add updated_at column"

    # update_note must not raise "no such column: updated_at"
    updated = update_note(1, content="Post-migration update")
    assert updated is not None
    assert updated["content"] == "Post-migration update"
    assert updated["updated_at"], "updated_at must be set after update"

    # list_notes must not raise "no such column: updated_at"
    result = list_notes(1)
    assert result["total"] == 1
    assert "updated_at" in result["items"][0]


# --- add_note source_paths: preserve originals on import (project-hub#134) ---


def test_add_note_with_source_paths_attaches_originals(project_with_docs, tmp_path):
    f1 = tmp_path / "doc1.txt"
    f2 = tmp_path / "doc2.txt"
    f1.write_text("first")
    f2.write_text("second")

    note = add_note(
        project_with_docs["id"],
        "Imported",
        "extracted text",
        source_paths=[str(f1), str(f2)],
        home_override=_HOME,
    )

    assert {a["name"] for a in note["attachments_added"]} == {"doc1.txt", "doc2.txt"}
    assert "attachment_failures" not in note

    assert len(list_attachments(note["id"])) == 2
    # the returned note itself must reflect the attach, not the pre-attach DB snapshot
    assert len(json.loads(note["attachments"])) == 2


def test_add_note_source_paths_surfaces_partial_failures(project_with_docs, tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("ok")

    note = add_note(
        project_with_docs["id"],
        "Imported",
        "extracted text",
        source_paths=[str(good), "/nonexistent/missing.txt"],
        home_override=_HOME,
    )

    assert len(note["attachments_added"]) == 1
    assert note["attachments_added"][0]["name"] == "good.txt"
    assert len(note["attachment_failures"]) == 1
    assert note["attachment_failures"][0]["path"] == "/nonexistent/missing.txt"


def test_add_note_without_source_paths_unchanged(project_with_docs):
    note = add_note(project_with_docs["id"], "Plain", "just text")

    assert "attachments_added" not in note
    assert "attachment_failures" not in note


def test_add_note_source_paths_empty_list_unchanged(project_with_docs):
    note = add_note(project_with_docs["id"], "Plain", "just text", source_paths=[])

    assert "attachments_added" not in note


def test_add_note_source_paths_with_no_docs_path_does_not_raise(project_with_docs, tmp_path):
    # Regression: attach_files() raises ValueError when the project has no docs_path
    # (e.g. a legacy/edge-case row — notes.py itself already guards write_note_to_disk
    # against this). That raise must never propagate past add_note(): the note row is
    # already committed by this point, so an uncaught exception here would recreate
    # #134's failure mode — a hard tool error that looks like "nothing was saved" when
    # the note in fact exists, just without its attachment.
    import tools.db as db_module

    with db_module.db_connection() as conn:
        with conn:
            conn.execute(
                "UPDATE projects SET docs_path = '' WHERE id = ?", (project_with_docs["id"],)
            )

    f = tmp_path / "doc.txt"
    f.write_text("content")

    note = add_note(
        project_with_docs["id"], "No Docs Path", "text",
        source_paths=[str(f)], home_override=_HOME,
    )

    assert note["title"] == "No Docs Path"
    assert note["attachments_added"] == []
    assert len(note["attachment_failures"]) == 1
    assert note["attachment_failures"][0]["path"] == str(f)


def test_add_note_source_paths_all_fail_without_raising(project_with_docs):
    # Distinct from the no-docs_path case above: here the note-level attach_files()
    # call succeeds normally (valid docs_path) but every individual path fails inside
    # its per-file loop — must surface as attachment_failures, not raise.
    note = add_note(
        project_with_docs["id"],
        "Imported",
        "extracted text",
        source_paths=["/nonexistent/a.txt", "/nonexistent/b.txt"],
        home_override=_HOME,
    )

    assert note["attachments_added"] == []
    assert len(note["attachment_failures"]) == 2
