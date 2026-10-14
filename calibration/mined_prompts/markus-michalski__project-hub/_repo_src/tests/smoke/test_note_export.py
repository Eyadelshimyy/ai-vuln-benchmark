"""Smoke: add_note auto-exports markdown files to docs_path subfolders."""
from pathlib import Path

from tools.notes import add_note, delete_note
from tools.projects import create_project


def test_meeting_note_creates_file_in_meeting_notes():
    p = create_project("Meeting Export Project")
    note = add_note(p["id"], "Team Standup", "Daily sync content", "meeting-notes")

    assert "file_path" in note
    path = Path(note["file_path"])
    assert path.exists()
    assert "meeting-notes" in str(path)
    content = path.read_text()
    assert "Team Standup" in content
    assert "Daily sync content" in content


def test_email_note_creates_file_in_emails():
    p = create_project("Email Export Project")
    note = add_note(p["id"], "Subject: Important", "Email body here", "email")

    path = Path(note["file_path"])
    assert path.exists()
    assert "emails" in str(path)


def test_regular_note_creates_file_in_misc():
    p = create_project("Misc Note Project")
    note = add_note(p["id"], "Random thought", "Something to remember", "note")

    path = Path(note["file_path"])
    assert path.exists()
    assert "misc" in str(path)


def test_decision_note_creates_file_in_misc():
    p = create_project("Decision Project")
    note = add_note(p["id"], "Architecture Decision", "We chose PostgreSQL", "decision")

    path = Path(note["file_path"])
    assert path.exists()
    assert "misc" in str(path)


def test_action_item_appends_to_single_todo_md():
    p = create_project("Action Item Project")
    note1 = add_note(p["id"], "Fix the auth bug", "In module auth.py", "action-item")
    note2 = add_note(p["id"], "Write API docs", "Cover all endpoints", "action-item")

    assert note1["file_path"].endswith("todo.md")
    assert note1["file_path"] == note2["file_path"]

    content = Path(note1["file_path"]).read_text()
    assert "Fix the auth bug" in content
    assert "Write API docs" in content


def test_todo_md_has_open_and_done_sections():
    p = create_project("Todo Sections Project")
    add_note(p["id"], "First task", "Do this first", "action-item")

    todo_path = Path(p["docs_path"]) / "misc" / "todo.md"
    content = todo_path.read_text()
    assert "## Open" in content
    assert "## Done" in content


def test_file_has_date_slug_filename():
    p = create_project("Filename Project")
    note = add_note(p["id"], "My Meeting Notes!", "Content", "meeting-notes")

    filename = Path(note["file_path"]).name
    # Must match YYYY-MM-DD-slug.md
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}-.+\.md$", filename)


def test_note_file_path_is_none_when_docs_path_empty(monkeypatch):
    """add_note returns file_path=None when project has no docs_path configured."""
    from tools import projects as proj_module

    original = proj_module.get_project_by_id

    def patched(pid):
        result = original(pid)
        if result:
            result = dict(result)
            result["docs_path"] = ""
        return result

    import tools.notes as notes_module
    monkeypatch.setattr(notes_module, "get_project_by_id", patched)

    p = create_project("No Docs Path Project")
    note = add_note(p["id"], "Nowhere to write", "Content", "note")

    assert note["id"] is not None
    assert note.get("file_path") is None


def test_delete_note_removes_companion_file():
    """delete_note must remove the .md file it wrote at creation time."""
    p = create_project("Delete Cleanup Project")
    note = add_note(p["id"], "Cleanup Note", "Will be deleted", "note")

    file_path = Path(note["file_path"])
    assert file_path.exists(), "file should exist before delete"

    deleted = delete_note(note["id"])
    assert deleted is True
    assert not file_path.exists(), "file should be removed after delete_note"


def test_delete_note_action_item_leaves_todo_md():
    """delete_note must NOT remove the shared todo.md for action-item notes."""
    p = create_project("Action Delete Project")
    note1 = add_note(p["id"], "Task A", "Do this", "action-item")
    note2 = add_note(p["id"], "Task B", "Do that", "action-item")

    todo_path = Path(note1["file_path"])
    assert note1["file_path"] == note2["file_path"], "both should share todo.md"
    assert todo_path.exists()

    # Deleting one action-item must not destroy the shared todo.md
    delete_note(note1["id"])
    assert todo_path.exists(), "todo.md must survive deletion of a single action-item note"


def test_file_path_stored_in_db():
    """add_note stores file_path in the DB so delete_note can find it later."""
    from tools.notes import get_note

    p = create_project("File Path DB Project")
    note = add_note(p["id"], "Persisted Path", "Body", "email")

    fetched = get_note(note["id"])
    assert fetched is not None
    assert fetched.get("file_path"), "file_path must be persisted in DB row"
    assert fetched["file_path"] == note["file_path"]
