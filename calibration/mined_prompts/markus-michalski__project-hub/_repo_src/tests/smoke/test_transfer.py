"""Smoke: export/import roundtrip and import_project edge-case fixes.

Covers:
- Basic export → import roundtrip (slug, contacts, notes)
- #96: imported project gets a real docs_path (not empty string)
- #101: overwrite strategy uses delete_project safeguards (session FK, shared contacts)
- merge_strategy="skip" returns early without importing
- merge_strategy="rename" avoids collision via suffix
"""
from pathlib import Path

from tools.contacts import add_contact
from tools.notes import add_note
from tools.projects import create_project, get_project
from tools.session import get_session, set_session
from tools.transfer import export_project, import_project


# ---------------------------------------------------------------------------
# Basic roundtrip
# ---------------------------------------------------------------------------

def test_export_creates_json_file(tmp_path):
    p = create_project("Export Basic")
    add_contact(p["id"], name="Alice", role="PM")
    add_note(p["id"], "First Note", "Content", "note")

    result = export_project(p["id"], str(tmp_path / "export.json"))

    assert result["contacts"] == 1
    assert result["notes"] == 1
    assert Path(result["path"]).exists()


def test_import_roundtrip_restores_project(tmp_path):
    p = create_project("Import Source")
    add_contact(p["id"], name="Bob", role="Dev")
    add_note(p["id"], "A Note", "Body", "note")

    export_result = export_project(p["id"], str(tmp_path / "export.json"))
    json_path = export_result["path"]

    # Use rename so slug doesn't collide with the still-existing source project
    result = import_project(json_path, merge_strategy="rename")

    assert result["imported"] is True
    assert result["contacts"] == 1
    assert result["notes"] == 1


# ---------------------------------------------------------------------------
# Fix #96 — docs_path must be set after import
# ---------------------------------------------------------------------------

def test_imported_project_has_real_docs_path(tmp_path):
    """#96: import_project must create a docs folder; docs_path must not be empty."""
    p = create_project("Docs Path Source")
    export_result = export_project(p["id"], str(tmp_path / "export.json"))

    result = import_project(export_result["path"], merge_strategy="rename")
    assert result["imported"] is True

    imported = get_project(result["slug"])
    assert imported is not None
    assert imported["docs_path"], "docs_path must not be empty after import"
    assert Path(imported["docs_path"]).exists(), "docs folder must exist on disk"


# ---------------------------------------------------------------------------
# merge_strategy="skip"
# ---------------------------------------------------------------------------

def test_skip_strategy_returns_not_imported(tmp_path):
    p = create_project("Skip Me")
    export_result = export_project(p["id"], str(tmp_path / "export.json"))

    result = import_project(export_result["path"], merge_strategy="skip")
    assert result["imported"] is False
    assert "already exists" in result["reason"]


# ---------------------------------------------------------------------------
# merge_strategy="overwrite" — Fix #101
# ---------------------------------------------------------------------------

def test_overwrite_replaces_project(tmp_path):
    """#101: overwrite must succeed and not leave orphan data."""
    p = create_project("Overwrite Target")
    export_result = export_project(p["id"], str(tmp_path / "export.json"))

    result = import_project(export_result["path"], merge_strategy="overwrite")
    assert result["imported"] is True
    restored = get_project(result["slug"])
    assert restored is not None


def test_overwrite_active_project_clears_session_fk(tmp_path):
    """#101: overwriting the currently active project must not raise a FK IntegrityError."""
    p = create_project("Active Project")
    set_session(p["slug"])

    session_before = get_session()
    assert session_before["project_id"] == p["id"], "session should point at the project"

    export_result = export_project(p["id"], str(tmp_path / "export.json"))

    # This must not raise sqlite3.IntegrityError
    result = import_project(export_result["path"], merge_strategy="overwrite")
    assert result["imported"] is True


def test_overwrite_shared_contacts_are_reparented(tmp_path):
    """#101: shared contacts must survive the overwrite of their original project."""
    survivor = create_project("Survivor")
    p = create_project("Overwrite With Shared")
    shared = add_contact(p["id"], name="Shared Person", role="Exec", is_shared=True)

    export_result = export_project(p["id"], str(tmp_path / "export.json"))

    result = import_project(export_result["path"], merge_strategy="overwrite")
    assert result["imported"] is True

    # The shared contact must still exist somewhere (re-parented to survivor)
    from tools.contacts import list_shared_contacts
    shared_after = list_shared_contacts()
    ids_after = [c["id"] for c in shared_after["items"]]
    assert shared["id"] in ids_after, "shared contact must survive the overwrite"
