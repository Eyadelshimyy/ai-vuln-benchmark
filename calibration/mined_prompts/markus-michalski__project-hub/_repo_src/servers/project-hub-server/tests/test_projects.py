"""Tests for project CRUD operations."""
import shutil
from pathlib import Path

from tools.contacts import add_contact, list_contacts, list_shared_contacts
from tools.notes import add_note, list_notes
from tools.project_links import link_project
from tools.projects import (
    create_project,
    delete_project,
    get_project,
    get_project_by_id,
    list_projects,
    update_project,
)
from tools.session import get_session, set_session


def test_create_project_basic(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("My Test Project")

    assert project["name"] == "My Test Project"
    assert project["slug"] == "my-test-project"
    assert project["type"] == "generic"
    assert project["status"] == "active"
    assert project["id"] is not None


def test_create_project_with_type(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Acme Onboarding", project_type="merchant-onboarding", market="DE")

    assert project["type"] == "merchant-onboarding"
    assert project["market"] == "DE"


def test_slugify_special_chars(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Müller & Partner GmbH")

    # Special chars stripped, spaces become dashes
    assert "-" in project["slug"]
    assert "&" not in project["slug"]


def test_get_project_by_slug(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Search Me")

    found = get_project("search-me")
    assert found is not None
    assert found["name"] == "Search Me"


def test_get_project_by_name_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Case Test")

    found = get_project("case test")
    assert found is not None
    assert found["name"] == "Case Test"


def test_get_project_not_found():
    result = get_project("does-not-exist")
    assert result is None


def test_list_projects_empty():
    result = list_projects()
    assert result["items"] == []
    assert result["total"] == 0


def test_list_projects(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Alpha")
    create_project("Beta")

    result = list_projects()
    assert len(result["items"]) == 2
    assert result["total"] == 2


def test_list_projects_filter_by_status(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    p = create_project("Active One")
    create_project("Another Active")
    update_project(p["slug"], status="paused")

    active = list_projects(status="active")
    paused = list_projects(status="paused")

    assert len(active["items"]) == 1
    assert active["total"] == 1
    assert len(paused["items"]) == 1


def test_list_projects_pagination(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    for i in range(5):
        create_project(f"Project {i}")

    page1 = list_projects(limit=3, offset=0)
    page2 = list_projects(limit=3, offset=3)

    assert len(page1["items"]) == 3
    assert page1["total"] == 5
    assert len(page2["items"]) == 2
    assert page2["offset"] == 3


def test_update_project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Update Me")
    updated = update_project(project["slug"], phase="UAT", status="paused")

    assert updated["phase"] == "UAT"
    assert updated["status"] == "paused"


def test_update_project_updated_at_changes(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Timestamp Test")
    original_ts = project["updated_at"]

    import time
    time.sleep(1.1)  # SQLite datetime() has 1s resolution

    update_project(project["slug"], phase="Done")
    refreshed = get_project(project["slug"])

    assert refreshed["updated_at"] != original_ts


def test_update_project_no_fields_returns_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("No Change")
    result = update_project(project["slug"])

    assert result["name"] == "No Change"


def test_get_project_includes_empty_links_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Unlinked Project")

    assert get_project(project["slug"])["links"] == []
    assert get_project_by_id(project["id"])["links"] == []


def test_get_project_includes_links_after_linking(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    joybuy = create_project("Joybuy")
    joybuy2 = create_project("Joybuy 2")
    link_project(joybuy2["slug"], joybuy["slug"], "successor")

    found = get_project(joybuy2["slug"])
    assert found["links"] == [
        {"relation": "successor", "project": {"id": joybuy["id"], "slug": joybuy["slug"], "name": joybuy["name"]}}
    ]

    found_by_id = get_project_by_id(joybuy["id"])
    assert found_by_id["links"] == [
        {"relation": "predecessor", "project": {"id": joybuy2["id"], "slug": joybuy2["slug"], "name": joybuy2["name"]}}
    ]


def test_list_projects_includes_empty_links_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Lonely Project")

    items = list_projects()["items"]
    assert items[0]["links"] == []


def test_list_projects_includes_links_after_linking(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    joybuy = create_project("Joybuy")
    joybuy2 = create_project("Joybuy 2")
    link_project(joybuy2["slug"], joybuy["slug"], "successor")

    items = {p["slug"]: p for p in list_projects()["items"]}
    assert items[joybuy2["slug"]]["links"] == [
        {"relation": "successor", "project": {"id": joybuy["id"], "slug": joybuy["slug"], "name": joybuy["name"]}}
    ]
    assert items[joybuy["slug"]]["links"] == [
        {"relation": "predecessor", "project": {"id": joybuy2["id"], "slug": joybuy2["slug"], "name": joybuy2["name"]}}
    ]


# ---------------------------------------------------------------------------
# delete_project — needed for the create-testdata/reset-testdata/delete-testdata
# sandbox convention (project-hub#82): the project row has no other way to be
# fully removed, and its UNIQUE slug constraint means a re-run of create_project
# after a soft-delete would fail without a real delete path.
# ---------------------------------------------------------------------------


def test_delete_project_removes_it(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Delete Me")
    result = delete_project(project["slug"])

    assert result is True
    assert get_project(project["slug"]) is None


def test_delete_project_not_found_returns_false():
    assert delete_project("does-not-exist") is False


def test_delete_project_by_name_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Case Delete")
    result = delete_project("case delete")

    assert result is True
    assert get_project(project["slug"]) is None


def test_delete_project_cascades_to_contacts_and_notes(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Cascade Test")
    add_contact(project["id"], name="Cascade Contact")
    add_note(project["id"], title="Cascade Note", content="...")

    delete_project(project["slug"])

    assert list_contacts(project["id"])["items"] == []
    assert list_notes(project["id"])["items"] == []


def test_delete_project_does_not_affect_other_projects(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    keep = create_project("Keep Me")
    remove = create_project("Remove Me")

    delete_project(remove["slug"])

    assert get_project(keep["slug"]) is not None
    assert get_project(remove["slug"]) is None


def test_delete_project_clears_active_session(tmp_path, monkeypatch):
    # Regression: the session table has a FK to projects with no ON DELETE action —
    # deleting the active-session project must not crash with an IntegrityError.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Active Session Project")
    set_session(project["slug"])

    result = delete_project(project["slug"])

    assert result is True
    assert get_session()["project_id"] is None


def test_delete_project_removes_docs_folder(tmp_path, monkeypatch):
    # Regression: delete_project only removed DB rows, leaving the docs folder
    # (and any note .md files inside it) orphaned on disk.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Disk Cleanup Test")
    add_note(project["id"], title="A Note", content="...")
    docs_path = Path(project["docs_path"])
    assert docs_path.exists()

    delete_project(project["slug"])

    assert not docs_path.exists()


def test_delete_project_cascades_to_project_links(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    p1 = create_project("Link Cascade A")
    p2 = create_project("Link Cascade B")
    link_project(p2["slug"], p1["slug"], "successor")

    delete_project(p1["slug"])

    assert get_project(p2["slug"])["links"] == []


def test_delete_project_survives_already_missing_docs_folder(tmp_path, monkeypatch):
    # The DB delete commits BEFORE shutil.rmtree runs (ignore_errors=True) — a docs
    # folder that's already gone (manually deleted, or left over from a prior partial
    # run) must not crash delete_project or block the DB delete.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    project = create_project("Missing Docs Folder Test")
    shutil.rmtree(project["docs_path"])
    assert not Path(project["docs_path"]).exists()

    result = delete_project(project["slug"])

    assert result is True
    assert get_project(project["slug"]) is None


def test_delete_project_with_empty_docs_path(tmp_path, monkeypatch):
    # Imported projects (tools/transfer.py) can have docs_path == "" — must not crash.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    from tools.db import db_connection

    project = create_project("Empty Docs Path Test")
    with db_connection() as conn:
        with conn:
            conn.execute("UPDATE projects SET docs_path = '' WHERE id = ?", (project["id"],))

    result = delete_project(project["slug"])

    assert result is True
    assert get_project(project["slug"]) is None


def test_delete_project_reparents_shared_contacts_instead_of_deleting_them(tmp_path, monkeypatch):
    # Shared contacts are explicitly cross-project — deleting whichever project happens
    # to "own" the row must not destroy it for every other project surfacing it too.
    # Re-parent to another surviving project before the cascade runs.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    owner = create_project("Shared Contact Owner")
    other = create_project("Unrelated Other Project")
    contact = add_contact(owner["id"], name="Cross-Project Shared Person", is_shared=True)

    delete_project(owner["slug"])

    assert any(
        c["name"] == "Cross-Project Shared Person" for c in list_shared_contacts()["items"]
    )
    survivors = list_contacts(other["id"])["items"]
    assert any(c["id"] == contact["id"] and c["project_id"] == other["id"] for c in survivors)
    assert get_project(other["slug"]) is not None


def test_delete_project_does_not_reparent_non_shared_contacts(tmp_path, monkeypatch):
    # Only is_shared contacts get re-parented — project-local contacts are legitimately
    # scoped to their project and must still be cascade-deleted with it.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    owner = create_project("Local Contact Owner")
    create_project("Unrelated Other Project")
    add_contact(owner["id"], name="Local Only Person", is_shared=False)

    delete_project(owner["slug"])

    assert not any(
        c["name"] == "Local Only Person" for c in list_shared_contacts()["items"]
    )


def test_delete_project_last_project_still_deletes_its_shared_contacts(tmp_path, monkeypatch):
    # If the deleted project is the only project left, there is nowhere to re-parent a
    # shared contact to — sharing is moot with zero other projects, so it is deleted too.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    only = create_project("Only Project")
    add_contact(only["id"], name="Last Shared Person", is_shared=True)

    result = delete_project(only["slug"])

    assert result is True
    assert not any(
        c["name"] == "Last Shared Person" for c in list_shared_contacts()["items"]
    )


def test_create_project_duplicate_name_returns_structured_error(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Dup Project")
    result = create_project("Dup Project")

    assert result == {
        "error": "A project with slug 'dup-project' already exists (existing project: 'Dup Project')"
    }


def test_create_project_duplicate_slug_different_name_reports_existing_name(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Acme GmbH")
    result = create_project("Acme  GmbH!")  # slugifies to the same "acme-gmbh"

    assert result == {
        "error": "A project with slug 'acme-gmbh' already exists (existing project: 'Acme GmbH')"
    }


def test_create_project_duplicate_name_case_insensitive_slug(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Dup Project")
    result = create_project("dup project")

    assert "error" in result


def test_create_project_after_duplicate_error_connection_still_usable(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)

    create_project("Dup Project")
    error = create_project("Dup Project")
    assert "error" in error

    other = create_project("Other Project")
    assert other["name"] == "Other Project"
    assert list_projects()["total"] == 2
