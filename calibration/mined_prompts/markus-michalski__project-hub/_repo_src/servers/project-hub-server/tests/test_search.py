"""Tests for cross-project search (notes and contacts)."""
import pytest
from tools.contacts import add_contact
from tools.notes import add_note
from tools.projects import create_project
from tools.search import search_contacts, search_notes


@pytest.fixture
def projects(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Alpha Project")
    p2 = create_project("Beta Project")
    return p1, p2


# ---------------------------------------------------------------------------
# search_notes
# ---------------------------------------------------------------------------

def test_search_notes_by_title(projects):
    p1, _ = projects
    add_note(p1["id"], title="Budget Review Q1", content="Numbers look good.")
    add_note(p1["id"], title="Meeting Minutes", content="Discussed timelines.")

    results = search_notes("Budget")
    assert len(results) == 1
    assert results[0]["title"] == "Budget Review Q1"


def test_search_notes_by_content(projects):
    p1, _ = projects
    add_note(p1["id"], title="Status Update", content="Project is on track for go-live.")

    results = search_notes("go-live")
    assert len(results) == 1


def test_search_notes_case_insensitive(projects):
    p1, _ = projects
    add_note(p1["id"], title="Risk Assessment", content="Low risk.")

    assert search_notes("risk assessment") != []
    assert search_notes("RISK ASSESSMENT") != []


def test_search_notes_across_projects(projects):
    p1, p2 = projects
    add_note(p1["id"], title="Alpha Note", content="shared keyword here")
    add_note(p2["id"], title="Beta Note", content="shared keyword here")

    results = search_notes("shared keyword")
    assert len(results) == 2


def test_search_notes_filter_by_project_id(projects):
    p1, p2 = projects
    add_note(p1["id"], title="P1 Note", content="unique term xyz")
    add_note(p2["id"], title="P2 Note", content="unique term xyz")

    results = search_notes("unique term xyz", project_id=p1["id"])
    assert len(results) == 1
    assert results[0]["title"] == "P1 Note"


def test_search_notes_empty_query_returns_empty(projects):
    p1, _ = projects
    add_note(p1["id"], title="Some Note", content="Some content.")

    assert search_notes("") == []


def test_search_notes_no_match_returns_empty(projects):
    p1, _ = projects
    add_note(p1["id"], title="Note", content="Content.")

    assert search_notes("absolutely-no-match-xyz") == []


# ---------------------------------------------------------------------------
# search_contacts
# ---------------------------------------------------------------------------

def test_search_contacts_by_name(projects):
    p1, _ = projects
    add_contact(p1["id"], name="Maria Schmidt", role="PM")
    add_contact(p1["id"], name="John Doe", role="Dev")

    results = search_contacts("Maria")
    assert len(results) == 1
    assert results[0]["name"] == "Maria Schmidt"


def test_search_contacts_by_role(projects):
    p1, _ = projects
    add_contact(p1["id"], name="Alice", role="Tech Lead")

    results = search_contacts("Tech Lead")
    assert len(results) == 1


def test_search_contacts_by_email(projects):
    p1, _ = projects
    add_contact(p1["id"], name="Bob", email="bob@example.com")

    results = search_contacts("bob@example.com")
    assert len(results) == 1


def test_search_contacts_by_company(projects):
    p1, _ = projects
    add_contact(p1["id"], name="Eve", company="ACME Corp")

    results = search_contacts("ACME")
    assert len(results) == 1


def test_search_contacts_case_insensitive(projects):
    p1, _ = projects
    add_contact(p1["id"], name="Franz Kafka")

    assert search_contacts("franz kafka") != []
    assert search_contacts("FRANZ KAFKA") != []


def test_search_contacts_across_projects(projects):
    p1, p2 = projects
    add_contact(p1["id"], name="Shared Name", company="Same Corp")
    add_contact(p2["id"], name="Other Person", company="Same Corp")

    results = search_contacts("Same Corp")
    assert len(results) == 2


def test_search_contacts_filter_by_project_id(projects):
    p1, p2 = projects
    add_contact(p1["id"], name="P1 Contact", company="Corp X")
    add_contact(p2["id"], name="P2 Contact", company="Corp X")

    results = search_contacts("Corp X", project_id=p1["id"])
    assert len(results) == 1
    assert results[0]["name"] == "P1 Contact"


def test_search_contacts_empty_query_returns_empty(projects):
    p1, _ = projects
    add_contact(p1["id"], name="Someone")

    assert search_contacts("") == []
