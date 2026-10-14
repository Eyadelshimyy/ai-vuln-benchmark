"""Tests for project-to-project relations (successor / predecessor / related)."""
import pytest
from tools.project_links import get_links_for_project, link_project, unlink_project
from tools.projects import create_project


def _mk(name, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    return create_project(name)


def test_link_project_successor_shows_from_both_sides(tmp_path, monkeypatch):
    joybuy = _mk("Joybuy", tmp_path, monkeypatch)
    joybuy2 = _mk("Joybuy 2", tmp_path, monkeypatch)

    link_project(joybuy2["slug"], joybuy["slug"], "successor")

    successor_links = get_links_for_project(joybuy2["id"])
    assert successor_links == [
        {"relation": "successor", "project": {"id": joybuy["id"], "slug": joybuy["slug"], "name": joybuy["name"]}}
    ]

    predecessor_links = get_links_for_project(joybuy["id"])
    assert predecessor_links == [
        {"relation": "predecessor", "project": {"id": joybuy2["id"], "slug": joybuy2["slug"], "name": joybuy2["name"]}}
    ]


def test_link_project_predecessor_normalizes_to_successor_on_other_side(tmp_path, monkeypatch):
    joybuy = _mk("Joybuy", tmp_path, monkeypatch)
    joybuy2 = _mk("Joybuy 2", tmp_path, monkeypatch)

    # "Joybuy is predecessor of Joybuy 2" == "Joybuy 2 is successor of Joybuy"
    link_project(joybuy["slug"], joybuy2["slug"], "predecessor")

    assert get_links_for_project(joybuy["id"]) == [
        {"relation": "predecessor", "project": {"id": joybuy2["id"], "slug": joybuy2["slug"], "name": joybuy2["name"]}}
    ]
    assert get_links_for_project(joybuy2["id"]) == [
        {"relation": "successor", "project": {"id": joybuy["id"], "slug": joybuy["slug"], "name": joybuy["name"]}}
    ]


def test_link_project_related_is_symmetric(tmp_path, monkeypatch):
    a = _mk("Parkbee", tmp_path, monkeypatch)
    b = _mk("Parkbee NL", tmp_path, monkeypatch)

    link_project(a["slug"], b["slug"], "related")

    assert get_links_for_project(a["id"]) == [
        {"relation": "related", "project": {"id": b["id"], "slug": b["slug"], "name": b["name"]}}
    ]
    assert get_links_for_project(b["id"]) == [
        {"relation": "related", "project": {"id": a["id"], "slug": a["slug"], "name": a["name"]}}
    ]


def test_get_links_for_project_multiple_links_sorted_by_name(tmp_path, monkeypatch):
    hub = _mk("Hub Project", tmp_path, monkeypatch)
    zeta = _mk("Zeta", tmp_path, monkeypatch)
    alpha = _mk("Alpha", tmp_path, monkeypatch)

    link_project(hub["slug"], zeta["slug"], "related")
    link_project(hub["slug"], alpha["slug"], "successor")

    links = get_links_for_project(hub["id"])

    assert [link["project"]["name"] for link in links] == ["Alpha", "Zeta"]
    assert links[0]["relation"] == "successor"
    assert links[1]["relation"] == "related"


def test_link_project_resolves_by_name_case_insensitive(tmp_path, monkeypatch):
    a = _mk("Case Test A", tmp_path, monkeypatch)
    b = _mk("Case Test B", tmp_path, monkeypatch)

    link_project("case test a", "CASE TEST B", "related")

    assert get_links_for_project(a["id"])[0]["project"]["id"] == b["id"]


def test_link_project_returns_link_summary(tmp_path, monkeypatch):
    a = _mk("Project A", tmp_path, monkeypatch)
    b = _mk("Project B", tmp_path, monkeypatch)

    result = link_project(a["slug"], b["slug"], "successor")

    assert result["relation_type"] == "successor"
    assert result["project"]["slug"] == a["slug"]
    assert result["related_project"]["slug"] == b["slug"]


def test_link_project_self_link_raises(tmp_path, monkeypatch):
    a = _mk("Solo Project", tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="itself"):
        link_project(a["slug"], a["slug"], "related")


def test_link_project_unknown_identifier_raises(tmp_path, monkeypatch):
    a = _mk("Known Project", tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="not found"):
        link_project("ghost-project", a["slug"], "related")

    with pytest.raises(ValueError, match="not found"):
        link_project(a["slug"], "ghost-project", "related")


def test_link_project_invalid_relation_type_raises(tmp_path, monkeypatch):
    a = _mk("A Project", tmp_path, monkeypatch)
    b = _mk("B Project", tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="relation_type"):
        link_project(a["slug"], b["slug"], "sibling")


def test_link_project_duplicate_raises(tmp_path, monkeypatch):
    a = _mk("A Project", tmp_path, monkeypatch)
    b = _mk("B Project", tmp_path, monkeypatch)

    link_project(a["slug"], b["slug"], "successor")

    with pytest.raises(ValueError, match="already linked"):
        link_project(a["slug"], b["slug"], "related")


def test_link_project_duplicate_reverse_direction_raises(tmp_path, monkeypatch):
    a = _mk("A Project", tmp_path, monkeypatch)
    b = _mk("B Project", tmp_path, monkeypatch)

    link_project(a["slug"], b["slug"], "successor")

    with pytest.raises(ValueError, match="already linked"):
        link_project(b["slug"], a["slug"], "related")


def test_unlink_project_removes_link(tmp_path, monkeypatch):
    a = _mk("A Project", tmp_path, monkeypatch)
    b = _mk("B Project", tmp_path, monkeypatch)
    link_project(a["slug"], b["slug"], "successor")

    result = unlink_project(a["slug"], b["slug"])

    assert result is True
    assert get_links_for_project(a["id"]) == []
    assert get_links_for_project(b["id"]) == []


def test_unlink_project_direction_independent(tmp_path, monkeypatch):
    a = _mk("A Project", tmp_path, monkeypatch)
    b = _mk("B Project", tmp_path, monkeypatch)
    link_project(a["slug"], b["slug"], "successor")

    # unlink called with swapped identifier order must still find and remove the link
    result = unlink_project(b["slug"], a["slug"])

    assert result is True
    assert get_links_for_project(a["id"]) == []


def test_unlink_project_not_linked_returns_false(tmp_path, monkeypatch):
    a = _mk("A Project", tmp_path, monkeypatch)
    b = _mk("B Project", tmp_path, monkeypatch)

    assert unlink_project(a["slug"], b["slug"]) is False


def test_unlink_project_unknown_identifier_returns_false(tmp_path, monkeypatch):
    a = _mk("A Project", tmp_path, monkeypatch)

    assert unlink_project(a["slug"], "ghost-project") is False
    assert unlink_project("ghost-project", a["slug"]) is False


def test_project_deletion_cascades_link(tmp_path, monkeypatch):
    from tools.db import db_connection

    a = _mk("A Project", tmp_path, monkeypatch)
    b = _mk("B Project", tmp_path, monkeypatch)
    link_project(a["slug"], b["slug"], "successor")

    with db_connection() as conn:
        with conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (b["id"],))

    assert get_links_for_project(a["id"]) == []
