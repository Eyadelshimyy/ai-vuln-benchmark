"""Tests for static HTML report generation (issue #26)."""
from __future__ import annotations

from pathlib import Path

import pytest
from tools.contacts import add_contact
from tools.notes import add_note
from tools.project_links import link_project
from tools.projects import create_project
from tools.report import generate_report


@pytest.fixture()
def populated_project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p = create_project(
        "Acme GmbH",
        project_type="consulting",
        description="Beratungsprojekt Q2",
        phase="Konzeption",
        go_live="2026-07-01",
        budget="50.000 EUR",
    )
    add_contact(
        p["id"], "Alice Muster", role="Projektleiterin",
        contact_type="internal", email="alice@example.com", phone="+49 123 456",
    )
    add_contact(
        p["id"], "Bob Extern", role="Berater",
        contact_type="external", company="Beratungshaus AG", email="bob@beratungshaus.de",
    )
    add_note(p["id"], "Kickoff-Meeting", "Wir haben Meilensteine besprochen.", note_type="meeting-notes")
    add_note(p["id"], "Entscheidung Go-Live", "Go-Live wurde auf Q3 gesetzt.", note_type="decision")
    add_note(p["id"], "Aufgabe: Anforderungen sammeln", "Bis Ende Mai.", note_type="action-item")
    return p


class TestGenerateReportFull:
    def test_creates_html_file(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        result = generate_report(populated_project["id"], report_type="full", output_path=dest)

        assert result["path"] == dest
        assert Path(dest).exists()

    def test_returns_project_name(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        result = generate_report(populated_project["id"], report_type="full", output_path=dest)

        assert result["project"] == "Acme GmbH"

    def test_html_contains_project_name(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Acme GmbH" in html

    def test_html_contains_contacts(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Alice Muster" in html
        assert "Bob Extern" in html

    def test_html_contains_notes(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Kickoff-Meeting" in html
        assert "Entscheidung Go-Live" in html

    def test_html_contains_action_items(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Aufgabe: Anforderungen sammeln" in html

    def test_html_is_valid_html_structure(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_html_contains_chartjs(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "chart.js" in html.lower() or "Chart.js" in html or "cdn.jsdelivr.net" in html

    def test_default_output_path(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report.Path.home", lambda: tmp_path)
        result = generate_report(populated_project["id"], report_type="full")

        assert result["path"].endswith(".html")
        assert "reports" in result["path"]
        assert "acme-gmbh" in result["path"]

    def test_full_and_summary_default_paths_differ(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report.Path.home", lambda: tmp_path)
        full = generate_report(populated_project["id"], report_type="full")
        summary = generate_report(populated_project["id"], report_type="summary")
        assert full["path"] != summary["path"]
        assert "full" in full["path"]
        assert "summary" in summary["path"]

    def test_nonexistent_project_raises(self):
        with pytest.raises(ValueError, match="not found"):
            generate_report(99999, report_type="full", output_path="/tmp/nope.html")

    def test_invalid_report_type_raises(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        with pytest.raises(ValueError, match="report_type"):
            generate_report(populated_project["id"], report_type="invalid", output_path=dest)

    def test_html_escapes_user_controlled_content(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        p = create_project("<script>alert(1)</script>")
        add_note(p["id"], "note", "<img src=x onerror=alert(1)>", note_type="note")

        dest = str(tmp_path / "report.html")
        generate_report(p["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
        assert "<img src=x onerror=alert(1)>" not in html

    def test_no_links_section_when_unlinked(self, populated_project, tmp_path):
        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Nachfolger von" not in html
        assert "Vorgänger von" not in html
        assert "Verknüpft mit" not in html

    def test_shows_successor_relation(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        predecessor = create_project("Acme GmbH Vorgänger")
        link_project(populated_project["slug"], predecessor["slug"], "successor")

        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Nachfolger von" in html
        assert "Acme GmbH Vorgänger" in html

    def test_shows_predecessor_relation(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        successor = create_project("Acme GmbH Nachfolger")
        link_project(successor["slug"], populated_project["slug"], "successor")

        dest = str(tmp_path / "report.html")
        generate_report(populated_project["id"], report_type="full", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Vorgänger von" in html
        assert "Acme GmbH Nachfolger" in html


class TestGenerateReportSummary:
    def test_creates_html_file(self, populated_project, tmp_path):
        dest = str(tmp_path / "summary.html")
        result = generate_report(populated_project["id"], report_type="summary", output_path=dest)

        assert result["path"] == dest
        assert Path(dest).exists()

    def test_html_contains_project_name(self, populated_project, tmp_path):
        dest = str(tmp_path / "summary.html")
        generate_report(populated_project["id"], report_type="summary", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Acme GmbH" in html


class TestGenerateReportAllProjects:
    def test_creates_html_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        create_project("Projekt Alpha", project_type="generic")
        create_project("Projekt Beta", project_type="consulting")

        dest = str(tmp_path / "all.html")
        result = generate_report(None, report_type="all-projects", output_path=dest)

        assert result["path"] == dest
        assert Path(dest).exists()

    def test_html_contains_all_project_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        create_project("Projekt Alpha", project_type="generic")
        create_project("Projekt Beta", project_type="consulting")

        dest = str(tmp_path / "all.html")
        generate_report(None, report_type="all-projects", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Projekt Alpha" in html
        assert "Projekt Beta" in html

    def test_default_output_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        monkeypatch.setattr("tools.report.Path.home", lambda: tmp_path)
        create_project("Projekt Alpha", project_type="generic")

        result = generate_report(None, report_type="all-projects")

        assert result["path"].endswith(".html")
        assert "all-projects" in result["path"]

    def test_html_shows_relation_between_projects(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        alpha = create_project("Projekt Alpha", project_type="generic")
        beta = create_project("Projekt Beta", project_type="consulting")
        link_project(beta["slug"], alpha["slug"], "successor")

        dest = str(tmp_path / "all.html")
        generate_report(None, report_type="all-projects", output_path=dest)

        html = Path(dest).read_text(encoding="utf-8")
        assert "Nachfolger von" in html
        assert "Vorgänger von" in html
