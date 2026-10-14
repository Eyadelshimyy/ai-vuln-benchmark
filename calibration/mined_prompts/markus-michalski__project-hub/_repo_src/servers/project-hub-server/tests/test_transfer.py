"""Tests for project export/import (multi-user handoff)."""
import json

import pytest
from tools.contacts import add_contact
from tools.notes import add_note
from tools.project_links import get_links_for_project, link_project
from tools.projects import create_project
from tools.transfer import export_project, import_project


@pytest.fixture()
def populated_project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p = create_project("Export Test Corp", project_type="generic", description="A test project")
    add_contact(p["id"], "Alice", role="PM", contact_type="internal", email="alice@example.com")
    add_contact(p["id"], "Bob", role="Dev", contact_type="external", company="Acme")
    add_note(p["id"], "Kickoff Notes", "We discussed milestones.", note_type="meeting-notes")
    add_note(p["id"], "Decision", "Go-live set to Q3.", note_type="decision")
    return p


class TestExportProject:
    def test_export_creates_json_file(self, populated_project, tmp_path):
        dest = str(tmp_path / "export.json")
        result = export_project(populated_project["id"], dest)

        assert result["path"] == dest
        assert result["project"] == "Export Test Corp"
        assert result["contacts"] == 2
        assert result["notes"] == 2

    def test_export_file_content(self, populated_project, tmp_path):
        dest = tmp_path / "export.json"
        export_project(populated_project["id"], str(dest))

        payload = json.loads(dest.read_text())
        assert payload["export_version"] == 1
        assert payload["project"]["name"] == "Export Test Corp"
        assert len(payload["contacts"]) == 2
        assert len(payload["notes"]) == 2
        assert "exported_at" in payload

    def test_export_default_path(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.transfer.Path.home", lambda: tmp_path)
        result = export_project(populated_project["id"])
        assert result["path"].endswith(".json")

    def test_export_nonexistent_project_raises(self):
        with pytest.raises(ValueError, match="not found"):
            export_project(99999)

    def test_export_excludes_old_ids(self, populated_project, tmp_path):
        dest = tmp_path / "export.json"
        export_project(populated_project["id"], str(dest))

        payload = json.loads(dest.read_text())
        # IDs are present in the export (they are included for reference but stripped on import)
        assert "id" in payload["project"]


class TestImportProject:
    def _make_export(self, populated_project, tmp_path) -> str:
        dest = str(tmp_path / "export.json")
        export_project(populated_project["id"], dest)
        return dest

    def test_import_roundtrip(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        export_path = self._make_export(populated_project, tmp_path)

        # Delete original so we can re-import without conflict
        from tools.db import db_connection
        with db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (populated_project["id"],))
            conn.commit()

        result = import_project(export_path, merge_strategy="skip")

        assert result["imported"] is True
        assert result["project"] == "Export Test Corp"
        assert result["contacts"] == 2
        assert result["notes"] == 2

    def test_import_skip_on_conflict(self, populated_project, tmp_path):
        export_path = self._make_export(populated_project, tmp_path)
        result = import_project(export_path, merge_strategy="skip")

        assert result["imported"] is False
        assert "already exists" in result["reason"]
        assert result["links"] == 0
        assert result["links_not_restored"] == []

    def test_import_rename_on_conflict(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        export_path = self._make_export(populated_project, tmp_path)
        result = import_project(export_path, merge_strategy="rename")

        assert result["imported"] is True
        assert "imported" in result["slug"]
        assert result["project"] != "Export Test Corp"  # name got suffix

    def test_import_overwrite_on_conflict(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        export_path = self._make_export(populated_project, tmp_path)
        result = import_project(export_path, merge_strategy="overwrite")

        assert result["imported"] is True
        assert result["slug"] == populated_project["slug"]

    def test_import_invalid_strategy_raises(self, populated_project, tmp_path):
        export_path = self._make_export(populated_project, tmp_path)
        with pytest.raises(ValueError, match="merge_strategy"):
            import_project(export_path, merge_strategy="magic")

    def test_import_missing_file_raises(self):
        with pytest.raises(ValueError, match="not found"):
            import_project("/tmp/nonexistent_export_xyz.json")

    def test_import_wrong_version_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({"export_version": 99, "project": {}, "contacts": [], "notes": []}))
        with pytest.raises(ValueError, match="export_version"):
            import_project(str(bad_file))


class TestExportImportLinks:
    def test_export_includes_links(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        other = create_project("Export Test Corp 2")
        link_project(populated_project["slug"], other["slug"], "successor")

        dest = tmp_path / "export.json"
        result = export_project(populated_project["id"], str(dest))

        assert result["links"] == 1
        payload = json.loads(dest.read_text())
        assert payload["links"] == [
            {"relation": "successor", "slug": other["slug"], "name": other["name"]}
        ]

    def test_export_no_links(self, populated_project, tmp_path):
        dest = tmp_path / "export.json"
        result = export_project(populated_project["id"], str(dest))

        assert result["links"] == 0
        payload = json.loads(dest.read_text())
        assert payload["links"] == []

    def test_import_restores_link_when_target_exists(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        other = create_project("Export Test Corp 2")
        link_project(populated_project["slug"], other["slug"], "successor")

        dest = str(tmp_path / "export.json")
        export_project(populated_project["id"], dest)

        from tools.db import db_connection
        with db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (populated_project["id"],))
            conn.commit()

        result = import_project(dest, merge_strategy="skip")

        assert result["links"] == 1
        assert result["links_not_restored"] == []
        restored = get_links_for_project(result["project_id"])
        assert restored == [
            {"relation": "successor", "project": {"id": other["id"], "slug": other["slug"], "name": other["name"]}}
        ]

    def test_import_skips_link_when_target_missing(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        other = create_project("Export Test Corp 2")
        link_project(populated_project["slug"], other["slug"], "successor")

        dest = str(tmp_path / "export.json")
        export_project(populated_project["id"], dest)

        from tools.db import db_connection
        with db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (populated_project["id"],))
            conn.execute("DELETE FROM projects WHERE id = ?", (other["id"],))
            conn.commit()

        result = import_project(dest, merge_strategy="skip")

        assert result["links"] == 0
        assert len(result["links_not_restored"]) == 1
        assert result["links_not_restored"][0]["slug"] == other["slug"]
        assert result["links_not_restored"][0]["relation"] == "successor"

    def test_import_restores_link_with_rename_strategy(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        other = create_project("Export Test Corp 2")
        link_project(populated_project["slug"], other["slug"], "successor")

        dest = str(tmp_path / "export.json")
        export_project(populated_project["id"], dest)

        # Do NOT delete the original — forces merge_strategy="rename" to kick in,
        # which mutates project["slug"] before the restore loop runs.
        result = import_project(dest, merge_strategy="rename")

        assert result["imported"] is True
        assert "imported" in result["slug"]
        assert result["links"] == 1
        assert result["links_not_restored"] == []
        restored = get_links_for_project(result["project_id"])
        assert restored == [
            {"relation": "successor", "project": {"id": other["id"], "slug": other["slug"], "name": other["name"]}}
        ]

    def test_import_restores_link_with_overwrite_strategy(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        other = create_project("Export Test Corp 2")
        link_project(populated_project["slug"], other["slug"], "successor")

        dest = str(tmp_path / "export.json")
        export_project(populated_project["id"], dest)

        # Do NOT delete the original — forces merge_strategy="overwrite", which
        # deletes (cascading the old project_links row) and re-inserts under the
        # same slug but a new project id.
        result = import_project(dest, merge_strategy="overwrite")

        assert result["imported"] is True
        assert result["slug"] == populated_project["slug"]
        assert result["links"] == 1
        assert result["links_not_restored"] == []
        restored = get_links_for_project(result["project_id"])
        assert restored == [
            {"relation": "successor", "project": {"id": other["id"], "slug": other["slug"], "name": other["name"]}}
        ]

    def test_import_restores_multiple_mixed_relation_links(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        successor = create_project("Export Test Corp Successor")
        related = create_project("Export Test Corp Related")
        link_project(populated_project["slug"], successor["slug"], "successor")
        link_project(populated_project["slug"], related["slug"], "related")

        dest = str(tmp_path / "export.json")
        export_project(populated_project["id"], dest)

        from tools.db import db_connection
        with db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (populated_project["id"],))
            conn.commit()

        result = import_project(dest, merge_strategy="skip")

        assert result["links"] == 2
        assert result["links_not_restored"] == []
        restored = get_links_for_project(result["project_id"])
        assert {(link["relation"], link["project"]["slug"]) for link in restored} == {
            ("successor", successor["slug"]),
            ("related", related["slug"]),
        }

    def test_import_link_missing_name_falls_back_to_slug_in_report(self, populated_project, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        other = create_project("Export Test Corp 2")
        link_project(populated_project["slug"], other["slug"], "successor")

        dest = tmp_path / "export.json"
        export_project(populated_project["id"], str(dest))
        payload = json.loads(dest.read_text())
        del payload["links"][0]["name"]
        dest.write_text(json.dumps(payload))

        from tools.db import db_connection
        with db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (populated_project["id"],))
            conn.execute("DELETE FROM projects WHERE id = ?", (other["id"],))
            conn.commit()

        result = import_project(str(dest), merge_strategy="skip")

        assert result["links"] == 0
        assert result["links_not_restored"] == [
            {
                "slug": other["slug"],
                "name": other["slug"],
                "relation": "successor",
                "reason": result["links_not_restored"][0]["reason"],
            }
        ]

    def test_import_invalid_relation_type_in_export_is_reported_not_restored(
        self, populated_project, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
        other = create_project("Export Test Corp 2")
        link_project(populated_project["slug"], other["slug"], "successor")

        dest = tmp_path / "export.json"
        export_project(populated_project["id"], str(dest))
        payload = json.loads(dest.read_text())
        payload["links"][0]["relation"] = "sibling"
        dest.write_text(json.dumps(payload))

        from tools.db import db_connection
        with db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (populated_project["id"],))
            conn.commit()

        result = import_project(str(dest), merge_strategy="skip")

        assert result["links"] == 0
        assert len(result["links_not_restored"]) == 1
        assert result["links_not_restored"][0]["relation"] == "sibling"
        assert "relation_type" in result["links_not_restored"][0]["reason"]

    def test_import_old_export_without_links_key_still_works(self, populated_project, tmp_path):
        dest = tmp_path / "export.json"
        export_project(populated_project["id"], str(dest))
        payload = json.loads(dest.read_text())
        del payload["links"]
        dest.write_text(json.dumps(payload))

        from tools.db import db_connection
        with db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (populated_project["id"],))
            conn.commit()

        result = import_project(str(dest), merge_strategy="skip")

        assert result["imported"] is True
        assert result["links"] == 0
        assert result["links_not_restored"] == []
