"""Tests for the knowledge tool functions (save/get/list/delete/sync)."""
import pytest
from tools.knowledge import (
    delete_knowledge,
    get_all_knowledge,
    get_knowledge,
    list_knowledge,
    save_knowledge,
    sync_knowledge_templates,
)


@pytest.fixture
def knowledge_root(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.knowledge.get_knowledge_root", lambda: tmp_path)
    return tmp_path


def test_save_and_get_roundtrip(knowledge_root):
    saved = save_knowledge("generic", "governance", "# Governance\n\nContent here.")
    assert saved["topic"] == "governance"
    assert saved["title"] == "Governance"
    assert "Content here." in saved["content"]

    fetched = get_knowledge("generic", "governance")
    assert fetched is not None
    assert fetched["content"] == saved["content"]


def test_get_knowledge_not_found(knowledge_root):
    assert get_knowledge("generic", "nonexistent") is None


def test_save_overwrites_existing(knowledge_root):
    save_knowledge("generic", "process", "# Process\n\nVersion 1.")
    updated = save_knowledge("generic", "process", "# Process\n\nVersion 2.")
    assert "Version 2." in updated["content"]


def test_list_knowledge_returns_topics(knowledge_root):
    save_knowledge("it-project", "architecture", "# Architecture\n\nDiagram.")
    save_knowledge("it-project", "charter", "# Charter\n\nGoals.")

    items = list_knowledge("it-project")
    topics = {i["topic"] for i in items}
    assert topics == {"architecture", "charter"}


def test_list_knowledge_extracts_h1_title(knowledge_root):
    save_knowledge("generic", "roles", "# Role Matrix\n\nContent.")

    items = list_knowledge("generic")
    assert items[0]["title"] == "Role Matrix"


def test_list_knowledge_falls_back_to_stem_when_no_h1(knowledge_root):
    save_knowledge("generic", "notes", "No heading here.")

    items = list_knowledge("generic")
    assert items[0]["title"] == "notes"


def test_list_knowledge_empty_returns_empty(knowledge_root):
    assert list_knowledge("generic") == []


def test_get_all_knowledge_loads_full_content(knowledge_root):
    save_knowledge("consulting", "engagement", "# Engagement\n\nFull content.")
    save_knowledge("consulting", "process", "# Process\n\nSteps.")

    all_items = get_all_knowledge("consulting")
    assert len(all_items) == 2
    assert all("content" in item for item in all_items)


def test_delete_knowledge_removes_file(knowledge_root):
    save_knowledge("generic", "to-delete", "# Delete Me")

    result = delete_knowledge("generic", "to-delete")

    assert result is True
    assert get_knowledge("generic", "to-delete") is None


def test_delete_knowledge_not_found_returns_false(knowledge_root):
    assert delete_knowledge("generic", "does-not-exist") is False


def test_save_knowledge_path_traversal_rejected(knowledge_root):
    with pytest.raises(ValueError, match="Invalid topic"):
        save_knowledge("generic", "../../etc/passwd", "malicious")


def test_get_knowledge_path_traversal_rejected(knowledge_root):
    with pytest.raises(ValueError, match="Invalid topic"):
        get_knowledge("generic", "../escape")


def test_delete_knowledge_path_traversal_rejected(knowledge_root):
    with pytest.raises(ValueError, match="Invalid topic"):
        delete_knowledge("generic", "../../sensitive")


# ---------------------------------------------------------------------------
# sync_knowledge_templates
# ---------------------------------------------------------------------------

@pytest.fixture
def plugin_templates(tmp_path):
    """Create a fake plugin templates directory with two files."""
    tmpl_root = tmp_path / "plugin_templates"
    (tmpl_root / "it-project").mkdir(parents=True)
    (tmpl_root / "it-project" / "charter.md").write_text("# IT Charter\n\nContent.", encoding="utf-8")
    (tmpl_root / "consulting").mkdir(parents=True)
    (tmpl_root / "consulting" / "engagement.md").write_text("# Engagement\n\nContent.", encoding="utf-8")
    return tmpl_root


def test_sync_detects_new_files(knowledge_root, plugin_templates):
    report = sync_knowledge_templates(force=False, plugin_templates_dir=plugin_templates)

    assert len(report["items"]) == 2
    statuses = {i["path"]: i["status"] for i in report["items"]}
    assert statuses["it-project/charter.md"] == "new"
    assert statuses["consulting/engagement.md"] == "new"
    assert report["synced"] == []


def test_sync_detects_up_to_date(knowledge_root, plugin_templates, monkeypatch):
    monkeypatch.setattr("tools.knowledge.get_knowledge_root", lambda: knowledge_root)
    # Copy plugin file to user root so it matches
    user_file = knowledge_root / "it-project" / "charter.md"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("# IT Charter\n\nContent.", encoding="utf-8")

    report = sync_knowledge_templates(force=False, plugin_templates_dir=plugin_templates)

    charter_item = next(i for i in report["items"] if "charter" in i["path"])
    assert charter_item["status"] == "up-to-date"


def test_sync_detects_changed_file(knowledge_root, plugin_templates, monkeypatch):
    monkeypatch.setattr("tools.knowledge.get_knowledge_root", lambda: knowledge_root)
    user_file = knowledge_root / "consulting" / "engagement.md"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("# Engagement\n\nOld content.", encoding="utf-8")

    report = sync_knowledge_templates(force=False, plugin_templates_dir=plugin_templates)

    eng_item = next(i for i in report["items"] if "engagement" in i["path"])
    assert eng_item["status"] == "newer-version-available"
    assert "local_bytes" in eng_item


def test_sync_force_copies_new_and_changed(knowledge_root, plugin_templates, monkeypatch):
    monkeypatch.setattr("tools.knowledge.get_knowledge_root", lambda: knowledge_root)

    report = sync_knowledge_templates(force=True, plugin_templates_dir=plugin_templates)

    assert len(report["synced"]) == 2
    charter = knowledge_root / "it-project" / "charter.md"
    assert charter.exists()
    assert "IT Charter" in charter.read_text()


def test_sync_force_skips_up_to_date(knowledge_root, plugin_templates, monkeypatch):
    monkeypatch.setattr("tools.knowledge.get_knowledge_root", lambda: knowledge_root)
    user_file = knowledge_root / "it-project" / "charter.md"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("# IT Charter\n\nContent.", encoding="utf-8")

    report = sync_knowledge_templates(force=True, plugin_templates_dir=plugin_templates)

    # Only the engagement.md (new) should be synced, not the up-to-date charter
    assert "it-project/charter.md" not in report["synced"]
    assert "consulting/engagement.md" in report["synced"]
