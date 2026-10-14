"""Tests for docs_writer — UTF-8 encoding and file-write correctness."""
from __future__ import annotations

from pathlib import Path

from tools.docs_writer import write_note_to_disk


def test_write_note_non_cp1252_content(tmp_path):
    """Bug 1: write_text without encoding= fails on Windows (cp1252) for non-ASCII chars."""
    content = "Go-Live 15.09.2026 → ca. 2 Monate"
    file_path = write_note_to_disk(str(tmp_path), "Test", content, "note")

    written = Path(file_path).read_text(encoding="utf-8")
    assert "→" in written


def test_write_note_file_is_utf8(tmp_path):
    """The written file must be valid UTF-8 (not cp1252 or locale default)."""
    content = "Symbols: → ✓ — 🎉"
    file_path = write_note_to_disk(str(tmp_path), "Unicode Test", content, "note")

    raw = Path(file_path).read_bytes()
    # If the file is UTF-8 this must not raise
    raw.decode("utf-8")


def test_append_action_item_non_cp1252(tmp_path):
    """Bug 1: action-item TODO file write must handle non-cp1252 chars."""
    todo_path = tmp_path / "misc" / "todo.md"
    todo_path.parent.mkdir(parents=True)

    write_note_to_disk(str(tmp_path), "Prüfe → Ergebnis", "ignored", "action-item")

    written = todo_path.read_text(encoding="utf-8")
    assert "→" in written


def test_append_action_item_existing_file_non_cp1252(tmp_path):
    """Appending to an existing TODO file must use UTF-8 for both read and write."""
    todo_path = tmp_path / "misc" / "todo.md"
    todo_path.parent.mkdir(parents=True)
    # Pre-seed with UTF-8 content
    todo_path.write_text("# Action Items\n\n## Open\n\n## Done\n", encoding="utf-8")

    write_note_to_disk(str(tmp_path), "Follow-up → done", "ignored", "action-item")

    written = todo_path.read_text(encoding="utf-8")
    assert "→" in written
