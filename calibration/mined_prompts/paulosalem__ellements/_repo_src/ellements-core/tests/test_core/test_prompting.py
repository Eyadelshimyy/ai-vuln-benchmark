"""Tests for PersonaLibrary, GuidelineLibrary, and PromptContext."""

from __future__ import annotations

import json

import pytest
from ellements.core import (
    GuidelineLibrary,
    GuidelineNotFoundError,
    PersonaLibrary,
    PersonaNotFoundError,
)
from ellements.core.prompting import PromptContext

# ── PersonaLibrary ─────────────────────────────────────────────────


def _write_json_persona(folder, filename: str, *, name: str) -> None:
    payload = {
        "persona": {
            "name": name,
            "occupation": {"title": "Engineer"},
            "preferences": {"interests": ["AI"]},
        }
    }
    (folder / filename).write_text(json.dumps(payload), encoding="utf-8")


def _write_md_persona(folder, filename: str, *, name: str) -> None:
    text = (
        f"---\nname: {name}\nstyle: terse\n---\n\nLong-form description here."
    )
    (folder / filename).write_text(text, encoding="utf-8")


class TestPersonaLibrary:
    def test_loads_json_and_md(self, tmp_path):
        folder = tmp_path / "personas"
        folder.mkdir()
        _write_json_persona(folder, "alice.json", name="Alice")
        _write_md_persona(folder, "bob.md", name="Bob")

        library = PersonaLibrary(folder)
        assert library.list_ids() == ["alice", "bob"]
        assert "Name: Alice" in library.load("alice")
        assert "Name: Bob" in library.load("bob")

    def test_unknown_id_raises(self, tmp_path):
        folder = tmp_path / "personas"
        folder.mkdir()
        _write_json_persona(folder, "alice.json", name="Alice")
        library = PersonaLibrary(folder)
        with pytest.raises(PersonaNotFoundError):
            library.load("missing")

    def test_id_collision_raises(self, tmp_path):
        folder = tmp_path / "personas"
        folder.mkdir()
        _write_json_persona(folder, "alice.json", name="Alice")
        _write_md_persona(folder, "alice.md", name="Alice")
        with pytest.raises(ValueError, match="collision"):
            PersonaLibrary(folder)

    def test_add_from_path_outside_folder(self, tmp_path):
        folder = tmp_path / "personas"
        folder.mkdir()
        _write_json_persona(folder, "alice.json", name="Alice")
        library = PersonaLibrary(folder)

        extra = tmp_path / "extra"
        extra.mkdir()
        _write_md_persona(extra, "charlie.md", name="Charlie")
        registered = library.add_from_path(extra / "charlie.md")

        assert registered == "charlie"
        assert "charlie" in library
        assert "Name: Charlie" in library.load("charlie")

    def test_contains(self, tmp_path):
        folder = tmp_path / "personas"
        folder.mkdir()
        _write_json_persona(folder, "alice.json", name="Alice")
        library = PersonaLibrary(folder)
        assert "alice" in library
        assert "missing" not in library

    def test_unsupported_extension_rejected(self, tmp_path):
        folder = tmp_path / "personas"
        folder.mkdir()
        _write_json_persona(folder, "alice.json", name="Alice")
        library = PersonaLibrary(folder)

        bad = tmp_path / "bad.yml"
        bad.write_text("name: Bad", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported persona file extension"):
            library.add_from_path(bad)


# ── GuidelineLibrary ────────────────────────────────────────────────


class TestGuidelineLibrary:
    def test_loads_md_and_json(self, tmp_path):
        folder = tmp_path / "guidelines"
        folder.mkdir()
        (folder / "concise.md").write_text("Be concise.", encoding="utf-8")
        (folder / "verbose.json").write_text(
            json.dumps({"name": "verbose", "text": "Use long prose."}),
            encoding="utf-8",
        )

        library = GuidelineLibrary(folder)
        assert library.list_ids() == ["concise", "verbose"]
        assert library.load("concise") == "Be concise."
        assert library.load("verbose") == "Use long prose."

    def test_id_collision_raises(self, tmp_path):
        folder = tmp_path / "guidelines"
        folder.mkdir()
        (folder / "concise.md").write_text("Be concise.", encoding="utf-8")
        (folder / "concise.json").write_text(
            json.dumps({"name": "concise", "text": "Be brief."}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="collision"):
            GuidelineLibrary(folder)

    def test_unknown_id_raises(self, tmp_path):
        folder = tmp_path / "guidelines"
        folder.mkdir()
        (folder / "concise.md").write_text("Be concise.", encoding="utf-8")
        library = GuidelineLibrary(folder)
        with pytest.raises(GuidelineNotFoundError):
            library.load("missing")


# ── PromptContext ──────────────────────────────────────────────────


class TestPromptContext:
    def test_set_persona_by_id_against_library(self, tmp_path):
        folder = tmp_path / "personas"
        folder.mkdir()
        _write_json_persona(folder, "alice.json", name="Alice")

        ctx = PromptContext()
        ctx.attach_persona_library(PersonaLibrary(folder))
        ctx.set_persona_by_id("alice")
        assert ctx.current_persona_id == "alice"
        assert "Name: Alice" in ctx.get_persona_text()

    def test_set_persona_from_data_no_library(self):
        ctx = PromptContext()
        ctx.set_persona_from_data(
            {"persona": {"name": "Inline", "occupation": {"title": "Tester"}}}
        )
        assert ctx.current_persona_id is None
        assert "Name: Inline" in ctx.get_persona_text()

    def test_set_persona_from_path_without_library(self, tmp_path):
        path = tmp_path / "p.md"
        path.write_text("---\nname: Onefile\n---\nbody.", encoding="utf-8")

        ctx = PromptContext()
        ctx.set_persona_from_path(path)
        assert ctx.current_persona_id is None
        assert "Name: Onefile" in ctx.get_persona_text()

    def test_set_guideline_by_id_against_library(self, tmp_path):
        folder = tmp_path / "guidelines"
        folder.mkdir()
        (folder / "brief.md").write_text("Be concise.", encoding="utf-8")

        ctx = PromptContext()
        ctx.attach_guideline_library(GuidelineLibrary(folder))
        ctx.set_guideline_by_id("brief")
        assert ctx.current_guideline_id == "brief"
        assert ctx.apply_guideline("Hello") == "Be concise.\n\nHello"

    def test_clear_guideline(self, tmp_path):
        folder = tmp_path / "guidelines"
        folder.mkdir()
        (folder / "brief.md").write_text("Be brief.", encoding="utf-8")

        ctx = PromptContext()
        ctx.attach_guideline_library(GuidelineLibrary(folder))
        ctx.set_guideline_by_id("brief")
        ctx.clear_guideline()
        assert ctx.current_guideline_id is None
        assert ctx.apply_guideline("Hello") == "Hello"

    def test_set_persona_by_id_without_library_raises(self):
        ctx = PromptContext()
        with pytest.raises(ValueError, match="No persona library attached"):
            ctx.set_persona_by_id("anything")

    def test_set_guideline_by_id_without_library_raises(self):
        ctx = PromptContext()
        with pytest.raises(ValueError, match="No guideline library attached"):
            ctx.set_guideline_by_id("anything")
