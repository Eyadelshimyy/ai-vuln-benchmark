"""Tests for the application-agnostic AgentTUI plug-in surface."""

from __future__ import annotations

from pathlib import Path

import pytest
from ellements.cli import (
    BUILTIN_COMMANDS,
    AgentRunner,
    AgentTUI,
    FileSaveHandler,
    GuidelineProvider,
    Mode,
    PersonaProvider,
    RunResult,
    SaveHandler,
    SlashCommand,
    TuiConfig,
    TuiControls,
    guideline_provider_from_library,
    persona_provider_from_library,
)

# ── Plug-in protocol conformance ───────────────────────────────────


class _Runner:
    async def run(self, query: str, controls: TuiControls) -> RunResult:
        return RunResult(text=f"echo:{query}")


class _SaveHandler:
    def suggest_filename(self, query: str | None, response: str) -> str:
        return "out.md"

    def save(self, content: str, filename: str) -> Path:
        return Path(filename)


class _Persona:
    def __init__(self):
        self._cur = None

    def list_ids(self):
        return ["alice", "bob"]

    def current_id(self):
        return self._cur

    def select(self, persona_id: str) -> None:
        self._cur = persona_id


class _Guideline:
    def __init__(self):
        self._cur = None

    def list_ids(self):
        return ["concise"]

    def current_id(self):
        return self._cur

    def select(self, guideline_id: str) -> None:
        self._cur = guideline_id

    def clear(self) -> None:
        self._cur = None


def test_runner_protocol_conformance():
    assert isinstance(_Runner(), AgentRunner)


def test_save_handler_protocol_conformance():
    assert isinstance(_SaveHandler(), SaveHandler)
    assert isinstance(FileSaveHandler(), SaveHandler)


def test_persona_provider_protocol_conformance():
    assert isinstance(_Persona(), PersonaProvider)


def test_guideline_provider_protocol_conformance():
    assert isinstance(_Guideline(), GuidelineProvider)


# ── FileSaveHandler ────────────────────────────────────────────────


def test_file_save_handler_writes_markdown(tmp_path):
    handler = FileSaveHandler(tmp_path)
    path = handler.save("hello", "note")
    assert path.suffix == ".md"
    assert path.read_text() == "hello"


def test_file_save_handler_keeps_existing_md_suffix(tmp_path):
    handler = FileSaveHandler(tmp_path)
    path = handler.save("hi", "already.md")
    assert path.name == "already.md"


def test_file_save_handler_suggest_uses_query_slug(tmp_path):
    handler = FileSaveHandler(tmp_path)
    name = handler.suggest_filename("Quick brown fox", "response")
    assert name.endswith(".md")
    assert "quick" in name


# ── BUILTIN_COMMANDS / SlashCommand ────────────────────────────────


def test_builtin_commands_include_core_commands():
    names = {c.name for c in BUILTIN_COMMANDS}
    for required in ("/help", "/cancel", "/save", "/quit", "/reset"):
        assert required in names


def test_slash_command_is_immutable():
    from dataclasses import FrozenInstanceError

    cmd = SlashCommand("/foo", "bar")
    with pytest.raises(FrozenInstanceError):
        cmd.name = "/baz"


# ── Mode enum ──────────────────────────────────────────────────────


def test_mode_enum_has_expected_states():
    assert {m.name for m in Mode} == {
        "IDLE",
        "AWAITING_FILENAME",
        "AWAITING_PERSONA",
        "AWAITING_GUIDELINE",
    }


# ── TuiConfig ──────────────────────────────────────────────────────


def test_tui_config_defaults_save_handler_and_optional_providers():
    config = TuiConfig(
        agent_name="Agent",
        agent_description="Test",
        runner=_Runner(),
    )
    assert isinstance(config.save_handler, FileSaveHandler)
    assert config.persona_provider is None
    assert config.guideline_provider is None
    assert config.custom_widget is None


def test_tui_config_accepts_full_plugin_set():
    config = TuiConfig(
        agent_name="Agent",
        agent_description="x",
        runner=_Runner(),
        save_handler=_SaveHandler(),
        persona_provider=_Persona(),
        guideline_provider=_Guideline(),
        initial_stats={"count": 0},
    )
    assert isinstance(config.save_handler, SaveHandler)
    assert isinstance(config.persona_provider, PersonaProvider)
    assert isinstance(config.guideline_provider, GuidelineProvider)
    assert config.initial_stats == {"count": 0}


# ── Library adapter helpers ────────────────────────────────────────


class _FakeLibrary:
    def list_ids(self):
        return ["a", "b"]


class _FakeController:
    def __init__(self):
        self.current_persona_id = "a"
        self.current_guideline_id = None
        self.calls: list[tuple[str, str | None]] = []

    def set_persona_by_id(self, persona_id: str) -> None:
        self.current_persona_id = persona_id
        self.calls.append(("persona", persona_id))

    def set_guideline_by_id(self, guideline_id: str) -> None:
        self.current_guideline_id = guideline_id
        self.calls.append(("guideline", guideline_id))

    def clear_guideline(self) -> None:
        self.current_guideline_id = None
        self.calls.append(("clear_guideline", None))


def test_persona_provider_from_library_delegates_to_controller():
    library = _FakeLibrary()
    controller = _FakeController()
    provider = persona_provider_from_library(library, controller)
    assert provider.list_ids() == ["a", "b"]
    assert provider.current_id() == "a"
    provider.select("b")
    assert controller.calls == [("persona", "b")]
    assert provider.current_id() == "b"


def test_guideline_provider_from_library_delegates_to_controller():
    library = _FakeLibrary()
    controller = _FakeController()
    provider = guideline_provider_from_library(library, controller)
    assert provider.list_ids() == ["a", "b"]
    assert provider.current_id() is None
    provider.select("a")
    assert controller.calls == [("guideline", "a")]
    provider.clear()
    assert controller.calls[-1] == ("clear_guideline", None)


# ── AgentTUI constructs without textual mounting ────────────────────


def test_agent_tui_constructs_with_runner_only():
    tui = AgentTUI(
        TuiConfig(
            agent_name="Demo",
            agent_description="x",
            runner=_Runner(),
        )
    )
    assert tui._mode is Mode.IDLE  # type: ignore[attr-defined]
