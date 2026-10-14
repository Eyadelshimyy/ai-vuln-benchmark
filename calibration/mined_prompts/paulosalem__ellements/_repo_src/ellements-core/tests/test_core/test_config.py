"""Tests for the lightweight config-loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from ellements.core.config import ConfigError, load_json, load_toml, overlay


@dataclass(frozen=True, slots=True)
class _AppConfig:
    model: str
    temperature: float = 0.7
    retries: int = 3


@dataclass(frozen=True, slots=True)
class _NestedConfig:
    name: str
    nested: dict[str, int] | None = None


# ── load_toml ─────────────────────────────────────────────────────────


def test_load_toml_with_all_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        'model = "openai/gpt-4o"\ntemperature = 0.2\nretries = 5\n'
    )
    config = load_toml(config_file, _AppConfig)
    assert config == _AppConfig(model="openai/gpt-4o", temperature=0.2, retries=5)


def test_load_toml_uses_dataclass_defaults_for_omitted_fields(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "minimal.toml"
    config_file.write_text('model = "anthropic/claude-3-5-sonnet"\n')
    config = load_toml(config_file, _AppConfig)
    assert config.model == "anthropic/claude-3-5-sonnet"
    assert config.temperature == 0.7  # default preserved
    assert config.retries == 3


def test_load_toml_missing_required_field_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "incomplete.toml"
    config_file.write_text("temperature = 0.5\n")  # missing required ``model``
    with pytest.raises(ConfigError, match="Failed to instantiate"):
        load_toml(config_file, _AppConfig)


def test_load_toml_unknown_keys_raise(tmp_path: Path) -> None:
    config_file = tmp_path / "extra.toml"
    config_file.write_text('model = "x"\nunknown_field = 42\n')
    with pytest.raises(ConfigError, match="Unknown config keys"):
        load_toml(config_file, _AppConfig)


def test_load_toml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_toml(tmp_path / "does_not_exist.toml", _AppConfig)


def test_load_toml_malformed_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.toml"
    config_file.write_text("not valid =:= toml")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_toml(config_file, _AppConfig)


# ── load_json ─────────────────────────────────────────────────────────


def test_load_json_with_all_fields(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"model": "openai/gpt-4o", "temperature": 0.2, "retries": 5}'
    )
    config = load_json(config_file, _AppConfig)
    assert config == _AppConfig(model="openai/gpt-4o", temperature=0.2, retries=5)


def test_load_json_requires_object_at_top_level(tmp_path: Path) -> None:
    config_file = tmp_path / "list.json"
    config_file.write_text('["not", "an", "object"]')
    with pytest.raises(ConfigError, match="must contain a JSON object"):
        load_json(config_file, _AppConfig)


def test_load_json_unknown_keys_raise(tmp_path: Path) -> None:
    config_file = tmp_path / "extra.json"
    config_file.write_text('{"model": "x", "rogue": true}')
    with pytest.raises(ConfigError, match="Unknown config keys"):
        load_json(config_file, _AppConfig)


def test_load_json_malformed_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.json"
    config_file.write_text("{not valid json")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        load_json(config_file, _AppConfig)


# ── overlay ───────────────────────────────────────────────────────────


def test_overlay_returns_new_instance_with_changes() -> None:
    base = _AppConfig(model="m1", temperature=0.5, retries=2)
    new = overlay(base, {"temperature": 0.9})
    assert new == _AppConfig(model="m1", temperature=0.9, retries=2)
    assert base.temperature == 0.5  # original untouched (frozen)


def test_overlay_rejects_unknown_field() -> None:
    base = _AppConfig(model="m1")
    with pytest.raises(ConfigError, match="Unknown config fields"):
        overlay(base, {"nope": 1})


def test_overlay_rejects_non_dataclass() -> None:
    with pytest.raises(ConfigError, match="dataclass instance"):
        overlay({"model": "m"}, {"temperature": 0.1})  # type: ignore[arg-type]


def test_overlay_with_empty_changes_returns_equal_copy() -> None:
    base = _AppConfig(model="m")
    new = overlay(base, {})
    assert new == base
    assert new is not base


# ── instantiation against a non-dataclass ──────────────────────────────


def test_load_toml_rejects_non_dataclass(tmp_path: Path) -> None:
    class _NotADataclass:
        pass

    config_file = tmp_path / "anything.toml"
    config_file.write_text('x = "y"\n')
    with pytest.raises(ConfigError, match="not a dataclass"):
        load_toml(config_file, _NotADataclass)  # type: ignore[type-var]
