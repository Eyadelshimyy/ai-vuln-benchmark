"""Lightweight, opinionated configuration support for ellements.

Goals:

* Apps describe their settings as **frozen dataclasses** — strongly
  typed, immutable, easy to introspect, trivially picklable, plays
  well with Pydantic or attrs at the boundary.
* Loading from disk is **one function call** per supported format
  (TOML, JSON). Both formats accept identical key→value structures, so
  callers can pick whichever suits their tooling.
* No global "settings singleton". Apps build their config explicitly
  and pass it to whoever needs it — this keeps tests trivial and
  removes hidden coupling.
* Loaders never inject defaults silently. If the dataclass declares a
  default, that's what callers get when the file omits the field; if
  no default is declared, the loader raises a clear error.
* Composition is by **plain object construction** (``dataclasses.replace``
  for overlays) rather than a bespoke merge DSL. We provide
  :func:`overlay` as a tiny convenience for the common "base config +
  per-environment overrides" pattern.

Example::

    from dataclasses import dataclass
    from ellements.core.config import load_toml, overlay

    @dataclass(frozen=True, slots=True)
    class AppConfig:
        model: str
        temperature: float = 0.7
        retries: int = 3

    base = load_toml("config/base.toml", AppConfig)
    prod = overlay(base, {"temperature": 0.0})
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from .exceptions import EllementsError

ConfigT = TypeVar("ConfigT")


class ConfigError(EllementsError):
    """Raised when a config payload cannot be turned into a dataclass."""


def load_toml(path: Path | str, config_class: type[ConfigT]) -> ConfigT:
    """Load *path* as TOML and instantiate *config_class*.

    Args:
        path: Path to a TOML file. Must be readable.
        config_class: A dataclass type. Its field set defines the
            expected schema. Extra keys in the file raise
            :class:`ConfigError` so typos surface loudly.

    Returns:
        A fully populated instance of *config_class*.

    Raises:
        ConfigError: If the file is unreadable, malformed TOML,
            references unknown fields, or omits required fields.
    """
    file_path = Path(path)
    try:
        with file_path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {file_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {file_path}: {exc}") from exc
    return _instantiate(config_class, data, source=str(file_path))


def load_json(path: Path | str, config_class: type[ConfigT]) -> ConfigT:
    """Load *path* as JSON and instantiate *config_class*.

    Args:
        path: Path to a JSON file containing a single object literal.
        config_class: A dataclass type. Its field set defines the
            expected schema. Extra keys raise :class:`ConfigError`.

    Returns:
        A fully populated instance of *config_class*.

    Raises:
        ConfigError: If the file is unreadable, malformed JSON, the
            top-level value is not an object, or required fields are
            missing.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {file_path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {file_path} must contain a JSON object at the "
            f"top level (got {type(data).__name__})."
        )
    return _instantiate(config_class, data, source=str(file_path))


def overlay(base: ConfigT, changes: dict[str, Any]) -> ConfigT:
    """Return a copy of *base* with *changes* applied.

    Thin wrapper over :func:`dataclasses.replace` that validates the
    keys in *changes* against the dataclass schema, so callers get a
    crisp error if they typo an override.

    Args:
        base: An instance of a frozen dataclass.
        changes: A mapping of field name → new value. Every key must
            be an existing field on the dataclass.

    Returns:
        A new instance with the same type as *base*, sharing all
        unchanged fields and using overridden values for everything
        in *changes*.

    Raises:
        ConfigError: If *base* is not a dataclass instance, or if
            *changes* contains an unknown field name.
    """
    if not is_dataclass(base) or isinstance(base, type):
        raise ConfigError(
            f"overlay() expects a dataclass instance, got {type(base).__name__}."
        )
    known = {field.name for field in fields(base)}
    unknown = set(changes) - known
    if unknown:
        raise ConfigError(
            f"Unknown config fields for {type(base).__name__}: "
            f"{sorted(unknown)}. Known fields: {sorted(known)}."
        )
    return replace(base, **changes)


def _instantiate(
    config_class: type[ConfigT], data: dict[str, Any], *, source: str
) -> ConfigT:
    """Construct *config_class* from *data*, raising on schema mismatches."""
    if not is_dataclass(config_class):
        raise ConfigError(
            f"{config_class.__name__} is not a dataclass; config_class must "
            f"be a (preferably frozen) dataclass."
        )
    known = {field.name for field in fields(config_class)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"Unknown config keys in {source} for "
            f"{config_class.__name__}: {sorted(unknown)}. "
            f"Known fields: {sorted(known)}."
        )
    try:
        return config_class(**data)
    except TypeError as exc:
        raise ConfigError(
            f"Failed to instantiate {config_class.__name__} from {source}: "
            f"{exc}"
        ) from exc


__all__ = [
    "ConfigError",
    "load_json",
    "load_toml",
    "overlay",
]
