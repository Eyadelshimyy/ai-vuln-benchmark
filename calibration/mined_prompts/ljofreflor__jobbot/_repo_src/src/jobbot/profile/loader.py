"""Load Candidate from profile.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from jobbot.models.candidate import Candidate


class ProfileLoadError(Exception):
    """Raised when the profile file cannot be read or parsed."""


def load_profile(path: Path) -> Candidate:
    """Load and parse a profile YAML file into a Candidate."""
    if not path.is_file():
        msg = f"Profile not found: {path}"
        raise ProfileLoadError(msg)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in {path}: {exc}"
        raise ProfileLoadError(msg) from exc

    if raw is None:
        msg = f"Profile is empty: {path}"
        raise ProfileLoadError(msg)
    if not isinstance(raw, dict):
        msg = f"Profile root must be a mapping: {path}"
        raise ProfileLoadError(msg)

    try:
        return Candidate.model_validate(raw)
    except ValidationError as exc:
        msg = _format_validation_error(exc)
        raise ProfileLoadError(msg) from exc


def load_profile_raw(path: Path) -> dict[str, Any]:
    """Load raw YAML mapping without Pydantic validation."""
    if not path.is_file():
        msg = f"Profile not found: {path}"
        raise ProfileLoadError(msg)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Profile root must be a mapping: {path}"
        raise ProfileLoadError(msg)
    return raw


def _format_validation_error(exc: ValidationError) -> str:
    lines: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        lines.append(f"{loc}: {err['msg']}")
    return "\n".join(lines)
