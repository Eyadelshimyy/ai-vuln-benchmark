"""Shared provenance helpers for the JSONL observation hooks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Payload keys that identify which agent/tool call produced the event;
# different SDK versions expose different ones.
ACTOR_KEYS = {
    "agent",
    "agent_name",
    "actor",
    "subagent_type",
    "task_agent",
    "action_id",
    "tool_call_id",
    "parent_id",
    "call_id",
}


def actor(payload: dict[str, Any]) -> dict[str, Any] | None:
    found = {key: payload[key] for key in sorted(payload) if key in ACTOR_KEYS}
    return found or None


def project_dir(payload: dict[str, Any]) -> Path:
    return Path(
        os.environ.get("OPENHANDS_PROJECT_DIR") or payload.get("working_dir") or "."
    ).resolve()


def events_path(payload: dict[str, Any], env_name: str, relative_path: Path) -> Path:
    override = os.environ.get(env_name)
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else project_dir(payload) / path
    return project_dir(payload) / relative_path


def event_id(record: dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def next_sequence(path: Path) -> int:
    sequence = 1
    if path.exists():
        sequence += len(path.read_text(encoding="utf-8").splitlines())
    return sequence
