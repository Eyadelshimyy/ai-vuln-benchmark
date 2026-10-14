"""Project-scoped research memory. Credentials are never stored.

Entries are recursively sanitized. Restoring a persisted record uses the
original identity; ``remember()`` is not used to re-mint IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.security_agent.secrets import is_secret_key
from app.security_testing.secrets import redact_text

_FORBIDDEN = (
    "password",
    "passwd",
    "authorization",
    "cookie",
    "set-cookie",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "secret",
    "session",
    "credential",
)


@dataclass
class MemoryEntry:
    kind: str
    summary: str
    extra: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "extra": dict(self.extra),
        }


@dataclass
class ResearchMemory:
    project_id: str
    entries: list[MemoryEntry] = field(default_factory=list)

    def remember(self, kind: str, summary: str, extra: dict[str, Any] | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            kind=kind,
            summary=redact_text(summary),
            extra=_sanitize(extra or {}),
        )
        self.entries.append(entry)
        return entry

    def restore_entry(
        self,
        *,
        entry_id: str,
        kind: str,
        summary: str,
        extra: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Rehydrate a persisted record without minting a new identity."""
        entry = MemoryEntry(
            id=entry_id,
            kind=kind,
            summary=redact_text(summary),
            extra=_sanitize(extra or {}),
        )
        self.entries.append(entry)
        return entry

    def matching(self, kind: str) -> list[MemoryEntry]:
        return [item for item in self.entries if item.kind == kind]


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if is_secret_key(str(key)) or str(key).lower() in _FORBIDDEN:
                cleaned[str(key)] = "[REDACTED]"
                continue
            cleaned[str(key)] = _sanitize(item)
        return cleaned
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(str(value))
