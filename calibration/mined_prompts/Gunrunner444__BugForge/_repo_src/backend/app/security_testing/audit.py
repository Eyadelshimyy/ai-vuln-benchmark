"""Append-only, hash-chained audit log for active testing actions.

Persistence is behind :class:`AuditLogStore`. The default store is in-memory.
Secrets must never be written to events.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from app.security_testing.secrets import redact_text


def _canonical(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


@dataclass(frozen=True)
class AuditEvent:
    timestamp: datetime
    project: str
    target: str
    scope_decision: str
    tool: str
    action: str
    request_id: str
    rate_limit_decision: str
    result: str
    evidence_id: str
    human_approval: str
    prev_hash: str
    entry_hash: str
    id: str = field(default_factory=lambda: uuid4().hex)

    def to_mapping(self) -> dict[str, object]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class AuditLogStore(Protocol):
    """Persistence interface. Implementations must not store secrets."""

    def append(self, event: AuditEvent) -> None: ...

    def load(self) -> Sequence[AuditEvent]: ...


class InMemoryAuditStore:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self._events.append(event)

    def load(self) -> Sequence[AuditEvent]:
        return tuple(self._events)


class AuditLog:
    """Tamper-evident within the process: each entry hashes the previous hash."""

    def __init__(self, store: AuditLogStore | None = None) -> None:
        self._store: AuditLogStore = store or InMemoryAuditStore()
        loaded = list(self._store.load())
        self._entries: list[AuditEvent] = loaded
        self._head = loaded[-1].entry_hash if loaded else "0" * 64

    def record(
        self,
        *,
        project: str,
        target: str,
        scope_decision: str,
        tool: str,
        action: str,
        request_id: str = "",
        rate_limit_decision: str = "",
        result: str = "",
        evidence_id: str = "",
        human_approval: str = "",
    ) -> AuditEvent:
        timestamp = datetime.now(UTC)
        body = {
            "timestamp": timestamp.isoformat(),
            "project": project,
            "target": redact_text(target),
            "scope_decision": redact_text(scope_decision),
            "tool": tool,
            "action": redact_text(action),
            "request_id": request_id,
            "rate_limit_decision": rate_limit_decision,
            "result": redact_text(result),
            "evidence_id": evidence_id,
            "human_approval": human_approval,
            "prev_hash": self._head,
        }
        digest = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
        event = AuditEvent(
            timestamp=timestamp,
            project=project,
            target=redact_text(target),
            scope_decision=redact_text(scope_decision),
            tool=tool,
            action=redact_text(action),
            request_id=request_id,
            rate_limit_decision=rate_limit_decision,
            result=redact_text(result),
            evidence_id=evidence_id,
            human_approval=human_approval,
            prev_hash=self._head,
            entry_hash=digest,
        )
        self._entries.append(event)
        self._head = digest
        self._store.append(event)
        return event

    def entries(self) -> tuple[AuditEvent, ...]:
        return tuple(self._entries)

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for event in self._entries:
            body = {
                "timestamp": event.timestamp.isoformat(),
                "project": event.project,
                "target": event.target,
                "scope_decision": event.scope_decision,
                "tool": event.tool,
                "action": event.action,
                "request_id": event.request_id,
                "rate_limit_decision": event.rate_limit_decision,
                "result": event.result,
                "evidence_id": event.evidence_id,
                "human_approval": event.human_approval,
                "prev_hash": prev,
            }
            expected = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
            if event.prev_hash != prev or event.entry_hash != expected:
                return False
            prev = event.entry_hash
        return True

    def __len__(self) -> int:
        return len(self._entries)
