"""Human approval gates. The AI may propose; a human must authorize active work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from app.security_testing.errors import ApprovalRequiredError, RestrictedActivityError


class ApprovalKind(StrEnum):
    ENABLE_ACTIVE_TESTING = "enable_active_testing"
    START_LIVE_SCAN = "start_live_scan"
    ENABLE_FUZZING = "enable_fuzzing"
    HIGH_RISK_SCANNER = "high_risk_scanner"
    SEND_POC_REQUEST = "send_poc_request"
    SUBMIT_HACKERONE_REPORT = "submit_hackerone_report"


class ApprovalState(StrEnum):
    MISSING = "missing"
    GRANTED = "granted"
    DENIED = "denied"


@dataclass(frozen=True)
class ApprovalRecord:
    kind: ApprovalKind
    state: ApprovalState
    operator: str
    note: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    granted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    def expired(self, *, now: datetime | None = None) -> bool:
        moment = now or datetime.now(UTC)
        if self.expires_at is None:
            return False
        expires = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=UTC)
        current = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
        return current >= expires


_AI_OPERATORS = frozenset({"ai", "model", "llm", "assistant", "bugforge-ai", "qwen", "system"})


def is_ai_operator(operator: str) -> bool:
    return operator.strip().lower() in _AI_OPERATORS


class HumanApprovalGate:
    """In-memory approval ledger. AI operators cannot grant HackerOne submission."""

    def __init__(self) -> None:
        self._records: dict[ApprovalKind, ApprovalRecord] = {}

    def grant(
        self,
        kind: ApprovalKind,
        *,
        operator: str,
        note: str = "",
        expires_at: datetime | None = None,
    ) -> ApprovalRecord:
        if is_ai_operator(operator):
            raise RestrictedActivityError("The AI cannot grant approvals")
        record = ApprovalRecord(
            kind=kind,
            state=ApprovalState.GRANTED,
            operator=operator,
            note=note,
            expires_at=expires_at,
        )
        self._records[kind] = record
        return record

    def deny(self, kind: ApprovalKind, *, operator: str, note: str = "") -> ApprovalRecord:
        record = ApprovalRecord(kind=kind, state=ApprovalState.DENIED, operator=operator, note=note)
        self._records[kind] = record
        return record

    def state_of(self, kind: ApprovalKind) -> ApprovalState:
        record = self._records.get(kind)
        if record is None:
            return ApprovalState.MISSING
        if record.expired():
            return ApprovalState.MISSING
        return record.state

    def require(self, kind: ApprovalKind) -> ApprovalRecord:
        record = self._records.get(kind)
        if record is None or record.state is not ApprovalState.GRANTED or record.expired():
            raise ApprovalRequiredError(kind.value)
        return record

    def is_granted(self, kind: ApprovalKind) -> bool:
        record = self._records.get(kind)
        if record is None or record.state is not ApprovalState.GRANTED:
            return False
        if record.expired():
            return False
        return True

    def get_record(self, kind: ApprovalKind) -> ApprovalRecord | None:
        return self._records.get(kind)

    def iter_records(self) -> dict[ApprovalKind, ApprovalRecord]:
        return dict(self._records)

    def load_record(self, record: ApprovalRecord) -> None:
        self._records[record.kind] = record

    def clear_records(self) -> None:
        self._records.clear()

    def snapshot(self) -> dict[str, str]:
        return {kind.value: self.state_of(kind).value for kind in ApprovalKind}

    def detailed_snapshot(self) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = {}
        for kind in ApprovalKind:
            record = self._records.get(kind)
            if record is None:
                out[kind.value] = {"state": ApprovalState.MISSING.value}
                continue
            out[kind.value] = {
                "state": record.state.value
                if not record.expired()
                else ApprovalState.MISSING.value,
                "operator": record.operator,
                "granted_at": record.granted_at.isoformat(),
                "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            }
        return out
