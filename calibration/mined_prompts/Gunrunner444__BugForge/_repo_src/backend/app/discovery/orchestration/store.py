"""Versioned persistence and strict loading of campaign state.

A stored document is data, never instructions. It is parsed against the typed
model, its hash is recomputed, and anything that does not match is refused.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.discovery.orchestration.codec import CodecError, canonical_json, from_jsonable
from app.discovery.orchestration.model import (
    SCHEMA_VERSION,
    DecisionRecord,
    PersistenceError,
    ResearchState,
    ResumeError,
    ResumeStatus,
)

MAX_DOCUMENT_BYTES = 2_000_000


def dump_state(state: ResearchState) -> str:
    """Canonical text with the state hash sealed in."""
    state.seal()
    text = canonical_json(state)
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise PersistenceError("campaign state is larger than the stored document limit")
    return text


def load_state(text: str) -> ResearchState:
    """Parse and validate. Every failure is a typed ResumeError, never a guess."""
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ResumeError(ResumeStatus.CORRUPT, "document is too large")
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ResumeError(ResumeStatus.CORRUPT, "document is not JSON") from exc
    if not isinstance(data, dict):
        raise ResumeError(ResumeStatus.CORRUPT, "document is not an object")
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ResumeError(ResumeStatus.CORRUPT, "schema version is missing")
    if version > SCHEMA_VERSION:
        raise ResumeError(ResumeStatus.UNSUPPORTED_VERSION, f"schema {version}")
    if version < SCHEMA_VERSION:
        raise ResumeError(ResumeStatus.MIGRATION_REQUIRED, f"schema {version}")
    try:
        state = from_jsonable(ResearchState, data)
    except (CodecError, TypeError, KeyError, RecursionError) as exc:
        raise ResumeError(ResumeStatus.CORRUPT, str(exc)[:160]) from exc
    assert isinstance(state, ResearchState)
    _validate(state)
    return state


def _validate(state: ResearchState) -> None:
    if state.state_hash != state.hash_now():
        raise ResumeError(ResumeStatus.CORRUPT, "state hash does not match")
    if any(item.proves_safety for item in state.negatives):
        raise ResumeError(ResumeStatus.CORRUPT, "negative evidence claims safety")
    ledger = state.budget
    for name, value in (*ledger.limits.items(), *ledger.consumed.items(), *ledger.reserved.items()):
        if value < 0:
            raise ResumeError(ResumeStatus.CORRUPT, f"negative budget value for {name}")
    for number, record in enumerate(state.decisions, start=1):
        if record.sequence != number or record.record_hash != record.sealed().record_hash:
            raise ResumeError(ResumeStatus.CORRUPT, "decision history is not intact")
    if state.decision_number != len(state.decisions):
        raise ResumeError(ResumeStatus.CORRUPT, "decision count differs")
    if state.round < 0 or state.resume_count < 0:
        raise ResumeError(ResumeStatus.CORRUPT, "negative counter")


class ResearchStore(Protocol):
    def save(self, state: ResearchState) -> None: ...

    def load(self, campaign_id: str) -> ResearchState | None: ...


class MemoryStore:
    """Process-local store used by tests and for ephemeral campaigns."""

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}
        self.history: dict[str, dict[int, str]] = {}
        self.saves = 0

    def save(self, state: ResearchState) -> None:
        text = dump_state(state)
        self._check_history(state)
        self.documents[state.identity.campaign_id] = text
        self.saves += 1

    def load(self, campaign_id: str) -> ResearchState | None:
        text = self.documents.get(campaign_id)
        return None if text is None else load_state(text)

    def _check_history(self, state: ResearchState) -> None:
        known = self.history.setdefault(state.identity.campaign_id, {})
        for record in state.decisions:
            previous = known.get(record.sequence)
            if previous is not None and previous != record.record_hash:
                raise PersistenceError("decision history is immutable")
            known[record.sequence] = record.record_hash


class SqlStore:
    """Synchronous SQLAlchemy store with an optimistic revision check."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._revisions: dict[str, int] = {}

    def save(self, state: ResearchState) -> None:
        from app.models.discovery_orchestration import (
            DBOrchestrationCampaign,
            DBOrchestrationDecision,
        )

        text = dump_state(state)
        campaign_id = state.identity.campaign_id
        now = datetime.now(UTC)
        try:
            row = self.session.get(DBOrchestrationCampaign, campaign_id)
            if row is None:
                if campaign_id in self._revisions:
                    raise PersistenceError("campaign row disappeared")
                row = DBOrchestrationCampaign(campaign_id=campaign_id, revision=0, created_at=now)
                self.session.add(row)
            elif campaign_id not in self._revisions:
                raise PersistenceError("campaign exists and was not loaded")
            elif self._revisions[campaign_id] != row.revision:
                raise PersistenceError("campaign changed since it was loaded")
            row.project = state.identity.project[:256]
            row.target_digest = state.identity.target_digest()
            row.source_snapshot = state.identity.source_snapshot[:128]
            row.compiler_configuration = state.identity.compiler_configuration[:128]
            row.schema_version = state.schema_version
            row.orchestrator_version = state.orchestrator_version
            row.state = state.state.value
            row.stop_reason = state.stop_reason
            row.round = state.round
            row.revision = row.revision + 1
            row.state_hash = state.state_hash
            row.document = text
            row.updated_at = now
            existing = {
                item.sequence: item.record_hash
                for item in self.session.scalars(
                    select(DBOrchestrationDecision).where(
                        DBOrchestrationDecision.campaign_id == campaign_id
                    )
                )
            }
            for record in state.decisions:
                known = existing.get(record.sequence)
                if known is not None:
                    if known != record.record_hash:
                        raise PersistenceError("decision history is immutable")
                    continue
                self.session.add(
                    DBOrchestrationDecision(
                        decision_id=record.decision_id,
                        campaign_id=campaign_id,
                        sequence=record.sequence,
                        round=record.round,
                        record_hash=record.record_hash,
                        document=canonical_json(record),
                        created_at=now,
                    )
                )
            self.session.commit()
            self._revisions[campaign_id] = row.revision
        except PersistenceError:
            self.session.rollback()
            raise
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise PersistenceError("campaign could not be saved") from exc

    def load(self, campaign_id: str) -> ResearchState | None:
        from app.models.discovery_orchestration import DBOrchestrationCampaign

        try:
            row = self.session.get(DBOrchestrationCampaign, campaign_id)
        except SQLAlchemyError as exc:
            raise PersistenceError("campaign could not be read") from exc
        if row is None:
            return None
        state = load_state(row.document)
        if state.identity.campaign_id != campaign_id:
            raise ResumeError(ResumeStatus.CORRUPT, "document names another campaign")
        self._revisions[campaign_id] = row.revision
        return state

    def decisions(self, campaign_id: str) -> tuple[DecisionRecord, ...]:
        from app.models.discovery_orchestration import DBOrchestrationDecision

        rows = self.session.scalars(
            select(DBOrchestrationDecision)
            .where(DBOrchestrationDecision.campaign_id == campaign_id)
            .order_by(DBOrchestrationDecision.sequence)
        )
        records: list[DecisionRecord] = []
        for row in rows:
            try:
                parsed = from_jsonable(DecisionRecord, json.loads(row.document))
            except (CodecError, ValueError) as exc:
                raise ResumeError(ResumeStatus.CORRUPT, "decision row is invalid") from exc
            assert isinstance(parsed, DecisionRecord)
            records.append(parsed)
        return tuple(records)
