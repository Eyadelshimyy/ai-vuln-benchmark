"""Security Research Project lifecycle. Distinct from HackerOne remote states."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.security_agent.states import ResearchProjectState
from app.security_testing.errors import RestrictedActivityError

_TRANSITIONS: dict[ResearchProjectState, frozenset[ResearchProjectState]] = {
    ResearchProjectState.CREATE: frozenset({ResearchProjectState.CONFIGURE}),
    ResearchProjectState.CONFIGURE: frozenset(
        {ResearchProjectState.SCOPE_SYNCED, ResearchProjectState.READY}
    ),
    ResearchProjectState.SCOPE_SYNCED: frozenset({ResearchProjectState.READY}),
    ResearchProjectState.READY: frozenset(
        {ResearchProjectState.RESEARCHING, ResearchProjectState.PAUSED}
    ),
    ResearchProjectState.RESEARCHING: frozenset(
        {
            ResearchProjectState.PAUSED,
            ResearchProjectState.FINDINGS,
            ResearchProjectState.REVIEW,
            ResearchProjectState.COMPLETE,
        }
    ),
    ResearchProjectState.PAUSED: frozenset(
        {ResearchProjectState.RESEARCHING, ResearchProjectState.COMPLETE}
    ),
    ResearchProjectState.FINDINGS: frozenset(
        {ResearchProjectState.REVIEW, ResearchProjectState.RESEARCHING}
    ),
    ResearchProjectState.REVIEW: frozenset(
        {ResearchProjectState.HANDOFF, ResearchProjectState.COMPLETE, ResearchProjectState.FINDINGS}
    ),
    ResearchProjectState.HANDOFF: frozenset(
        {ResearchProjectState.COMPLETE, ResearchProjectState.REVIEW}
    ),
    ResearchProjectState.COMPLETE: frozenset(),
}


@dataclass
class SecurityResearchProject:
    name: str
    project_id: str
    target: str
    mode: str = "lab"
    program_handle: str = ""
    session_id: str = ""
    strategy: str = "passive_recon"
    operator_identity: str = ""
    state: ResearchProjectState = ResearchProjectState.CREATE
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def transition(self, destination: ResearchProjectState) -> None:
        allowed = _TRANSITIONS.get(self.state, frozenset())
        if destination not in allowed:
            raise RestrictedActivityError(
                f"invalid_project_transition:{self.state.value}->{destination.value}"
            )
        self.state = destination
        self.updated_at = datetime.now(UTC)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "target": self.target,
            "mode": self.mode,
            "program_handle": self.program_handle,
            "strategy": self.strategy,
            "state": self.state.value,
            "operator_identity": self.operator_identity,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: Any) -> SecurityResearchProject:
        try:
            state = ResearchProjectState(str(row.state or "create"))
        except ValueError:
            state = ResearchProjectState.CREATE
        return cls(
            id=str(row.id),
            name=row.name or "",
            project_id=row.project_id or "",
            target=row.target or "",
            mode=row.mode or "lab",
            program_handle=row.program_handle or "",
            session_id=row.session_id or "",
            strategy=row.strategy or "passive_recon",
            operator_identity=row.operator_identity or "",
            state=state,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
