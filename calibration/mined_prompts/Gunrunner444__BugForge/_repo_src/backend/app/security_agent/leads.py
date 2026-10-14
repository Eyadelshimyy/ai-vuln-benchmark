"""Research leads stored in BugForge's existing research persistence.

A lead is a work item. It is not a finding and it is never verified by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.security_agent import DBResearchLead
from app.security_agent.prioritize import score_lead

LEAD_STATUSES = frozenset(
    {"NEW", "ACTIVE", "BLOCKED", "KILLED", "PROMOTED", "REPRODUCED", "VERIFIED", "REPORTED"}
)


@dataclass
class ResearchLead:
    id: str
    project_id: str
    session_id: str
    target: str
    hypothesis: str
    status: str = "NEW"
    priority: str = "medium"
    next_action: str = ""
    kill_reason: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    observation_ids: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    chain_ids: list[str] = field(default_factory=list)
    hypothesis_ids: list[str] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    semantic_node_ids: list[str] = field(default_factory=list)
    updated_at: str = ""

    def snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "target": self.target,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "priority": self.priority,
            "next_action": self.next_action,
            "kill_reason": self.kill_reason,
            "evidence_ids": list(self.evidence_ids),
            "observation_ids": list(self.observation_ids),
            "related_ids": list(self.related_ids),
            "chain_ids": list(self.chain_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "finding_ids": list(self.finding_ids),
            "semantic_node_ids": list(self.semantic_node_ids),
            "updated_at": self.updated_at,
        }

    def __post_init__(self) -> None:
        if self.status not in LEAD_STATUSES:
            raise ValueError(f"unknown lead status {self.status}")
        if not self.updated_at:
            self.updated_at = datetime.now(UTC).isoformat()


class LeadStore:
    def __init__(self) -> None:
        self._items: dict[str, ResearchLead] = {}

    def upsert(self, lead: ResearchLead) -> ResearchLead:
        _validate(lead)
        lead.updated_at = datetime.now(UTC).isoformat()
        self._items[lead.id] = lead
        return lead

    def get(self, lead_id: str) -> ResearchLead | None:
        return self._items.get(lead_id)

    def list_project(self, project_id: str) -> list[ResearchLead]:
        return [item for item in self._items.values() if item.project_id == project_id]

    def stale(self, project_id: str, *, before: str) -> list[ResearchLead]:
        return [
            item
            for item in self.list_project(project_id)
            if item.status in {"NEW", "ACTIVE", "BLOCKED"} and item.updated_at < before
        ]


def new_lead(
    *,
    project_id: str,
    session_id: str,
    target: str,
    hypothesis: str,
    priority: str = "medium",
) -> ResearchLead:
    return ResearchLead(
        id=uuid4().hex,
        project_id=project_id,
        session_id=session_id,
        target=target,
        hypothesis=hypothesis,
        priority=priority,
    )


def backfill_lead_relationships(
    *,
    related_ids: list[str],
    evidence_ids: list[str],
    observation_ids: list[str],
    chain_ids: list[str],
    hypothesis_ids: list[str] | None,
    finding_ids: list[str] | None,
    semantic_node_ids: list[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    """Move legacy hypothesis ids out of related_ids without inventing new links.

    Phase 36 stored the hypothesis id in related_ids. Ids that already belong
    to evidence, observations, chains, findings, or semantic nodes stay there.
    """
    hypotheses = list(hypothesis_ids or [])
    findings = list(finding_ids or [])
    semantic = list(semantic_node_ids or [])
    occupied = (
        set(evidence_ids) | set(observation_ids) | set(chain_ids) | set(findings) | set(semantic)
    )
    for item in related_ids:
        if not item or item in occupied or item in hypotheses:
            continue
        hypotheses.append(item)
        occupied.add(item)
    return hypotheses, findings, semantic


def _validate(lead: ResearchLead) -> None:
    if lead.status == "KILLED" and not lead.kill_reason.strip():
        raise ValueError("a killed lead requires a kill reason")
    if lead.status == "VERIFIED" and not lead.evidence_ids:
        raise ValueError("a lead cannot be verified without evidence ids")


def _stamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def lead_from_row(row: DBResearchLead) -> ResearchLead:
    updated = row.updated_at
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return ResearchLead(
        id=row.id,
        project_id=row.project_id,
        session_id=row.session_id,
        target=row.target,
        hypothesis=row.hypothesis,
        status=row.status,
        priority=row.priority,
        next_action=row.next_action,
        kill_reason=row.kill_reason,
        evidence_ids=list(row.evidence_ids or []),
        observation_ids=list(row.observation_ids or []),
        related_ids=list(row.related_ids or []),
        chain_ids=list(row.chain_ids or []),
        hypothesis_ids=backfill_lead_relationships(
            related_ids=list(row.related_ids or []),
            evidence_ids=list(row.evidence_ids or []),
            observation_ids=list(row.observation_ids or []),
            chain_ids=list(row.chain_ids or []),
            hypothesis_ids=list(getattr(row, "hypothesis_ids", None) or []),
            finding_ids=list(getattr(row, "finding_ids", None) or []),
            semantic_node_ids=list(getattr(row, "semantic_node_ids", None) or []),
        )[0],
        finding_ids=list(getattr(row, "finding_ids", None) or []),
        semantic_node_ids=list(getattr(row, "semantic_node_ids", None) or []),
        updated_at=updated.isoformat(),
    )


async def save_lead(session: AsyncSession, lead: ResearchLead) -> ResearchLead:
    """Write a lead into the existing research_leads table."""
    _validate(lead)
    lead.updated_at = datetime.now(UTC).isoformat()
    row = await session.get(DBResearchLead, lead.id)
    if row is None:
        row = DBResearchLead(id=lead.id, updated_at=_stamp(lead.updated_at))
        session.add(row)
    row.project_id = lead.project_id
    row.session_id = lead.session_id
    row.target = lead.target
    row.hypothesis = lead.hypothesis
    row.status = lead.status
    row.priority = lead.priority
    row.next_action = lead.next_action
    row.kill_reason = lead.kill_reason
    row.evidence_ids = list(lead.evidence_ids)
    row.observation_ids = list(lead.observation_ids)
    row.related_ids = list(lead.related_ids)
    row.chain_ids = list(lead.chain_ids)
    row.hypothesis_ids = list(lead.hypothesis_ids)
    row.finding_ids = list(lead.finding_ids)
    row.semantic_node_ids = list(lead.semantic_node_ids)
    row.updated_at = _stamp(lead.updated_at)
    await session.flush()
    return lead


async def load_project_leads(session: AsyncSession, project_id: str) -> list[ResearchLead]:
    result = await session.execute(
        select(DBResearchLead).where(DBResearchLead.project_id == project_id)
    )
    return [lead_from_row(row) for row in result.scalars()]


_CLOSED = frozenset({"KILLED", "REPORTED"})
_OPEN = frozenset({"NEW", "ACTIVE", "BLOCKED", "PROMOTED", "REPRODUCED"})


def lead_for_hypothesis(research: Any, hypothesis: Any) -> ResearchLead:
    """Create or update the lead that tracks one hypothesis. Closed leads stay closed."""
    related = str(getattr(hypothesis, "id", "") or "")
    found: ResearchLead | None = next(
        (
            item
            for item in research.leads
            if isinstance(item, ResearchLead) and related and related in item.hypothesis_ids
        ),
        None,
    )
    existing = found
    if existing is not None and existing.status in _CLOSED:
        return existing
    if existing is None:
        existing = new_lead(
            project_id=str(research.project_id),
            session_id=str(research.id),
            target=str(getattr(hypothesis, "target", "") or research.target),
            hypothesis=str(getattr(hypothesis, "title", "") or ""),
            priority=str(getattr(hypothesis, "severity", "") or "medium"),
        )
        existing.hypothesis_ids.append(related)
        research.leads.append(existing)
    existing.status = "ACTIVE"
    existing.next_action = str(
        getattr(hypothesis, "suggested_next_action", "") or existing.next_action
    )
    existing.evidence_ids = list(getattr(hypothesis, "supporting_evidence_ids", ()) or ())
    existing.hypothesis_ids = list(dict.fromkeys([*existing.hypothesis_ids, related]))
    _validate(existing)
    return existing


def planner_leads(
    leads: list[ResearchLead], *, stale_before: datetime | None = None
) -> dict[str, list[dict[str, object]]]:
    """Unresolved leads for the planner. Killed and reported leads stay out."""
    unresolved: list[dict[str, object]] = []
    stale: list[dict[str, object]] = []
    for lead in leads:
        if lead.status in _CLOSED or lead.status == "VERIFIED":
            continue
        if lead.status not in _OPEN:
            continue
        payload = lead.snapshot()
        payload["score"] = score_lead(
            impact=lead.priority if lead.priority in {"low", "medium", "high"} else "medium",
            evidence="static" if lead.evidence_ids else "none",
        )
        unresolved.append(payload)
        if stale_before is None or lead.status not in {"NEW", "ACTIVE", "BLOCKED"}:
            continue
        updated = _stamp(lead.updated_at) if lead.updated_at else stale_before
        if updated < stale_before:
            stale.append(payload)
    unresolved.sort(key=lambda item: (-int(str(item["score"])), str(item["id"])))
    return {"unresolved": unresolved, "stale": stale}
