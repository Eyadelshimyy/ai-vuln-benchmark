"""Planner methodology. Routes and experiments do not authorize tools or verify findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureMap:
    feature: str
    entry_points: tuple[str, ...] = ()
    actors: tuple[str, ...] = ()
    state_touched: tuple[str, ...] = ()
    external_calls: tuple[str, ...] = ()
    value_movement: tuple[str, ...] = ()
    validations: tuple[str, ...] = ()
    callbacks: tuple[str, ...] = ()
    alternate_paths: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "entry_points": list(self.entry_points),
            "actors": list(self.actors),
            "state_touched": list(self.state_touched),
            "external_calls": list(self.external_calls),
            "value_movement": list(self.value_movement),
            "validations": list(self.validations),
            "callbacks": list(self.callbacks),
            "alternate_paths": list(self.alternate_paths),
        }


@dataclass
class WhatIf:
    prompt: str
    feature: str = ""
    lead_id: str = ""

    def snapshot(self) -> dict[str, str]:
        return {"prompt": self.prompt, "feature": self.feature, "lead_id": self.lead_id}


ROUTES = frozenset({"wide", "deep"})


def set_route(session: Any, route: str) -> str:
    cleaned = route.strip().lower()
    if cleaned not in ROUTES:
        raise ValueError("research route must be wide or deep")
    session.research_route = cleaned
    return cleaned


def anomaly_lead(session: Any, *, target: str, summary: str, kind: str) -> Any:
    """Open a lead from unexpected behavior. The lead is not a finding."""
    from app.security_agent.leads import new_lead

    lead = new_lead(
        project_id=str(session.project_id),
        session_id=str(session.id),
        target=target,
        hypothesis=f"{kind}: {summary}",
        priority="medium",
    )
    lead.next_action = f"Investigate anomaly `{kind}` before promoting it"
    session.leads.append(lead)
    return lead


def what_if(feature: str, question: str, *, lead_id: str = "") -> WhatIf:
    return WhatIf(prompt=question, feature=feature, lead_id=lead_id)


def planner_brief(session: Any) -> dict[str, Any]:
    from datetime import UTC, datetime, timedelta

    from app.security_agent.leads import planner_leads

    route = str(getattr(session, "research_route", "wide") or "wide")
    leads = planner_leads(
        list(getattr(session, "leads", []) or []),
        stale_before=datetime.now(UTC) - timedelta(hours=24),
    )
    focus = (
        "Inspect many features and record leads."
        if route == "wide"
        else ("Stay on one feature until its leads are killed, blocked, or evidenced.")
    )
    return {
        "route": route,
        "focus": focus,
        "unresolved_leads": leads["unresolved"],
        "stale_leads": leads["stale"],
        "chains": [
            item.snapshot()
            for item in getattr(session, "chains", []) or []
            if hasattr(item, "snapshot")
        ],
        "what_if_examples": [
            "What happens if this value is attacker-controlled?",
            "What happens if this condition is exactly zero?",
            "What happens if this path executes instead of the normal path?",
            "What happens if the callback occurs before this write?",
        ],
        "verification": "Experiments stay proposals until a tool records evidence.",
    }
