"""API workflow state machine. Suspicious transitions are hypotheses, not findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowState:
    name: str
    evidence_id: str = ""


@dataclass
class WorkflowTransition:
    source: str
    destination: str
    method: str
    url: str
    status: int | None = None
    legal: bool = True
    evidence_id: str = ""


@dataclass
class APIStateMachine:
    states: list[WorkflowState] = field(default_factory=list)
    transitions: list[WorkflowTransition] = field(default_factory=list)

    def add_state(self, name: str, *, evidence_id: str = "") -> WorkflowState:
        state = WorkflowState(name=name, evidence_id=evidence_id)
        self.states.append(state)
        return state

    def add_transition(self, transition: WorkflowTransition) -> None:
        self.transitions.append(transition)

    def suspicious(self) -> list[WorkflowTransition]:
        return [item for item in self.transitions if not item.legal]

    def snapshot(self) -> dict[str, Any]:
        return {
            "states": [state.name for state in self.states],
            "transitions": [
                {
                    "from": item.source,
                    "to": item.destination,
                    "method": item.method,
                    "url": item.url,
                    "status": item.status,
                    "legal": item.legal,
                    "evidence_id": item.evidence_id,
                }
                for item in self.transitions
            ],
        }
