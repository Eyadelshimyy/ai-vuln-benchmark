"""Conservative business-logic hypotheses. Differences are not vulnerabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BusinessLogicIndicator(StrEnum):
    SEQUENCE_DEPENDENCE = "sequence_dependence"
    INCONSISTENT_STATE = "inconsistent_state_transition"
    REPLAY = "replay_behavior"
    PRIVILEGE_TRANSITION = "unexpected_privilege_transition"
    DUPLICATE_OPERATION = "duplicate_operation"
    MISSING_PREREQUISITE = "missing_prerequisite"
    STATE_INCONSISTENCY = "state_inconsistency"


@dataclass
class BusinessLogicHypothesis:
    indicator: BusinessLogicIndicator
    expectation: str
    observation: str
    evidence_ids: tuple[str, ...] = ()
    is_vulnerability: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator.value,
            "expectation": self.expectation,
            "observation": self.observation,
            "evidence_ids": list(self.evidence_ids),
            "is_vulnerability": False,
        }


def hypothesize(
    indicator: BusinessLogicIndicator,
    *,
    expectation: str,
    observation: str,
    evidence_ids: tuple[str, ...] = (),
) -> BusinessLogicHypothesis:
    if not expectation.strip():
        raise ValueError("A defined security expectation is required")
    return BusinessLogicHypothesis(
        indicator=indicator,
        expectation=expectation,
        observation=observation,
        evidence_ids=evidence_ids,
        is_vulnerability=False,
    )
