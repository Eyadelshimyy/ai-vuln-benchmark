"""Single-step capability suggestion.

`choose_next` names one capability from one observation. It plans nothing, it
ignores budget and availability, and it cannot raise a limit. Adaptive,
budgeted, persisted research is `app.discovery.orchestration` (Phase 49),
which does not call this function.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchObservation:
    uncertainty: str
    evidence: str


@dataclass(frozen=True)
class ResearchAction:
    capability: str
    reason: str


def choose_next(
    observation: ResearchObservation,
    *,
    stalled: bool,
    has_property: bool,
    static_known: bool,
) -> ResearchAction:
    """Pick a complementary capability. The choice is not a confirmation."""
    if observation.uncertainty in {"cross-contract", "protocol"}:
        return ResearchAction(
            "cross_contract_analysis",
            "protocol graph evidence is missing",
        )
    if observation.uncertainty == "runtime":
        return ResearchAction(
            "runtime_validation",
            "runtime evidence is missing",
        )
    if observation.uncertainty == "fork":
        return ResearchAction(
            "fork_validation",
            "fork-state evidence is missing",
        )
    if observation.uncertainty == "differential":
        return ResearchAction(
            "differential_validation",
            "differential evidence is missing",
        )
    if observation.uncertainty == "economic":
        return ResearchAction(
            "economic_simulation",
            "an economic candidate has no asset-delta evidence",
        )
    if observation.uncertainty in {"reachability", "stalled"} or stalled:
        return ResearchAction(
            "symbolic_execution",
            "reachability stalled; symbolic execution is the next capability",
        )
    if has_property or observation.uncertainty == "property":
        return ResearchAction(
            "test_execution",
            "a bounded property can be replayed",
        )
    if static_known or observation.uncertainty == "candidate":
        return ResearchAction(
            "fuzzing",
            "a static candidate needs stateful exploration",
        )
    return ResearchAction("static_analysis", "no static candidate is known yet")
