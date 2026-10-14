"""Research strategies. The AI may recommend; BugForge decides allowed tools.

A strategy is more than a list of tool names: it names a goal, preferred
tools, allowed evidence types, minimum evidence expectations, a preferred
oracle, whether reproduction is required, and stop conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.security_agent.oracles import AuthorizationOracle, OracleType
from app.security_agent.states import ToolRiskLevel


class ResearchStrategy(StrEnum):
    PASSIVE_RECON = "passive_recon"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    API_BEHAVIOR = "api_behavior"
    BUSINESS_LOGIC = "business_logic"
    CLIENT_SIDE = "client_side"
    SERVER_SIDE = "server_side"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    REPRODUCTION = "reproduction"


@dataclass(frozen=True)
class StrategySpec:
    strategy: ResearchStrategy
    goal: str
    preferred_tools: tuple[str, ...]
    allowed_evidence_types: tuple[str, ...]
    minimum_evidence_expectations: tuple[str, ...]
    preferred_oracle: str
    reproduction_requirement: bool
    stop_conditions: tuple[str, ...]

    def snapshot(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "goal": self.goal,
            "preferred_tools": list(self.preferred_tools),
            "allowed_evidence_types": list(self.allowed_evidence_types),
            "minimum_evidence_expectations": list(self.minimum_evidence_expectations),
            "preferred_oracle": self.preferred_oracle,
            "reproduction_requirement": self.reproduction_requirement,
            "stop_conditions": list(self.stop_conditions),
        }


_STRATEGY_SPECS: dict[ResearchStrategy, StrategySpec] = {
    ResearchStrategy.PASSIVE_RECON: StrategySpec(
        strategy=ResearchStrategy.PASSIVE_RECON,
        goal="Map in-scope source, captured HTTP, and existing evidence without sending active probes.",
        preferred_tools=(
            "source_inspect",
            "evidence_inspect",
            "proxy_evidence",
            "exploratory_test",
        ),
        allowed_evidence_types=("source", "observation", "request", "response"),
        minimum_evidence_expectations=("at least one source or HTTP observation",),
        preferred_oracle=OracleType.SOURCE_RUNTIME_CONSISTENCY.value,
        reproduction_requirement=False,
        stop_conditions=("hypothesis_formed", "budget_exhausted", "operator_stop"),
    ),
    ResearchStrategy.AUTHENTICATION: StrategySpec(
        strategy=ResearchStrategy.AUTHENTICATION,
        goal="Determine how authentication is enforced and whether unauthenticated callers are rejected.",
        preferred_tools=("http_request", "browser_navigate", "source_inspect"),
        allowed_evidence_types=("request", "response", "browser_observation", "source"),
        minimum_evidence_expectations=("authenticated vs anonymous comparison",),
        preferred_oracle=AuthorizationOracle.AUTHENTICATED_USER_ONLY.value,
        reproduction_requirement=True,
        stop_conditions=("oracle_evaluated", "reproduction_complete", "blocked"),
    ),
    ResearchStrategy.AUTHORIZATION: StrategySpec(
        strategy=ResearchStrategy.AUTHORIZATION,
        goal="Test object/role/tenant authorization using isolated identities and an explicit oracle.",
        preferred_tools=("http_request", "api_test", "reproduce"),
        allowed_evidence_types=("request", "response", "reproduction", "observation"),
        minimum_evidence_expectations=(
            "authorization oracle",
            "identity A and B observations",
            "live reproduction",
        ),
        preferred_oracle=AuthorizationOracle.OWNER_ONLY.value,
        reproduction_requirement=True,
        stop_conditions=("reproduced_or_inconclusive", "budget_exhausted"),
    ),
    ResearchStrategy.INPUT_VALIDATION: StrategySpec(
        strategy=ResearchStrategy.INPUT_VALIDATION,
        goal="Probe input handling with a bounded fuzz/API envelope after human approval.",
        preferred_tools=("fuzz", "http_request", "api_test"),
        allowed_evidence_types=("observation", "request", "response", "reproduction"),
        minimum_evidence_expectations=("interesting differential", "reproduction"),
        preferred_oracle=OracleType.INVARIANT_VIOLATION.value,
        reproduction_requirement=True,
        stop_conditions=("interesting_result_reproduced", "budget_exhausted"),
    ),
    ResearchStrategy.API_BEHAVIOR: StrategySpec(
        strategy=ResearchStrategy.API_BEHAVIOR,
        goal="Exercise imported API operations under ScopeGuard with reviewed candidate requests.",
        preferred_tools=("api_test", "http_request", "source_inspect"),
        allowed_evidence_types=("request", "response", "source", "reproduction"),
        minimum_evidence_expectations=("candidate review", "executed request evidence"),
        preferred_oracle=OracleType.RESPONSE_FIELD.value,
        reproduction_requirement=True,
        stop_conditions=("spec_exhausted", "reproduction_required"),
    ),
    ResearchStrategy.BUSINESS_LOGIC: StrategySpec(
        strategy=ResearchStrategy.BUSINESS_LOGIC,
        goal="Check workflow invariants (replay, duplicate, skip-step) against an explicit expectation.",
        preferred_tools=("api_test", "http_request", "reproduce"),
        allowed_evidence_types=("request", "response", "reproduction"),
        minimum_evidence_expectations=("stated expectation", "observed transition"),
        preferred_oracle=OracleType.STATE_TRANSITION.value,
        reproduction_requirement=True,
        stop_conditions=("invariant_evaluated", "inconclusive"),
    ),
    ResearchStrategy.CLIENT_SIDE: StrategySpec(
        strategy=ResearchStrategy.CLIENT_SIDE,
        goal="Inspect client-side behavior in a lab or approved live browser context.",
        preferred_tools=("browser_navigate", "source_inspect", "http_request"),
        allowed_evidence_types=("browser_observation", "source", "request"),
        minimum_evidence_expectations=("browser observation or source excerpt",),
        preferred_oracle=OracleType.SENSITIVE_DATA_EXPOSURE.value,
        reproduction_requirement=True,
        stop_conditions=("observation_complete", "tool_unavailable"),
    ),
    ResearchStrategy.SERVER_SIDE: StrategySpec(
        strategy=ResearchStrategy.SERVER_SIDE,
        goal="Correlate server-side source with authorized HTTP observations.",
        preferred_tools=("source_inspect", "http_request", "nuclei_scan"),
        allowed_evidence_types=("source", "request", "response", "scanner_result"),
        minimum_evidence_expectations=(
            "source or HTTP observation",
            "scanner alerts are not verification",
        ),
        preferred_oracle=OracleType.SOURCE_RUNTIME_CONSISTENCY.value,
        reproduction_requirement=True,
        stop_conditions=("hypothesis_supported", "scan_ingested"),
    ),
    ResearchStrategy.CONFIGURATION: StrategySpec(
        strategy=ResearchStrategy.CONFIGURATION,
        goal="Review configuration and scanner policy without treating alerts as verified findings.",
        preferred_tools=("nuclei_scan", "zap_scan", "source_inspect"),
        allowed_evidence_types=("scanner_result", "source", "observation"),
        minimum_evidence_expectations=("authorized scan plan", "ingested results"),
        preferred_oracle=OracleType.INVARIANT_VIOLATION.value,
        reproduction_requirement=False,
        stop_conditions=("results_ingested", "tool_unavailable", "approval_required"),
    ),
    ResearchStrategy.DEPENDENCY: StrategySpec(
        strategy=ResearchStrategy.DEPENDENCY,
        goal="Identify dependency and template hits as leads, never as verified vulnerabilities.",
        preferred_tools=("nuclei_scan", "source_inspect"),
        allowed_evidence_types=("scanner_result", "source"),
        minimum_evidence_expectations=("scanner or source lead",),
        preferred_oracle=OracleType.SOURCE_RUNTIME_CONSISTENCY.value,
        reproduction_requirement=False,
        stop_conditions=("leads_recorded", "tool_unavailable"),
    ),
    ResearchStrategy.REPRODUCTION: StrategySpec(
        strategy=ResearchStrategy.REPRODUCTION,
        goal="Execute a ReproductionPlan with a deterministic oracle. Status codes alone are inconclusive.",
        preferred_tools=("reproduce", "http_request"),
        allowed_evidence_types=("reproduction", "request", "response"),
        minimum_evidence_expectations=("oracle", "live execution", "not replay"),
        preferred_oracle=OracleType.EXACT_RESPONSE.value,
        reproduction_requirement=True,
        stop_conditions=("reproduced", "not_reproduced", "inconclusive", "blocked"),
    ),
}


def spec_for(strategy: ResearchStrategy) -> StrategySpec:
    return _STRATEGY_SPECS[strategy]


def tools_for_strategy(strategy: ResearchStrategy) -> tuple[str, ...]:
    return spec_for(strategy).preferred_tools


def strategy_risk(strategy: ResearchStrategy) -> ToolRiskLevel:
    if strategy in {ResearchStrategy.PASSIVE_RECON, ResearchStrategy.CONFIGURATION}:
        return ToolRiskLevel.PASSIVE
    if strategy in {ResearchStrategy.INPUT_VALIDATION}:
        return ToolRiskLevel.ACTIVE
    if strategy in {ResearchStrategy.REPRODUCTION}:
        return ToolRiskLevel.ACTIVE
    return ToolRiskLevel.LOW_RISK_ACTIVE
