"""Advanced research orchestrator. Reuses SecurityResearchAgent; no second executor.

Supporting evidence IDs never imply successful completion. A hypothesis is
sufficient for COMPLETED_SUCCESS only after live (non-replay) verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.findings import FindingStatus
from app.security_agent.agent import AgentDecision, ResearchSession, SecurityResearchAgent
from app.security_agent.correlation import prioritize
from app.security_agent.cost import estimate_operation_cost
from app.security_agent.states import (
    EvidenceCompleteness,
    HypothesisStatus,
    ResearchState,
    TerminationReason,
)
from app.security_agent.strategies import ResearchStrategy, spec_for, tools_for_strategy
from app.security_testing.errors import RestrictedActivityError

_NON_LIVE_PROVENANCE = frozenset(
    {
        "ai_hypothesis",
        "replay",
        "tool_request",
        "scanner_plan",
        "generated_test",
        "sandbox_execution",
    }
)
_AUTHZ_CLASSES = frozenset(
    {
        "idor",
        "broken_access_control",
        "broken_authorization",
        "privilege_escalation",
        "insecure_direct_object_reference",
        "bola",
        "bfla",
    }
)


@dataclass
class NextActionExplanation:
    tool: str
    target: str
    reason: str
    expected_evidence: str
    estimated_requests: int
    risk_level: str
    approval_required: bool
    is_estimate: bool = True
    estimate_unit: str = "requests"
    estimated_units: int = 1
    planned: int = 0
    reserved: int = 0
    consumed: int = 0
    scope_decision: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "target": self.target,
            "reason": self.reason,
            "expected_evidence": self.expected_evidence,
            "estimated_requests": self.estimated_requests,
            "risk_level": self.risk_level,
            "approval_required": self.approval_required,
            "is_estimate": self.is_estimate,
            "estimate_unit": self.estimate_unit,
            "estimated_units": self.estimated_units,
            "planned": self.planned,
            "reserved": self.reserved,
            "consumed": self.consumed,
            "scope_decision": self.scope_decision,
        }


@dataclass
class AdvancedResearchOrchestrator:
    agent: SecurityResearchAgent
    strategy: ResearchStrategy = ResearchStrategy.PASSIVE_RECON
    next_action: NextActionExplanation | None = None
    false_positives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def session(self) -> ResearchSession:
        return self.agent.session

    def recommend_strategy(self, suggested: str) -> ResearchStrategy:
        try:
            strategy = ResearchStrategy(suggested)
        except ValueError:
            strategy = ResearchStrategy.PASSIVE_RECON
        self.strategy = strategy
        self.session.strategy = strategy.value
        return strategy

    def permitted_tools(self) -> tuple[str, ...]:
        wanted = tools_for_strategy(self.strategy)
        known = set(self.agent.tools.known())
        disabled = self.session.disabled_tools
        return tuple(name for name in wanted if name in known and name not in disabled)

    def identify_missing_evidence(self) -> list[str]:
        missing: list[str] = []
        graph = self.session.graph
        for hyp in self.session.hypotheses:
            missing.extend(self._missing_for_hypothesis(hyp, graph))
        return missing

    def _missing_for_hypothesis(self, hyp: Any, graph: Any) -> list[str]:
        missing: list[str] = []
        live_support: list[str] = []
        for evidence_id in hyp.supporting_evidence_ids:
            node = graph.nodes.get(evidence_id) if hasattr(graph, "nodes") else None
            if node is None:
                missing.append(f"{hyp.id}:missing_id:{evidence_id}")
                continue
            if not _linked_to_hypothesis(graph, hyp.id, evidence_id):
                missing.append(f"{hyp.id}:unlinked:{evidence_id}")
                linked = False
            else:
                linked = True
            provenance = str(getattr(node, "provenance", "") or "")
            if not provenance:
                missing.append(f"{hyp.id}:invalid_provenance:{evidence_id}")
                continue
            if provenance in _NON_LIVE_PROVENANCE:
                missing.append(f"{hyp.id}:non_live_provenance:{evidence_id}")
                continue
            if linked:
                live_support.append(evidence_id)
        if not live_support:
            missing.append(f"{hyp.id}:no_evidence")
        if hyp.contradicting_evidence_ids:
            for evidence_id in hyp.contradicting_evidence_ids:
                node = graph.nodes.get(evidence_id) if hasattr(graph, "nodes") else None
                if node is None:
                    missing.append(f"{hyp.id}:contradiction_missing:{evidence_id}")
                elif str(getattr(node, "provenance", "") or "") not in _NON_LIVE_PROVENANCE:
                    missing.append(f"{hyp.id}:contradiction")
                    break
        if _needs_reproduction(hyp, live_support):
            if not _has_live_reproduction(self.session, hyp):
                missing.append(f"{hyp.id}:reproduction")
        klass = (hyp.vulnerability_class or "").lower()
        if any(token in klass for token in _AUTHZ_CLASSES):
            if not _has_identity_or_authz_evidence(graph, hyp):
                missing.append(f"{hyp.id}:class_requirement:authorization_oracle")
        exploratory = _exploratory_for(graph, hyp.id)
        if any(item.get("repeat") == "flaky" for item in exploratory):
            missing.append(f"{hyp.id}:exploratory_flaky")
        elif exploratory and all(item.get("meaningful") is not True for item in exploratory):
            if any(item.get("classification") == "FAILED" for item in exploratory):
                missing.append(f"{hyp.id}:exploratory_follow_up")
        return missing

    def classify_hypothesis(self, hyp: Any) -> EvidenceCompleteness:
        missing = self._missing_for_hypothesis(hyp, self.session.graph)
        if hyp.status is HypothesisStatus.VERIFIED and not missing:
            return EvidenceCompleteness.VERIFIED
        if hyp.status is HypothesisStatus.INCONCLUSIVE or any(
            item.endswith(":contradiction") for item in missing
        ):
            if ":reproduction" in str(missing):
                return EvidenceCompleteness.REPRODUCTION_REQUIRED
            return EvidenceCompleteness.INCONCLUSIVE
        if any(item.endswith(":reproduction") for item in missing) or (
            hyp.status is HypothesisStatus.REQUIRES_REPRODUCTION
        ):
            return EvidenceCompleteness.REPRODUCTION_REQUIRED
        if (
            _has_live_reproduction(self.session, hyp)
            and hyp.status is not HypothesisStatus.VERIFIED
        ):
            return EvidenceCompleteness.REPRODUCED
        if hyp.supporting_evidence_ids and not any(
            item.endswith(":no_evidence") for item in missing
        ):
            return EvidenceCompleteness.HYPOTHESIS_SUPPORTED
        if self.session.graph.nodes:
            return EvidenceCompleteness.OBSERVATION_COMPLETE
        return EvidenceCompleteness.INCONCLUSIVE

    def research_completeness(self) -> EvidenceCompleteness:
        if any(item.is_verified for item in self.session.findings):
            if all(
                _finding_has_live_verification(item, self.session)
                for item in self.session.findings
                if item.is_verified
            ):
                return EvidenceCompleteness.VERIFIED
            return EvidenceCompleteness.INCONCLUSIVE
        if any(item.status is FindingStatus.REPRODUCED for item in self.session.findings):
            return EvidenceCompleteness.REPRODUCED
        ranked = prioritize(self.session.hypotheses) if self.session.hypotheses else []
        if ranked:
            return self.classify_hypothesis(ranked[0])
        if self.session.graph.nodes:
            return EvidenceCompleteness.OBSERVATION_COMPLETE
        return EvidenceCompleteness.INCONCLUSIVE

    def should_reproduce(self) -> bool:
        needs = any(
            self.classify_hypothesis(item) is EvidenceCompleteness.REPRODUCTION_REQUIRED
            for item in self.session.hypotheses
        )
        return needs and "reproduce" in self.permitted_tools()

    def evidence_sufficient(self) -> bool:
        """True only for live-verified findings. Supporting IDs are never enough."""
        verified = [
            item
            for item in self.session.findings
            if item.is_verified and _finding_has_live_verification(item, self.session)
        ]
        return bool(verified)

    def explain_next(
        self,
        tool: str,
        *,
        reason: str,
        target: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> NextActionExplanation:
        spec = self.agent.tools.spec(tool)
        remaining = int(self.session.budget.remaining().get("requests") or 0)
        estimate = estimate_operation_cost(
            tool, arguments or {}, spec, remaining_requests=remaining
        )
        used_raw = self.session.budget.snapshot()["used"]
        used = used_raw if isinstance(used_raw, dict) else {}
        consumed = int(
            used.get("requests", 0)
            if estimate.unit == "requests"
            else used.get("fuzz_requests", 0)
            if estimate.unit == "fuzz_requests"
            else used.get("browser_actions", 0)
            if estimate.unit == "browser_actions"
            else used.get("exploratory_tests", 0)
            if estimate.unit == "exploratory_tests"
            else used.get("tool_calls", 0)
        )
        self.session.budget.plan(estimate.unit, estimate.estimated_units)
        self.session.budget.reserve(estimate.unit, estimate.estimated_units)
        self.next_action = NextActionExplanation(
            tool=tool,
            target=target or self.session.target,
            reason=reason,
            expected_evidence=spec.description,
            estimated_requests=estimate.estimated_requests,
            risk_level=spec.risk_level.value,
            approval_required=bool(spec.requires_human_approval),
            is_estimate=True,
            estimate_unit=estimate.unit,
            estimated_units=estimate.estimated_units,
            planned=estimate.estimated_units,
            reserved=estimate.estimated_units,
            consumed=consumed,
            scope_decision=self._scope_decision_summary(target or self.session.target),
        )
        self.session.next_action = self.next_action.snapshot()
        return self.next_action

    def reject_escalation(self, kind: str) -> None:
        raise RestrictedActivityError(kind)

    async def step(self) -> AgentDecision:
        if self.session.budget.exhausted():
            self.session.state = ResearchState.BUDGET_EXHAUSTED
            self.session.termination_reason = TerminationReason.BUDGET_EXHAUSTED
            return AgentDecision(kind="budget_exhausted")
        if self.evidence_sufficient() and not self.should_reproduce():
            self.session.state = ResearchState.COMPLETED_SUCCESS
            self.session.termination_reason = TerminationReason.COMPLETED_SUCCESS
            return AgentDecision(kind="complete", note="verified_evidence")
        if self.should_reproduce():
            self.explain_next("reproduce", reason="hypothesis requires live reproduction")
            # Never silently skip reproduction.
        decision = await self.agent.step()
        if decision.kind == "complete" and not self.evidence_sufficient():
            if self.session.findings or self.session.hypotheses:
                self.session.state = ResearchState.INCONCLUSIVE
                self.session.termination_reason = TerminationReason.INCONCLUSIVE
                return AgentDecision(kind="complete", note="inconclusive")
            self.session.state = ResearchState.COMPLETED_NO_FINDINGS
            self.session.termination_reason = TerminationReason.COMPLETED_NO_FINDINGS
            return AgentDecision(kind="complete", note="no_findings")
        return decision

    def dashboard(self) -> dict[str, Any]:
        ranked = prioritize(self.session.hypotheses)
        current = ranked[0] if ranked else None
        completeness = (
            self.classify_hypothesis(current) if current else self.research_completeness()
        )
        engine = self.session.engine.session
        dry_run = bool(self.session.engine.safety.dry_run or engine.dry_run)
        active = bool(engine.active_testing_enabled and not dry_run)
        strategy_spec = spec_for(self.strategy)
        return {
            "project": self.session.project_id,
            "program": self.session.program_handle,
            "target": self.session.target,
            "mode": self.session.mode.value,
            "controller": self.session.controller.value,
            "ai_controller": (
                "cursor" if self.session.controller.value == "cursor" else "bugforge_llm"
            ),
            "ai_execution": (
                "none"
                if self.session.controller.value == "cursor"
                else self.session.provider.provider_name
            ),
            "session_kind": "LAB" if self.session.mode.value == "lab" else "LIVE",
            "execution_mode": "ACTIVE" if active else "DRY-RUN",
            "scope": {
                "program": engine.scope.program_name,
                "lab_mode": engine.scope.lab_mode,
                "includes": [rule.identifier for rule in engine.scope.includes][:40],
            },
            "scope_status": engine.scope.program_name,
            "strategy": self.strategy.value,
            "strategy_spec": strategy_spec.snapshot(),
            "hypothesis": current.snapshot() if current else None,
            "current_hypothesis": current.snapshot() if current else None,
            "evidence_strength": current.evidence_strength if current else 0,
            "evidence_completeness": completeness.value,
            "contradicting_evidence": list(current.contradicting_evidence_ids) if current else [],
            "current_tool": (self.next_action.tool if self.next_action else None),
            "tools_used": list(self.session.tool_history),
            "remaining_budget": self.session.budget.remaining(),
            "exploratory": _exploratory_dashboard(self.session),
            "budget": self.session.budget.snapshot(),
            "authorization": {
                "active_testing": engine.active_testing_enabled,
                "fuzzing": engine.fuzzing_enabled,
                "dry_run": dry_run,
                "mode": self.session.mode.value,
                "session_kind": "LAB" if self.session.mode.value == "lab" else "LIVE",
                "execution_mode": "ACTIVE" if active else "DRY-RUN",
            },
            "approval_status": self.session.engine.approvals.snapshot(),
            "approval_requirements": [
                spec.approval_kind
                for spec in (self.agent.tools.spec(name) for name in self.permitted_tools())
                if spec.approval_kind
            ],
            "research_state": self.session.state.value,
            "termination_reason": self.session.termination_reason.value
            if self.session.termination_reason
            else None,
            "reproduction_status": current.reproducibility if current else "unknown",
            "next_action": self.next_action.snapshot() if self.next_action else None,
        }

    def record_false_positive(self, *, why: str, evidence: str, source: str) -> None:
        self.false_positives.append({"why": why, "evidence": evidence, "source": source})

    def _scope_decision_summary(self, target: str) -> str:
        try:
            decision = self.session.engine.authorize(target, tool="http_request", active=False)
        except Exception:
            return "unevaluated"
        allowed = getattr(decision, "allowed", None)
        reason = getattr(decision, "reason", "") or ""
        if allowed is True:
            return f"in_scope:{reason}" if reason else "in_scope"
        if allowed is False:
            return f"blocked:{reason}" if reason else "blocked"
        return str(decision)


def _linked_to_hypothesis(graph: Any, hypothesis_id: str, evidence_id: str) -> bool:
    edges = getattr(graph, "edges", ())
    for src, dst, rel in edges:
        if rel in {"supports", "contradicted_by"} and (
            (src == hypothesis_id and dst == evidence_id)
            or (dst == hypothesis_id and src == evidence_id)
        ):
            return True
    node = graph.nodes.get(evidence_id)
    extra = getattr(node, "extra", {}) if node is not None else {}
    if isinstance(extra, dict) and extra.get("hypothesis_id") == hypothesis_id:
        return True
    # Supporting IDs listed on a hypothesis are not a substitute for a graph link.
    return False


def _needs_reproduction(hyp: Any, live_support: list[str]) -> bool:
    if hyp.status is HypothesisStatus.REQUIRES_REPRODUCTION:
        return True
    if hyp.status in {
        HypothesisStatus.VERIFIED,
        HypothesisStatus.REJECTED,
        HypothesisStatus.DISPROVED,
    }:
        return False
    if live_support or hyp.status is HypothesisStatus.SUPPORTED:
        return True
    return False


def _has_live_reproduction(session: ResearchSession, hyp: Any) -> bool:
    for node in session.graph.nodes.values():
        if node.kind != "reproduction":
            continue
        if str(node.provenance or "") in _NON_LIVE_PROVENANCE:
            continue
        extra = node.extra if isinstance(node.extra, dict) else {}
        if extra.get("hypothesis_id") == hyp.id:
            return True
        if hyp.id in extra.get("hypothesis_ids", []):
            return True
    for finding in session.findings:
        if finding.status is FindingStatus.REPRODUCED and (
            str(finding.hypothesis or "") == hyp.id or finding.target == hyp.target
        ):
            if (
                _finding_has_live_verification(finding, session)
                or finding.evidence.verifying_items()
            ):
                return True
    return False


def _has_identity_or_authz_evidence(graph: Any, hyp: Any) -> bool:
    for src, dst, rel in getattr(graph, "edges", ()):
        if rel == "identity_diff":
            return True
    for node in getattr(graph, "nodes", {}).values():
        extra = node.extra if isinstance(node.extra, dict) else {}
        if extra.get("oracle") or extra.get("authorization_oracle"):
            return True
        if extra.get("hypothesis_id") == hyp.id and "authorization" in str(node.kind):
            return True
    return False


def _exploratory_for(graph: Any, hypothesis_id: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for node in getattr(graph, "nodes", {}).values():
        if getattr(node, "kind", "") != "exploratory_attempt":
            continue
        extra = node.extra if isinstance(node.extra, dict) else {}
        if extra.get("hypothesis_id") != hypothesis_id:
            continue
        found.append(extra)
    return found


def _exploratory_dashboard(session: ResearchSession) -> dict[str, Any]:
    attempts = [node for node in session.graph.nodes.values() if node.kind == "exploratory_attempt"]
    last = attempts[-1] if attempts else None
    extra = last.extra if last is not None and isinstance(last.extra, dict) else {}
    from app.core.config import get_settings

    remaining = session.budget.remaining()
    return {
        "enabled": bool(get_settings().security_agent_exploratory_enabled),
        "count": len(attempts),
        "last_attempt": None
        if last is None
        else {
            "id": last.id,
            "classification": extra.get("classification", ""),
            "hypothesis_id": extra.get("hypothesis_id", ""),
            "reproducibility": extra.get("snapshot", ""),
            "evidence_id": last.id,
            "verified": False,
        },
        "follow_up_recommended": extra.get("meaningful") is True
        and extra.get("classification") == "FAILED",
        "remaining_tests": remaining.get("exploratory_tests"),
        "remaining_iterations": remaining.get("exploratory_iterations"),
        "network": "none",
        "lab_only": True,
        "verified": False,
    }


def _finding_has_live_verification(finding: Any, session: ResearchSession) -> bool:
    items = finding.evidence.verifying_items() if finding.evidence else ()
    live = [
        item
        for item in items
        if getattr(item.provenance, "value", str(item.provenance)) not in _NON_LIVE_PROVENANCE
    ]
    if not live:
        return False
    for ref in finding.observation_refs:
        node = session.graph.nodes.get(ref)
        if node is not None and str(node.provenance or "") in _NON_LIVE_PROVENANCE:
            return False
    return True
