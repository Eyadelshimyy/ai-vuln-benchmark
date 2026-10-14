"""Researcher-facing workbench views. Reuses SecurityResearchAgent; no second executor."""

from __future__ import annotations

from typing import Any

from app.domain.findings import FindingStatus
from app.security_agent.cost import estimate_operation_cost
from app.security_agent.orchestrator import AdvancedResearchOrchestrator
from app.security_agent.states import FindingWorkbenchState, HypothesisStatus
from app.security_agent.strategies import spec_for
from app.security_agent.tool_status import describe_tools
from app.security_testing.secrets import redact_text


def dashboard(orch: AdvancedResearchOrchestrator) -> dict[str, Any]:
    body = orch.dashboard()
    body["tools"] = describe_tools(orch.agent)
    body["identities"] = (
        orch.session.identities.snapshot() if orch.session.identities is not None else None
    )
    return body


def timeline(
    orch: AdvancedResearchOrchestrator,
    *,
    tool: str | None = None,
    finding: str | None = None,
    strategy: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    items = []
    for event in orch.session.timeline:
        if tool and event.tool != tool:
            continue
        if finding and event.finding_id != finding:
            continue
        event_strategy = getattr(event, "strategy", "") or orch.session.strategy
        event_state = getattr(event, "state", "") or orch.session.state.value
        if strategy and event_strategy != strategy:
            continue
        if state and event_state != state:
            continue
        payload = event.snapshot()
        payload["result"] = redact_text(str(payload.get("result") or ""))
        payload["decision"] = redact_text(str(payload.get("decision") or ""))
        items.append(payload)
    return {
        "items": items,
        "filters": {"tool": tool, "finding": finding, "strategy": strategy, "state": state},
        "strategy": orch.session.strategy,
        "state": orch.session.state.value,
    }


def next_action_review(
    orch: AdvancedResearchOrchestrator,
    *,
    tool: str,
    arguments: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    spec = orch.agent.tools.spec(tool)
    remaining = int(orch.session.budget.remaining().get("requests") or 0)
    estimate = estimate_operation_cost(tool, arguments, spec, remaining_requests=remaining)
    explanation = orch.explain_next(tool, reason=reason, arguments=arguments)
    explanation.estimated_requests = estimate.estimated_requests
    explanation.estimated_units = estimate.estimated_units
    explanation.estimate_unit = estimate.unit
    explanation.is_estimate = True
    payload = explanation.snapshot()
    payload["what"] = f"{tool} against {explanation.target}"
    payload["why"] = reason
    payload["expected_evidence"] = spec.description
    payload["current_scope_decision"] = explanation.scope_decision
    payload["arguments_preview"] = _preview_arguments(arguments)
    payload["approval_required"] = bool(spec.requires_human_approval)
    return payload


def findings_workbench(orch: AdvancedResearchOrchestrator) -> dict[str, Any]:
    items = []
    for finding in orch.session.findings:
        hyp = next(
            (item for item in orch.session.hypotheses if item.id == str(finding.hypothesis or "")),
            None,
        )
        items.append(
            {
                "id": str(finding.id),
                "title": finding.title,
                "vulnerability_class": finding.vulnerability_class,
                "target": finding.target,
                "confidence": finding.confidence,
                "severity": finding.impact,
                "evidence_strength": hyp.evidence_strength if hyp else len(finding.evidence.items),
                "supporting_evidence": [
                    redact_text(item.summary) for item in finding.evidence.items
                ],
                "contradicting_evidence": list(hyp.contradicting_evidence_ids) if hyp else [],
                "reproduction": finding.reproduction,
                "verification_state": _workbench_state(finding, hyp).value,
                "status": finding.status.value,
                "hackerone_eligibility": None,
                "report_status": "not_submitted",
            }
        )
    return {"items": items}


def evidence_explorer(orch: AdvancedResearchOrchestrator) -> dict[str, Any]:
    items = []
    for node in sorted(orch.session.graph.nodes.values(), key=lambda item: item.id):
        extra = dict(node.extra) if isinstance(node.extra, dict) else {}
        extra.pop("authorization", None)
        extra.pop("cookie", None)
        extra.pop("token", None)
        items.append(
            {
                "id": node.id,
                "kind": node.kind,
                "source": node.source,
                "provenance": node.provenance,
                "tool": node.source,
                "timestamp": node.created_at.isoformat(),
                "related_hypothesis": extra.get("hypothesis_id") or "",
                "related_finding": extra.get("finding_id") or "",
                "related_tool_call": extra.get("tool_call_id") or "",
                "summary": redact_text(node.summary),
                "http": node.kind in {"request", "response", "observation"}
                and node.source == "http_request",
                "browser": node.kind == "browser_observation" or node.source == "browser_navigate",
                "scanner": node.kind == "scanner_result",
                "api": node.source == "api_test",
                "fuzzing": node.source == "fuzz",
                "reproduction": node.kind == "reproduction",
                "manual": node.source in {"operator", "manual"},
            }
        )
    return {"items": items}


def evidence_graph_view(orch: AdvancedResearchOrchestrator) -> dict[str, Any]:
    graph = orch.session.graph.snapshot()
    return {
        "nodes": graph.get("nodes") or [],
        "edges": graph.get("edges") or [],
        "chains": [
            {
                "finding": str(finding.id),
                "hypothesis": finding.hypothesis,
                "evidence": list(finding.observation_refs),
                "status": finding.status.value,
            }
            for finding in orch.session.findings
        ],
        "decorative": False,
    }


def identity_workbench(orch: AdvancedResearchOrchestrator) -> dict[str, Any]:
    pair = orch.session.identities
    if pair is None:
        return {"a": None, "b": None, "isolated": False}
    snap = pair.snapshot()
    snap["never_displays_raw_credentials"] = True
    return dict(snap)


def strategy_view(orch: AdvancedResearchOrchestrator) -> dict[str, Any]:
    from app.security_agent.strategies import ResearchStrategy

    try:
        strategy = ResearchStrategy(orch.session.strategy)
    except ValueError:
        strategy = orch.strategy
    return spec_for(strategy).snapshot()


def _preview_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    secret_parts = ("token", "cookie", "authorization", "password")
    preview: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "headers":
            continue
        lowered = key.lower()
        preview[key] = "<secret>" if any(part in lowered for part in secret_parts) else value
    return preview


def _workbench_state(finding: Any, hyp: Any) -> FindingWorkbenchState:
    if finding.status is FindingStatus.REJECTED:
        return FindingWorkbenchState.REJECTED
    if finding.status is FindingStatus.VERIFIED or finding.status is FindingStatus.HUMAN_ACCEPTED:
        return FindingWorkbenchState.VERIFIED
    if finding.status is FindingStatus.REPRODUCED:
        return FindingWorkbenchState.REPRODUCED
    if hyp is not None and hyp.status is HypothesisStatus.REQUIRES_REPRODUCTION:
        return FindingWorkbenchState.REQUIRES_REPRODUCTION
    if finding.status is FindingStatus.CORROBORATED:
        return FindingWorkbenchState.CORROBORATED
    if hyp is not None and hyp.status is HypothesisStatus.INCONCLUSIVE:
        return FindingWorkbenchState.INCONCLUSIVE
    return FindingWorkbenchState.POTENTIAL
