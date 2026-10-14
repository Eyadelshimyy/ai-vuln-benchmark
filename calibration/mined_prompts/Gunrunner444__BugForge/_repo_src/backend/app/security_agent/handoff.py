"""Prepare a verified finding for the existing HackerOne workflow. Human still approves.

Scope matching uses :class:`HackerOneScopeEvaluator`. Substring containment is
never used to infer that a target is in scope.
"""

from __future__ import annotations

from typing import Any

from app.adapters.hackerone.evaluator import HackerOneScopeEvaluator
from app.adapters.hackerone.models import (
    HackerOneProgram,
    StructuredScopeRecord,
)
from app.adapters.hackerone.models import (
    ScopeMode as HackerOneScopeMode,
)
from app.domain.findings import SecurityFinding
from app.security_agent.agent import ResearchSession
from app.security_agent.export import export_package
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.scope_model import ScopeMode


def prepare_hackerone_handoff(
    session: ResearchSession,
    finding: SecurityFinding,
    *,
    program: HackerOneProgram | None = None,
) -> dict[str, Any]:
    if not finding.is_verified:
        raise RestrictedActivityError("handoff_requires_verified_finding")
    resolved = program or _program_from_session(session)
    evaluator = HackerOneScopeEvaluator()
    target = finding.target or session.target
    decision = evaluator.evaluate(
        resolved,
        target,
        vulnerability_class=finding.vulnerability_class,
    )
    snapshot = decision.snapshot(resolved)
    return {
        "target": target,
        "matched_structured_scope_id": decision.structured_scope_id,
        "structured_scope_id": decision.structured_scope_id,
        "asset_type": decision.asset_type,
        "eligibility": {
            "in_scope": decision.in_scope,
            "eligible_for_submission": decision.eligible_for_submission,
            "eligible_for_bounty": decision.eligible_for_bounty,
            "allowed": decision.allowed,
        },
        "program": resolved.handle,
        "scope_snapshot": snapshot.as_dict(),
        "evaluation_reason": decision.reason,
        "scope_match": decision.asset_identifier or "",
        "weakness_candidate": finding.vulnerability_class,
        "severity_candidate": finding.impact or finding.confidence,
        "reproduction": finding.reproduction or finding.observed_behavior,
        "evidence_package": export_package(session, finding),
        "sanitized_report_candidate": {
            "title": finding.report_title or finding.title,
            "description": finding.report_description or finding.description,
        },
        "cannot_approve": True,
        "cannot_submit": True,
    }


def _program_from_session(session: ResearchSession) -> HackerOneProgram:
    scope = session.engine.session.scope
    records: list[StructuredScopeRecord] = []
    for rule in scope.includes:
        records.append(
            StructuredScopeRecord(
                id=str(rule.structured_scope_id or ""),
                asset_type_raw=str(getattr(rule.asset_type, "value", rule.asset_type)),
                asset_type=rule.asset_type,
                asset_identifier=rule.identifier,
                instruction=rule.instructions or "",
                eligible_for_submission=bool(rule.eligible_for_submission),
                eligible_for_bounty=bool(rule.eligible_for_bounty),
            )
        )
    mode = HackerOneScopeMode.CLOSED
    raw_mode = getattr(scope.scope_mode, "value", scope.scope_mode)
    if raw_mode == ScopeMode.OPEN.value or raw_mode == "open":
        mode = HackerOneScopeMode.OPEN
    return HackerOneProgram(
        handle=session.program_handle or scope.program_name or "unknown",
        name=scope.program_name or session.program_handle or "unknown",
        structured_scopes=tuple(records),
        instructions=scope.instructions or "",
        scope_mode=mode,
        open_scope_acknowledged=bool(scope.open_scope_acknowledged),
        open_scope_policy=scope.open_scope_policy or "",
    )
