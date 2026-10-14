"""Public tool capability views. Installed ≠ enabled ≠ approved ≠ authorized."""

from __future__ import annotations

from typing import Any

from app.security_agent.states import (
    ToolApprovalStatus,
    ToolAuthorization,
    ToolAvailability,
    ToolCapability,
    ToolEnablement,
)
from app.security_testing.approvals import ApprovalKind


def describe_tools(agent: Any) -> list[dict[str, Any]]:
    session = agent.session
    out: list[dict[str, Any]] = []
    for spec in (agent.tools.spec(name) for name in agent.tools.known()):
        available = spec.capability is not ToolCapability.UNAVAILABLE
        enabled = not agent.tools.is_disabled(spec.name)
        approved = True
        if spec.requires_human_approval and spec.approval_kind:
            try:
                kind = ApprovalKind(spec.approval_kind)
            except ValueError:
                kind = None
            approved = bool(kind and session.engine.approvals.is_granted(kind))
        auth = ToolAuthorization.BLOCKED
        if available and enabled:
            try:
                auth = agent._authorize_target(session.target, tool=spec.name, spec=spec)
            except Exception:
                auth = ToolAuthorization.BLOCKED
        out.append(
            {
                "name": spec.name,
                "capability": spec.capability.value,
                "availability": (
                    ToolAvailability.AVAILABLE.value
                    if available
                    else ToolAvailability.UNAVAILABLE.value
                ),
                "enablement": (
                    ToolEnablement.ENABLED.value if enabled else ToolEnablement.DISABLED.value
                ),
                "approval": (
                    ToolApprovalStatus.APPROVED.value
                    if approved
                    else ToolApprovalStatus.NOT_APPROVED.value
                ),
                "authorization": auth.value if isinstance(auth, ToolAuthorization) else str(auth),
                "installed_is_not_enabled": True,
                "enabled_is_not_approved": True,
                "approved_is_not_authorized": True,
                "authorized_is_not_success": True,
                "risk_level": spec.risk_level.value,
                "requires_human_approval": spec.requires_human_approval,
                "approval_kind": spec.approval_kind,
            }
        )
    return out
