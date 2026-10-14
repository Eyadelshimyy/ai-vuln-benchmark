"""Session checkpoints. Live resume re-checks current scope and permissions.

A checkpoint never restores stale authority. Resume intersects the stored
privilege snapshot with the *current* program, scope, target, approvals,
safety limits, disabled tools, and project association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.security_agent.agent import ResearchSession
from app.security_agent.privilege import (
    PrivilegeSnapshot,
    apply_privilege_snapshot,
    capture_privileges,
    intersect_privileges,
    scope_fingerprint,
)
from app.security_testing.approvals import ApprovalKind, ApprovalState
from app.security_testing.errors import RestrictedActivityError
from app.security_testing.safety import SafetyLimits


@dataclass
class ResearchCheckpoint:
    id: str
    label: str
    snapshot: dict[str, Any]

    @classmethod
    def capture(cls, session: ResearchSession, *, label: str = "") -> ResearchCheckpoint:
        privilege = capture_privileges(
            session.engine,
            mode=session.mode,
            program_handle=session.program_handle,
            disabled_tools=session.disabled_tools,
        )
        return cls(
            id=uuid4().hex,
            label=label or session.state.value,
            snapshot={
                "session": session.snapshot(),
                "model": {
                    "provider": session.provider.provider_name,
                    "model": session.provider.model_name,
                    "thinking": session.thinking_enabled,
                },
                "privilege": privilege.as_dict(),
                "budget": session.budget.snapshot(),
                "hypotheses": [item.snapshot() for item in session.hypotheses],
                "graph": session.graph.snapshot(),
                "tool_history": [
                    item.snapshot() if hasattr(item, "snapshot") else item
                    for item in getattr(session, "tool_call_records", session.tool_history)
                ],
                "plan": session.plan.snapshot() if session.plan else None,
                "findings": [
                    {"id": str(item.id), "status": item.status.value, "title": item.title}
                    for item in session.findings
                ],
                "project_id": session.project_id,
                "target": session.target,
                "disabled_tools": sorted(session.disabled_tools),
            },
        )

    def verify_live(self, session: ResearchSession) -> PrivilegeSnapshot:
        """Validate CURRENT authorization. Never trust the checkpoint snapshot alone."""
        stored_raw = self.snapshot.get("privilege") or {}
        stored = PrivilegeSnapshot.from_dict(stored_raw) if stored_raw else None
        stored_project = str(self.snapshot.get("project_id") or "")
        if stored_project and stored_project != session.project_id:
            raise RestrictedActivityError("checkpoint_project_mismatch")
        stored_target = str(self.snapshot.get("target") or "")
        if stored_target and session.target and stored_target != session.target:
            # Target change requires a fresh authorization decision; do not
            # restore the old active-testing grant automatically.
            pass
        current_hash = scope_fingerprint(session.engine.session.scope)
        stored_hash = str((stored_raw or {}).get("scope_hash") or "")
        if session.mode.value == "live_hackerone":
            if not session.program_handle:
                raise RestrictedActivityError("checkpoint_program_missing")
            if stored and stored.program_handle and stored.program_handle != session.program_handle:
                raise RestrictedActivityError("checkpoint_program_changed")
            if not session.engine.session.scope.includes:
                raise RestrictedActivityError("checkpoint_current_scope_missing")
            if stored_hash and stored_hash != current_hash:
                # Current scope is authoritative; intersect will downgrade.
                pass
        if stored:
            snapshot = intersect_privileges(
                stored,
                current_scope=session.engine.session.scope,
                current_program_handle=session.program_handle,
            )
        else:
            snapshot = capture_privileges(
                session.engine,
                mode=session.mode,
                program_handle=session.program_handle,
                disabled_tools=session.disabled_tools,
            )
        # Disabled tools: union — never re-enable a currently disabled tool.
        current_disabled = set(session.disabled_tools)
        stored_disabled = {
            name for name, enabled in (snapshot.tool_enablement or {}).items() if enabled is False
        }
        snapshot.tool_enablement = {
            name: False for name in sorted(current_disabled | stored_disabled)
        }
        snapshot.safety_limits = _weaker_limits(
            snapshot.safety_limits, session.engine.safety.limits
        )
        if (
            session.engine.session.mode.value == "live"
            and session.engine.session.active_testing_enabled
        ):
            approvals = snapshot.approvals or {}
            active = approvals.get(ApprovalKind.ENABLE_ACTIVE_TESTING.value) or {}
            if str(active.get("state") or "") != ApprovalState.GRANTED.value:
                raise RestrictedActivityError("checkpoint_active_testing_not_granted")
        return snapshot

    def resume(self, session: ResearchSession) -> PrivilegeSnapshot:
        snapshot = self.verify_live(session)
        apply_privilege_snapshot(session.engine, snapshot)
        session.disabled_tools |= {
            name for name, enabled in snapshot.tool_enablement.items() if enabled is False
        }
        session.privilege = snapshot
        return snapshot


def _weaker_limits(stored: dict[str, Any], current: SafetyLimits) -> dict[str, Any]:
    def _min_num(key: str, current_value: float | int) -> float | int:
        raw = stored.get(key, current_value)
        try:
            stored_value = type(current_value)(raw)
        except (TypeError, ValueError):
            stored_value = current_value
        return min(stored_value, current_value)

    return {
        "max_requests": int(_min_num("max_requests", current.max_requests)),
        "requests_per_second": float(_min_num("requests_per_second", current.requests_per_second)),
        "timeout_seconds": float(_min_num("timeout_seconds", current.timeout_seconds)),
        "max_scan_duration_seconds": float(
            _min_num("max_scan_duration_seconds", current.max_scan_duration_seconds)
        ),
        "max_payload_count": int(_min_num("max_payload_count", current.max_payload_count)),
        "max_payload_bytes": int(_min_num("max_payload_bytes", current.max_payload_bytes)),
    }
