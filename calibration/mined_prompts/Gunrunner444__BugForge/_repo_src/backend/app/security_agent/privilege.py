"""Privilege snapshots. Restored sessions never gain broader authority."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from app.adapters.hackerone.models import HackerOneProgram
from app.core.config import get_settings
from app.security_agent.states import ResearchMode
from app.security_testing.approvals import (
    ApprovalKind,
    ApprovalRecord,
    ApprovalState,
    HumanApprovalGate,
)
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.safety import SafetyLimits
from app.security_testing.scope_model import Eligibility, ProgramScope, ScopeMode, ScopeRule
from app.security_testing.target import AssetType


@dataclass
class PrivilegeSnapshot:
    mode: str
    program_handle: str
    scope_hash: str
    scope_includes: tuple[str, ...] = ()
    scope_excludes: tuple[str, ...] = ()
    active_testing: bool = False
    fuzzing: bool = False
    dry_run: bool = True
    tool_enablement: dict[str, bool] = field(default_factory=dict)
    safety_limits: dict[str, Any] = field(default_factory=dict)
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    captured_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "program_handle": self.program_handle,
            "scope_hash": self.scope_hash,
            "scope_includes": list(self.scope_includes),
            "scope_excludes": list(self.scope_excludes),
            "active_testing": self.active_testing,
            "fuzzing": self.fuzzing,
            "dry_run": self.dry_run,
            "tool_enablement": dict(self.tool_enablement),
            "safety_limits": dict(self.safety_limits),
            "approvals": dict(self.approvals),
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> PrivilegeSnapshot | None:
        if not payload:
            return None
        return cls(
            mode=str(payload.get("mode") or ""),
            program_handle=str(payload.get("program_handle") or ""),
            scope_hash=str(payload.get("scope_hash") or ""),
            scope_includes=tuple(str(item) for item in payload.get("scope_includes") or ()),
            scope_excludes=tuple(str(item) for item in payload.get("scope_excludes") or ()),
            active_testing=bool(payload.get("active_testing")),
            fuzzing=bool(payload.get("fuzzing")),
            dry_run=bool(payload.get("dry_run", True)),
            tool_enablement=dict(payload.get("tool_enablement") or {}),
            safety_limits=dict(payload.get("safety_limits") or {}),
            approvals=dict(payload.get("approvals") or {}),
            captured_at=str(payload.get("captured_at") or ""),
        )


def capture_privileges(
    engine: SecurityTestEngine,
    *,
    mode: ResearchMode,
    program_handle: str,
    disabled_tools: set[str],
) -> PrivilegeSnapshot:
    scope = engine.session.scope
    includes = tuple(rule.identifier for rule in scope.includes)
    excludes = tuple(rule.identifier for rule in scope.excludes)
    approvals: dict[str, dict[str, Any]] = {}
    for kind in ApprovalKind:
        record = engine.approvals.get_record(kind)
        if record is None:
            approvals[kind.value] = {"state": ApprovalState.MISSING.value}
            continue
        approvals[kind.value] = {
            "state": record.state.value,
            "kind": kind.value,
            "operator": record.operator,
            "note": record.note,
            "granted_at": record.granted_at.isoformat(),
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "id": record.id,
            "scope_hash": scope_fingerprint(scope),
            "approval_hash": _approval_hash(kind.value, record.operator, scope_fingerprint(scope)),
        }
    return PrivilegeSnapshot(
        mode=mode.value,
        program_handle=program_handle,
        scope_hash=scope_fingerprint(scope),
        scope_includes=includes,
        scope_excludes=excludes,
        active_testing=engine.session.active_testing_enabled,
        fuzzing=engine.session.fuzzing_enabled,
        dry_run=engine.safety.dry_run,
        tool_enablement={name: False for name in disabled_tools},
        safety_limits={
            "max_requests": engine.safety.limits.max_requests,
            "requests_per_second": engine.safety.limits.requests_per_second,
            "max_scan_duration_seconds": engine.safety.limits.max_scan_duration_seconds,
            "timeout_seconds": engine.safety.limits.timeout_seconds,
            "max_payload_count": engine.safety.limits.max_payload_count,
            "max_payload_bytes": engine.safety.limits.max_payload_bytes,
        },
        approvals=approvals,
        captured_at=datetime.now(UTC).isoformat(),
    )


def scope_fingerprint(scope: ProgramScope) -> str:
    """Canonical hash of every authorization-relevant scope field.

    Ordering cannot change the digest: rules are sorted and JSON is
    serialized with sorted keys.
    """
    payload = {
        "program_id": scope.program_id or "",
        "program_name": scope.program_name or "",
        "scope_mode": getattr(scope.scope_mode, "value", str(scope.scope_mode)),
        "includes": [_rule_fingerprint(rule) for rule in _sorted_rules(scope.includes)],
        "excludes": [_rule_fingerprint(rule) for rule in _sorted_rules(scope.excludes)],
        "allowed_http_methods": sorted(method.upper() for method in scope.allowed_methods),
        "allow_active_testing": bool(scope.allow_active_testing),
        "testing_restrictions": sorted(
            getattr(item, "value", str(item)) for item in scope.testing_restrictions
        ),
        "instructions": scope.instructions or "",
        "open_scope_acknowledged": bool(scope.open_scope_acknowledged),
        "open_scope_policy": scope.open_scope_policy or "",
        "lab_mode": bool(scope.lab_mode),
        "lab_hosts": sorted(scope.lab_hosts),
        "network_policy": "lab" if scope.lab_mode else "live",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(raw.encode("utf-8")).hexdigest()


def _sorted_rules(rules: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        sorted(
            rules,
            key=lambda rule: (
                rule.identifier,
                str(rule.structured_scope_id or ""),
                str(getattr(rule.asset_type, "value", rule.asset_type)),
            ),
        )
    )


def _rule_fingerprint(rule: Any) -> dict[str, Any]:
    return {
        "identifier": rule.identifier,
        "asset_type": getattr(rule.asset_type, "value", str(rule.asset_type)),
        "structured_scope_id": rule.structured_scope_id or "",
        "eligible_for_submission": bool(rule.eligible_for_submission),
        "eligible_for_bounty": bool(rule.eligible_for_bounty),
        "eligibility": getattr(rule.eligible, "value", str(rule.eligible)),
        "allowed_methods": sorted(method.upper() for method in rule.allowed_methods),
        "allow_active_testing": bool(rule.allow_active_testing),
        "restriction": getattr(rule.restriction, "value", str(rule.restriction)),
        "path_prefix": rule.path_prefix or "",
        "instructions": rule.instructions or "",
        "is_exclusion": bool(rule.is_exclusion),
        "max_severity": rule.max_severity or "",
        "program_id": rule.program_id or "",
        "reference": rule.reference or "",
        "exclusions": sorted(rule.exclusions),
    }


def program_scope_from_hackerone(program: HackerOneProgram) -> ProgramScope:
    """Bind a research session to one program's persisted structured scope."""
    includes: list[ScopeRule] = []
    for record in program.structured_scopes:
        identifier = (record.asset_identifier or "").strip()
        if not identifier:
            continue
        asset_type = (
            record.asset_type if isinstance(record.asset_type, AssetType) else AssetType.UNSUPPORTED
        )
        if asset_type is AssetType.OTHER:
            asset_type = AssetType.UNSUPPORTED
        includes.append(
            ScopeRule(
                identifier=identifier,
                asset_type=asset_type,
                eligible=Eligibility.ELIGIBLE
                if record.eligible_for_submission
                else Eligibility.INELIGIBLE,
                instructions=record.instruction,
                allow_active_testing=False,
                structured_scope_id=record.id,
                eligible_for_submission=record.eligible_for_submission,
                eligible_for_bounty=record.eligible_for_bounty,
                reference=record.reference,
            )
        )
    return ProgramScope(
        program_id=program.program_id or program.handle,
        program_name=program.handle,
        includes=tuple(includes),
        excludes=(),
        instructions=program.instructions,
        allow_active_testing=False,
        lab_mode=False,
        scope_mode=ScopeMode(getattr(program.scope_mode, "value", program.scope_mode) or "closed"),
        open_scope_acknowledged=program.open_scope_acknowledged,
        open_scope_policy=program.open_scope_policy,
    )


def intersect_privileges(
    persisted: PrivilegeSnapshot,
    *,
    current_scope: ProgramScope,
    current_program_handle: str,
) -> PrivilegeSnapshot:
    """Never restore stale permissions after scope/approval changes."""
    current_hash = scope_fingerprint(current_scope)
    current_includes = {rule.identifier for rule in current_scope.includes}
    persisted_includes = set(persisted.scope_includes)
    # If current authorization is weaker or the program changed, downgrade.
    downgrade_scope = current_hash != persisted.scope_hash or current_includes != persisted_includes
    includes = tuple(sorted(current_includes)) if downgrade_scope else persisted.scope_includes
    # Active testing is never restored from a stale snapshot when current program
    # does not currently allow it. Live sessions always start disabled unless
    # a still-valid approval exists.
    ttl_hours = get_settings().security_agent_approval_ttl_hours
    approvals: dict[str, dict[str, Any]] = {}
    for kind, payload in persisted.approvals.items():
        if not isinstance(payload, dict):
            continue
        if (
            _approval_still_valid(
                payload,
                ttl_hours=ttl_hours,
                current_scope_hash=current_hash,
                current_program_handle=current_program_handle,
                persisted_program_handle=persisted.program_handle,
            )
            and not downgrade_scope
        ):
            approvals[kind] = dict(payload)
        else:
            approvals[kind] = {
                "state": ApprovalState.MISSING.value,
                "reason": "stale_or_invalidated",
            }
    active_testing = bool(
        persisted.active_testing
        and not downgrade_scope
        and _approval_still_valid(
            persisted.approvals.get(ApprovalKind.ENABLE_ACTIVE_TESTING.value) or {},
            ttl_hours=ttl_hours,
            current_scope_hash=current_hash,
            current_program_handle=current_program_handle,
            persisted_program_handle=persisted.program_handle,
        )
        and current_program_handle == persisted.program_handle
    )
    fuzzing = bool(
        persisted.fuzzing
        and not downgrade_scope
        and _approval_still_valid(
            persisted.approvals.get(ApprovalKind.ENABLE_FUZZING.value) or {},
            ttl_hours=ttl_hours,
            current_scope_hash=current_hash,
            current_program_handle=current_program_handle,
            persisted_program_handle=persisted.program_handle,
        )
    )
    return PrivilegeSnapshot(
        mode=persisted.mode,
        program_handle=current_program_handle,
        scope_hash=current_hash,
        scope_includes=includes,
        scope_excludes=tuple(rule.identifier for rule in current_scope.excludes),
        active_testing=active_testing,
        fuzzing=fuzzing,
        dry_run=True if downgrade_scope else persisted.dry_run,
        tool_enablement=dict(persisted.tool_enablement),
        safety_limits=dict(persisted.safety_limits),
        approvals=approvals,
        captured_at=datetime.now(UTC).isoformat(),
    )


def expire_stale_privileges(snapshot: PrivilegeSnapshot) -> PrivilegeSnapshot:
    """Drop expired approvals. Never revive stale active-testing or fuzzing."""
    ttl_hours = get_settings().security_agent_approval_ttl_hours
    approvals: dict[str, dict[str, Any]] = {}
    for kind, payload in snapshot.approvals.items():
        if not isinstance(payload, dict):
            continue
        if _approval_still_valid(
            payload, ttl_hours=ttl_hours, current_scope_hash=snapshot.scope_hash
        ):
            approvals[kind] = dict(payload)
        else:
            approvals[kind] = {"state": ApprovalState.MISSING.value, "reason": "expired"}
    active_testing = bool(
        snapshot.active_testing
        and _approval_still_valid(
            snapshot.approvals.get(ApprovalKind.ENABLE_ACTIVE_TESTING.value) or {},
            ttl_hours=ttl_hours,
        )
    )
    fuzzing = bool(
        snapshot.fuzzing
        and _approval_still_valid(
            snapshot.approvals.get(ApprovalKind.ENABLE_FUZZING.value) or {},
            ttl_hours=ttl_hours,
        )
    )
    return PrivilegeSnapshot(
        mode=snapshot.mode,
        program_handle=snapshot.program_handle,
        scope_hash=snapshot.scope_hash,
        scope_includes=snapshot.scope_includes,
        scope_excludes=snapshot.scope_excludes,
        active_testing=active_testing,
        fuzzing=fuzzing,
        dry_run=snapshot.dry_run,
        tool_enablement=dict(snapshot.tool_enablement),
        safety_limits=dict(snapshot.safety_limits),
        approvals=approvals,
        captured_at=datetime.now(UTC).isoformat(),
    )


def apply_privilege_snapshot(engine: SecurityTestEngine, snapshot: PrivilegeSnapshot) -> None:
    engine.session.active_testing_enabled = snapshot.active_testing
    engine.session.fuzzing_enabled = snapshot.fuzzing
    engine.session.dry_run = snapshot.dry_run
    engine.safety.dry_run = snapshot.dry_run
    if snapshot.active_testing:
        engine.session.scope = ProgramScope(
            program_id=engine.session.scope.program_id,
            program_name=engine.session.scope.program_name,
            includes=engine.session.scope.includes,
            excludes=engine.session.scope.excludes,
            instructions=engine.session.scope.instructions,
            allow_active_testing=True,
            allowed_methods=engine.session.scope.allowed_methods,
            testing_restrictions=engine.session.scope.testing_restrictions,
            lab_mode=engine.session.scope.lab_mode,
            lab_hosts=engine.session.scope.lab_hosts,
            scope_mode=engine.session.scope.scope_mode,
            open_scope_acknowledged=engine.session.scope.open_scope_acknowledged,
            open_scope_policy=engine.session.scope.open_scope_policy,
        )
        from app.security_testing.scope_guard import ScopeGuard

        engine.scope_guard = ScopeGuard(engine.session.scope)
    _restore_approvals(engine.approvals, snapshot.approvals)
    limits = snapshot.safety_limits
    if limits:
        engine.safety.limits = SafetyLimits(
            max_requests=int(limits.get("max_requests", engine.safety.limits.max_requests)),
            requests_per_second=float(
                limits.get("requests_per_second", engine.safety.limits.requests_per_second)
            ),
            timeout_seconds=float(
                limits.get("timeout_seconds", engine.safety.limits.timeout_seconds)
            ),
            max_scan_duration_seconds=float(
                limits.get(
                    "max_scan_duration_seconds", engine.safety.limits.max_scan_duration_seconds
                )
            ),
            max_payload_count=int(
                limits.get("max_payload_count", engine.safety.limits.max_payload_count)
            ),
            max_payload_bytes=int(
                limits.get("max_payload_bytes", engine.safety.limits.max_payload_bytes)
            ),
            allowed_http_methods=engine.safety.limits.allowed_http_methods,
        )


def _approval_still_valid(
    payload: dict[str, Any],
    *,
    ttl_hours: int,
    current_scope_hash: str = "",
    current_program_handle: str = "",
    persisted_program_handle: str = "",
) -> bool:
    if str(payload.get("state") or "") != ApprovalState.GRANTED.value:
        return False
    if current_program_handle and persisted_program_handle:
        if current_program_handle != persisted_program_handle:
            return False
    stored_scope = str(payload.get("scope_hash") or "")
    if current_scope_hash and stored_scope and stored_scope != current_scope_hash:
        return False
    stored_hash = str(payload.get("approval_hash") or "")
    if stored_hash and current_scope_hash:
        expected = _approval_hash(
            str(payload.get("kind") or ""),
            str(payload.get("operator") or ""),
            current_scope_hash,
        )
        # If kind wasn't stored on the payload, fall back to scope_hash match above.
        if payload.get("kind") and stored_hash != expected:
            return False
    expires = payload.get("expires_at")
    granted = payload.get("granted_at")
    now = datetime.now(UTC)
    if isinstance(expires, str) and expires:
        try:
            exp = datetime.fromisoformat(expires)
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=UTC)
            if now >= exp:
                return False
        except ValueError:
            return False
    elif isinstance(granted, str) and granted and ttl_hours > 0:
        try:
            at = datetime.fromisoformat(granted)
            if at.tzinfo is None:
                at = at.replace(tzinfo=UTC)
            if now >= at + timedelta(hours=ttl_hours):
                return False
        except ValueError:
            return False
    return True


def _approval_hash(kind: str, operator: str, scope_hash: str) -> str:
    raw = f"{kind}:{operator}:{scope_hash}"
    return sha256(raw.encode("utf-8")).hexdigest()


def _restore_approvals(gate: HumanApprovalGate, payload: dict[str, dict[str, Any]]) -> None:
    gate.clear_records()
    for raw_kind, item in payload.items():
        try:
            kind = ApprovalKind(raw_kind)
        except ValueError:
            continue
        if str(item.get("state") or "") != ApprovalState.GRANTED.value:
            continue
        granted_at = datetime.now(UTC)
        raw_granted = item.get("granted_at")
        if isinstance(raw_granted, str) and raw_granted:
            try:
                granted_at = datetime.fromisoformat(raw_granted)
            except ValueError:
                granted_at = datetime.now(UTC)
        expires_at = None
        raw_exp = item.get("expires_at")
        if isinstance(raw_exp, str) and raw_exp:
            try:
                expires_at = datetime.fromisoformat(raw_exp)
            except ValueError:
                expires_at = None
        gate.load_record(
            ApprovalRecord(
                kind=kind,
                state=ApprovalState.GRANTED,
                operator=str(item.get("operator") or "restored"),
                note=str(item.get("note") or "restored"),
                id=str(item.get("id") or ""),
                granted_at=granted_at,
                expires_at=expires_at,
            )
        )
