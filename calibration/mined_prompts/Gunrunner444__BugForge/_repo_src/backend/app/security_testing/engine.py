"""SecurityTestEngine — the only entry point for active security testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4

from app.domain.evidence import Evidence, EvidenceBundle
from app.domain.findings import SecurityFinding
from app.domain.scope import ScopeConstraint
from app.security_testing.approvals import ApprovalKind, HumanApprovalGate, is_ai_operator
from app.security_testing.audit import AuditLog
from app.security_testing.dns import DnsAuthorizer, ResolvedTargetPolicy
from app.security_testing.errors import (
    ApprovalRequiredError,
    AuthorizationDeniedError,
    RestrictedActivityError,
)
from app.security_testing.http_client import GatedHttpClient
from app.security_testing.rate_limit import RateLimiter
from app.security_testing.runner import SecurityToolRunner
from app.security_testing.safety import RestrictedActivity, SafetyController, SafetyLimits
from app.security_testing.scanner_policy import ScannerExecutionPolicy
from app.security_testing.scope_guard import ScopeGuard
from app.security_testing.scope_model import AuthorizationDecision, ProgramScope


class TestingMode(StrEnum):
    __test__ = False
    LIVE = "live"
    LAB = "lab"


@dataclass
class SecurityTestSession:
    project_id: str
    mode: TestingMode
    scope: ProgramScope
    limits: SafetyLimits = field(default_factory=SafetyLimits.conservative)
    dry_run: bool = False
    active_testing_enabled: bool = False
    fuzzing_enabled: bool = False
    id: str = field(default_factory=lambda: uuid4().hex)


class SecurityTestEngine:
    """Compose ScopeGuard → SafetyController → RateLimiter → tool.

    No active network operation may bypass :meth:`authorize` / :meth:`http`.
    """

    def __init__(self, session: SecurityTestSession) -> None:
        if session.mode is TestingMode.LAB and not session.scope.lab_mode:
            session.scope = ProgramScope(
                program_id=session.scope.program_id,
                program_name=session.scope.program_name or "local-lab",
                includes=session.scope.includes,
                excludes=session.scope.excludes,
                instructions=session.scope.instructions,
                allow_active_testing=session.scope.allow_active_testing,
                allowed_methods=session.scope.allowed_methods,
                testing_restrictions=session.scope.testing_restrictions,
                lab_mode=True,
                lab_hosts=session.scope.lab_hosts,
                scope_mode=session.scope.scope_mode,
                open_scope_acknowledged=session.scope.open_scope_acknowledged,
                open_scope_policy=session.scope.open_scope_policy,
            )
        if session.mode is TestingMode.LIVE and session.scope.lab_mode:
            raise RestrictedActivityError("out_of_scope")
        self.session = session
        self.scope_guard = ScopeGuard(session.scope)
        self.safety = SafetyController(session.limits, dry_run=session.dry_run)
        self.rate_limiter = RateLimiter(session.limits)
        self.scanner_policy = ScannerExecutionPolicy().tighten(session.limits)
        self.dns = DnsAuthorizer(
            policy=ResolvedTargetPolicy.lab()
            if session.mode is TestingMode.LAB
            else ResolvedTargetPolicy.live()
        )
        self.approvals = HumanApprovalGate()
        self.audit = AuditLog()
        self.runner = SecurityToolRunner(self)
        self.evidence = EvidenceBundle()
        self.findings: list[SecurityFinding] = []

    @property
    def project_id(self) -> str:
        return self.session.project_id

    @property
    def mode(self) -> TestingMode:
        return self.session.mode

    @classmethod
    def lab(
        cls,
        project_id: str = "lab",
        *,
        hosts: tuple[str, ...] = ("127.0.0.1", "localhost"),
        allow_active_testing: bool = False,
        dry_run: bool = False,
        limits: SafetyLimits | None = None,
    ) -> SecurityTestEngine:
        scope = ProgramScope.from_hosts(
            hosts,
            allow_active_testing=allow_active_testing,
            program_name="local-lab",
            lab_mode=True,
        )
        session = SecurityTestSession(
            project_id=project_id,
            mode=TestingMode.LAB,
            scope=scope,
            limits=limits or SafetyLimits.lab(),
            dry_run=dry_run,
            active_testing_enabled=allow_active_testing,
        )
        return cls(session)

    @classmethod
    def from_constraint(
        cls,
        constraint: ScopeConstraint,
        *,
        project_id: str,
        mode: TestingMode = TestingMode.LIVE,
        dry_run: bool = True,
        limits: SafetyLimits | None = None,
    ) -> SecurityTestEngine:
        scope = ScopeGuard.from_constraint(constraint).scope
        session = SecurityTestSession(
            project_id=project_id,
            mode=mode,
            scope=scope,
            limits=limits or SafetyLimits.conservative(),
            dry_run=dry_run,
            active_testing_enabled=constraint.allow_active_testing,
        )
        return cls(session)

    def enable_active_testing(self, *, operator: str, note: str = "") -> None:
        self.approvals.grant(ApprovalKind.ENABLE_ACTIVE_TESTING, operator=operator, note=note)
        self._apply_active_testing(operator=operator)

    def grant(self, kind: ApprovalKind, *, operator: str, note: str = "") -> None:
        if kind is ApprovalKind.SUBMIT_HACKERONE_REPORT and is_ai_operator(operator):
            raise RestrictedActivityError("hackerone_submission")
        self.approvals.grant(kind, operator=operator, note=note)
        if kind is ApprovalKind.ENABLE_ACTIVE_TESTING:
            self._apply_active_testing(operator=operator)
        if kind is ApprovalKind.ENABLE_FUZZING:
            self.session.fuzzing_enabled = True

    def _apply_active_testing(self, *, operator: str) -> None:
        self.session.active_testing_enabled = True
        self.session.scope = ProgramScope(
            program_id=self.session.scope.program_id,
            program_name=self.session.scope.program_name,
            includes=self.session.scope.includes,
            excludes=self.session.scope.excludes,
            instructions=self.session.scope.instructions,
            allow_active_testing=True,
            allowed_methods=self.session.scope.allowed_methods,
            testing_restrictions=self.session.scope.testing_restrictions,
            lab_mode=self.session.scope.lab_mode,
            lab_hosts=self.session.scope.lab_hosts,
            scope_mode=self.session.scope.scope_mode,
            open_scope_acknowledged=self.session.scope.open_scope_acknowledged,
            open_scope_policy=self.session.scope.open_scope_policy,
        )
        self.scope_guard = ScopeGuard(self.session.scope)
        self.audit.record(
            project=self.project_id,
            target="*",
            scope_decision="active testing enabled",
            tool="safety",
            action="enable_active_testing",
            human_approval=operator,
            result="granted",
        )

    def authorize(
        self,
        target: str,
        *,
        method: str = "GET",
        tool: str = "unknown",
        active: bool = True,
        payload_bytes: int = 0,
        destructive: bool = False,
        require_live_scan_approval: bool = False,
        require_poc_approval: bool = False,
        high_risk: bool = False,
    ) -> AuthorizationDecision:
        if destructive or high_risk:
            # Never allow DoS-shaped work.
            if tool in {"dos", "flood", "stress"}:
                self.safety.forbid(RestrictedActivity.DENIAL_OF_SERVICE)
        if require_live_scan_approval and self.session.mode is TestingMode.LIVE:
            if not self.approvals.is_granted(ApprovalKind.START_LIVE_SCAN):
                raise ApprovalRequiredError(ApprovalKind.START_LIVE_SCAN.value)
        if require_poc_approval and not self.approvals.is_granted(ApprovalKind.SEND_POC_REQUEST):
            raise ApprovalRequiredError(ApprovalKind.SEND_POC_REQUEST.value)
        if high_risk and not self.approvals.is_granted(ApprovalKind.HIGH_RISK_SCANNER):
            raise ApprovalRequiredError(ApprovalKind.HIGH_RISK_SCANNER.value)
        if active and self.session.mode is TestingMode.LIVE:
            if not self.approvals.is_granted(ApprovalKind.ENABLE_ACTIVE_TESTING):
                # Still consult ScopeGuard so the reason is explicit, then deny.
                scope_decision = self.scope_guard.authorize(
                    target, method=method, tool=tool, active=active, dry_run=self.safety.dry_run
                )
                if not scope_decision.allowed:
                    self._log_decision(scope_decision)
                    return scope_decision
                denied = AuthorizationDecision(
                    allowed=False,
                    reason="NO ACTIVE-TESTING PERMISSION = DENY (human approval required)",
                    target_original=target,
                    method=method.upper(),
                    tool=tool,
                    matched_rule=scope_decision.matched_rule,
                    program=scope_decision.program,
                    dry_run=self.safety.dry_run,
                    approval_required=True,
                    approval_state="missing",
                )
                self._log_decision(denied)
                return denied
        if tool == "fuzzer" and not (
            self.session.fuzzing_enabled or self.approvals.is_granted(ApprovalKind.ENABLE_FUZZING)
        ):
            if active:
                denied = AuthorizationDecision(
                    allowed=False,
                    reason="Fuzzing requires explicit human approval",
                    target_original=target,
                    method=method.upper(),
                    tool=tool,
                    dry_run=self.safety.dry_run,
                    approval_required=True,
                    approval_state="missing",
                )
                self._log_decision(denied)
                return denied

        scope_decision = self.scope_guard.authorize(
            target, method=method, tool=tool, active=active, dry_run=self.safety.dry_run
        )
        if not scope_decision.allowed:
            self._log_decision(scope_decision)
            return scope_decision
        if self.session.mode is TestingMode.LIVE:
            dns_decision = self.dns.authorize(
                target, lab_mode=False, lab_hosts=self.session.scope.lab_hosts
            )
            if not dns_decision.allowed:
                denied = AuthorizationDecision(
                    allowed=False,
                    reason=dns_decision.reason,
                    target_original=target,
                    method=method.upper(),
                    tool=tool,
                    matched_rule=scope_decision.matched_rule,
                    program=scope_decision.program,
                    dry_run=self.safety.dry_run,
                )
                self._log_decision(denied)
                return denied
        safety = self.safety.check_request(
            target=target,
            method=method,
            tool=tool,
            payload_bytes=payload_bytes,
            destructive=destructive,
            scope=scope_decision,
        )
        if not safety.allowed:
            denied = AuthorizationDecision(
                allowed=False,
                reason=safety.reason,
                target_original=target,
                method=method.upper(),
                tool=tool,
                matched_rule=scope_decision.matched_rule,
                program=scope_decision.program,
                dry_run=self.safety.dry_run,
            )
            self._log_decision(denied)
            return denied
        rate = self.rate_limiter.check()
        if not rate.allowed and not self.safety.dry_run:
            denied = AuthorizationDecision(
                allowed=False,
                reason=rate.reason,
                target_original=target,
                method=method.upper(),
                tool=tool,
                matched_rule=scope_decision.matched_rule,
                program=scope_decision.program,
                rate_limit=rate.reason,
            )
            self._log_decision(denied)
            return denied
        allowed = AuthorizationDecision(
            allowed=True,
            reason=safety.reason if safety.dry_run else scope_decision.reason,
            target_original=target,
            method=method.upper(),
            tool=tool,
            matched_rule=scope_decision.matched_rule,
            program=scope_decision.program,
            dry_run=safety.dry_run,
            approval_state="granted"
            if self.approvals.is_granted(ApprovalKind.ENABLE_ACTIVE_TESTING)
            else "not_required",
            rate_limit=rate.reason,
        )
        self._log_decision(allowed)
        return allowed

    def http(self, tool: str) -> GatedHttpClient:
        return GatedHttpClient(self, tool=tool)

    def add_evidence(self, items: list[Evidence] | EvidenceBundle) -> None:
        extra = items.items if isinstance(items, EvidenceBundle) else tuple(items)
        self.evidence = self.evidence.extend(extra)

    def snapshot(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "mode": self.mode.value,
            "program": self.session.scope.program_name,
            "lab_mode": self.session.scope.lab_mode,
            "active_testing": self.session.active_testing_enabled,
            "fuzzing_enabled": self.session.fuzzing_enabled,
            "dry_run": self.safety.dry_run,
            "request_limit": self.safety.limits.max_requests,
            "requests_used": self.safety.request_count(),
            "rate_limit_rps": self.safety.limits.requests_per_second,
            "approvals": self.approvals.snapshot(),
            "audit_entries": len(self.audit),
            "finding_counts": {
                status: sum(1 for item in self.findings if item.status.value == status)
                for status in {item.status.value for item in self.findings}
            },
        }

    def _log_decision(self, decision: AuthorizationDecision) -> None:
        self.audit.record(
            project=self.project_id,
            target=decision.target_original,
            scope_decision=decision.reason,
            tool=decision.tool,
            action="authorize",
            rate_limit_decision=decision.rate_limit or "",
            result="allowed" if decision.allowed else "denied",
            human_approval=decision.approval_state,
        )


def require_allowed(decision: AuthorizationDecision) -> None:
    if not decision.allowed:
        raise AuthorizationDeniedError(
            decision.reason, target=decision.target_original, tool=decision.tool
        )
