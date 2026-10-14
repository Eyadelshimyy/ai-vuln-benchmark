"""Authorized security testing — ScopeGuard, SafetyController, and tool adapters."""

from __future__ import annotations

from app.security_testing.approvals import ApprovalKind, HumanApprovalGate, is_ai_operator
from app.security_testing.audit import AuditLog
from app.security_testing.engine import SecurityTestEngine, SecurityTestSession, TestingMode
from app.security_testing.errors import (
    ApprovalRequiredError,
    AuthorizationDeniedError,
    RestrictedActivityError,
    SafetyLimitExceededError,
    ToolExecutionError,
)
from app.security_testing.operator_auth import OperatorSession, require_operator
from app.security_testing.rate_limit import RateLimiter
from app.security_testing.safety import SafetyController, SafetyLimits
from app.security_testing.scope_guard import ScopeGuard
from app.security_testing.scope_model import AuthorizationDecision, ProgramScope, ScopeRule
from app.security_testing.target import NormalizedTarget, TargetNormalizer

__all__ = [
    "ApprovalKind",
    "ApprovalRequiredError",
    "AuditLog",
    "AuthorizationDecision",
    "AuthorizationDeniedError",
    "HumanApprovalGate",
    "NormalizedTarget",
    "OperatorSession",
    "ProgramScope",
    "RateLimiter",
    "RestrictedActivityError",
    "SafetyController",
    "SafetyLimitExceededError",
    "SafetyLimits",
    "ScopeGuard",
    "ScopeRule",
    "SecurityTestEngine",
    "SecurityTestSession",
    "TargetNormalizer",
    "TestingMode",
    "ToolExecutionError",
    "is_ai_operator",
    "require_operator",
]
