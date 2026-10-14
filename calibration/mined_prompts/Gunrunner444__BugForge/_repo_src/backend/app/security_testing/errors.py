"""Errors for authorized security testing. These are infrastructure, not findings."""

from __future__ import annotations

from app.plugins.errors import AdapterError


class SecurityTestingError(AdapterError):
    """Base error for the authorized-testing stack."""


class AuthorizationDeniedError(SecurityTestingError):
    """Raised when ScopeGuard or SafetyController denies an operation."""

    def __init__(self, reason: str, *, target: str = "", tool: str = "") -> None:
        self.reason = reason
        self.target = target
        self.tool = tool
        super().__init__(reason)


class SafetyLimitExceededError(SecurityTestingError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ApprovalRequiredError(SecurityTestingError):
    def __init__(self, action: str, *, detail: str | None = None) -> None:
        self.action = action
        message = f"Human approval is required before {action}."
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


class RestrictedActivityError(SecurityTestingError):
    """Always-denied activity (DoS, social engineering, report auto-submit, ...)."""

    def __init__(self, activity: str) -> None:
        self.activity = activity
        super().__init__(f"Activity {activity!r} is forbidden. BugForge does not perform it.")


class ToolExecutionError(SecurityTestingError):
    """Scanner/browser/proxy infrastructure failure. Must not become a finding."""

    def __init__(self, tool: str, state: str, *, detail: str = "") -> None:
        self.tool = tool
        self.state = state
        self.detail = detail
        super().__init__(f"{tool} execution {state}: {detail}".strip())
