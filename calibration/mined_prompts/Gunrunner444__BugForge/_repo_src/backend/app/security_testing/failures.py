"""Tool/infrastructure outcomes. These must never be stored as vulnerabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolExecutionState(StrEnum):
    """Canonical lifecycle for every security tool.

    Execution errors are infrastructure outcomes, not findings.
    """

    NOT_REQUESTED = "not_requested"
    AUTHORIZED = "authorized"
    PLANNED = "planned"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    RESULTS_AVAILABLE = "results_available"
    RESULTS_INGESTED = "results_ingested"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TIMEOUT = "timeout"
    NETWORK_FAILURE = "network_failure"
    INVALID_SCOPE = "invalid_scope"
    SAFETY_BLOCKED = "safety_blocked"
    EXECUTION_ERROR = "execution_error"
    FAILED = "failed"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMITED = "rate_limited"
    MALFORMED_INPUT = "malformed_input"
    DENIED = "denied"
    DRY_RUN = "dry_run"
    APPROVAL_REQUIRED = "approval_required"
    OK = "ok"


_EXECUTION_SUCCESS_STATES = frozenset(
    {
        ToolExecutionState.OK,
        ToolExecutionState.COMPLETED,
        ToolExecutionState.RESULTS_AVAILABLE,
        ToolExecutionState.RESULTS_INGESTED,
    }
)

_RESULT_AVAILABLE_STATES = frozenset(
    {
        ToolExecutionState.RESULTS_AVAILABLE,
        ToolExecutionState.RESULTS_INGESTED,
        ToolExecutionState.COMPLETED,
        ToolExecutionState.OK,
    }
)

_PLANNING_STATES = frozenset(
    {
        ToolExecutionState.NOT_REQUESTED,
        ToolExecutionState.PLANNED,
        ToolExecutionState.DRY_RUN,
        ToolExecutionState.WAITING_FOR_APPROVAL,
    }
)

_AUTHORIZATION_STATES = frozenset(
    {
        ToolExecutionState.AUTHORIZED,
        ToolExecutionState.DENIED,
        ToolExecutionState.INVALID_SCOPE,
        ToolExecutionState.SAFETY_BLOCKED,
        ToolExecutionState.APPROVAL_REQUIRED,
        ToolExecutionState.WAITING_FOR_APPROVAL,
    }
)

_TERMINAL_FAILURE_STATES = frozenset(
    {
        ToolExecutionState.TOOL_UNAVAILABLE,
        ToolExecutionState.TIMEOUT,
        ToolExecutionState.NETWORK_FAILURE,
        ToolExecutionState.EXECUTION_ERROR,
        ToolExecutionState.FAILED,
        ToolExecutionState.AUTHENTICATION_FAILURE,
        ToolExecutionState.RATE_LIMITED,
        ToolExecutionState.MALFORMED_INPUT,
        ToolExecutionState.DENIED,
        ToolExecutionState.INVALID_SCOPE,
        ToolExecutionState.SAFETY_BLOCKED,
    }
)


@dataclass(frozen=True)
class ToolExecutionResult:
    tool: str
    state: ToolExecutionState
    detail: str = ""
    evidence_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_finding(self) -> bool:
        return False

    @property
    def execution_success(self) -> bool:
        """True only when the tool actually completed or produced ingestible results."""
        return self.state in _EXECUTION_SUCCESS_STATES

    @property
    def result_available(self) -> bool:
        return self.state in _RESULT_AVAILABLE_STATES

    @property
    def planning_state(self) -> bool:
        return self.state in _PLANNING_STATES

    @property
    def authorization_state(self) -> bool:
        return self.state in _AUTHORIZATION_STATES

    @property
    def terminal_failure(self) -> bool:
        return self.state in _TERMINAL_FAILURE_STATES

    @property
    def ok(self) -> bool:
        """Backward-compatible alias for actual execution success.

        Planned, authorized, dry-run, and not-requested states are not success.
        """
        return self.execution_success
