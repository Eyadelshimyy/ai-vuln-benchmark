"""SecurityToolRunner — every tool invocation must pass the authorization chain."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from app.security_testing.errors import AuthorizationDeniedError
from app.security_testing.failures import ToolExecutionResult, ToolExecutionState
from app.security_testing.sanitization import wrap_untrusted

if TYPE_CHECKING:
    from app.security_testing.engine import SecurityTestEngine


class SecurityToolRunner:
    def __init__(self, engine: SecurityTestEngine) -> None:
        self._engine = engine

    def authorize_targets(
        self,
        targets: Sequence[str],
        *,
        tool: str,
        method: str = "GET",
        active: bool = True,
    ) -> list[str]:
        allowed: list[str] = []
        for target in targets:
            decision = self._engine.authorize(target, method=method, tool=tool, active=active)
            self._engine.audit.record(
                project=self._engine.project_id,
                target=target,
                scope_decision=decision.reason,
                tool=tool,
                action="authorize_target",
                result="allowed" if decision.allowed else "denied",
                human_approval=decision.approval_state,
            )
            if not decision.allowed:
                raise AuthorizationDeniedError(decision.reason, target=target, tool=tool)
            allowed.append(target)
        return allowed

    def wrap_output(self, tool: str, output: str) -> str:
        return wrap_untrusted(tool, output)

    async def run(
        self,
        tool: str,
        operation: Callable[..., Any],
        *,
        targets: Sequence[str],
        method: str = "GET",
        active: bool = True,
        **kwargs: Any,
    ) -> Any:
        try:
            self.authorize_targets(targets, tool=tool, method=method, active=active)
        except AuthorizationDeniedError as exc:
            return ToolExecutionResult(
                tool=tool, state=ToolExecutionState.INVALID_SCOPE, detail=str(exc)
            )
        try:
            result = operation(**kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        except TimeoutError as exc:
            return ToolExecutionResult(tool=tool, state=ToolExecutionState.TIMEOUT, detail=str(exc))
        except FileNotFoundError as exc:
            return ToolExecutionResult(
                tool=tool, state=ToolExecutionState.TOOL_UNAVAILABLE, detail=str(exc)
            )
