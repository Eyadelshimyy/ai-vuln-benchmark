"""ReproductionPlan executed through ScopeGuard + SafetyController."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.security_agent.oracles import OracleType, artifact_hash, evaluate_oracle
from app.security_agent.states import ReproductionOutcome
from app.security_testing.engine import SecurityTestEngine
from app.security_testing.errors import AuthorizationDeniedError, RestrictedActivityError
from app.security_testing.failures import ToolExecutionResult
from app.security_testing.sanitization import wrap_untrusted
from app.security_testing.secrets import redact_text


@dataclass
class ReproductionAction:
    method: str
    url: str
    content: str | None = None
    expected_status: int | None = None
    expected_body_contains: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ReproductionPlan:
    preconditions: tuple[str, ...] = ()
    setup: str = ""
    actions: tuple[ReproductionAction, ...] = ()
    expected_result: str = ""
    actual_result: str = ""
    evidence_to_collect: tuple[str, ...] = ()
    cleanup: str = ""
    status: str = "planned"
    outcome: ReproductionOutcome = ReproductionOutcome.INCONCLUSIVE
    reproducibility_count: int = 1
    runs: tuple[str, ...] = ()
    identity_context: str = ""
    oracle_type: str = ""
    request_hashes: tuple[str, ...] = ()
    response_hashes: tuple[str, ...] = ()
    cleanup_result: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "preconditions": list(self.preconditions),
            "setup": self.setup,
            "identity_context": self.identity_context,
            "actions": [
                {
                    "method": item.method,
                    "url": item.url,
                    "expected_status": item.expected_status,
                    "expected_body_contains": item.expected_body_contains,
                }
                for item in self.actions
            ],
            "expected_result": self.expected_result,
            "actual_result": self.actual_result,
            "evidence_to_collect": list(self.evidence_to_collect),
            "cleanup": self.cleanup,
            "cleanup_result": self.cleanup_result,
            "status": self.status,
            "outcome": self.outcome.value,
            "oracle_type": self.oracle_type,
            "reproducibility_count": self.reproducibility_count,
            "runs": list(self.runs),
            "request_hashes": list(self.request_hashes),
            "response_hashes": list(self.response_hashes),
        }


class ReproductionEngine:
    """Oracle-backed reproduction. HTTP status alone is never success."""

    def __init__(self, engine: SecurityTestEngine) -> None:
        self.engine = engine

    async def execute(self, plan: ReproductionPlan) -> ReproductionPlan:
        if not plan.actions:
            plan.status = ReproductionOutcome.INCONCLUSIVE.value
            plan.outcome = ReproductionOutcome.INCONCLUSIVE
            plan.actual_result = "no actions"
            return plan
        if not _has_oracle(plan):
            # Status-only plans cannot advance a finding to reproduced.
            plan.outcome = ReproductionOutcome.INCONCLUSIVE
            plan.status = plan.outcome.value
        successes = 0
        failures = 0
        blocked = 0
        env_errors = 0
        run_notes: list[str] = []
        last_actual = ""
        repeats = max(1, plan.reproducibility_count)
        for _ in range(repeats):
            try:
                outcome, actual = await self._run_once(plan)
            except RestrictedActivityError as exc:
                plan.outcome = ReproductionOutcome.BLOCKED
                plan.status = plan.outcome.value
                plan.actual_result = str(exc)
                plan.runs = tuple(run_notes + [str(exc)])
                return plan
            run_notes.append(f"{outcome.value}:{actual}")
            last_actual = actual
            if outcome is ReproductionOutcome.REPRODUCED:
                successes += 1
            elif outcome is ReproductionOutcome.NOT_REPRODUCED:
                failures += 1
            elif outcome is ReproductionOutcome.BLOCKED:
                blocked += 1
            elif outcome is ReproductionOutcome.ENVIRONMENT_ERROR:
                env_errors += 1
            else:
                failures += 0
        plan.runs = tuple(run_notes)
        plan.actual_result = last_actual
        if blocked and not successes:
            plan.outcome = ReproductionOutcome.BLOCKED
        elif env_errors and not successes:
            plan.outcome = ReproductionOutcome.ENVIRONMENT_ERROR
        elif successes == repeats and _has_oracle(plan):
            plan.outcome = ReproductionOutcome.REPRODUCED
        elif failures and not successes:
            plan.outcome = ReproductionOutcome.NOT_REPRODUCED
        else:
            plan.outcome = ReproductionOutcome.INCONCLUSIVE
        plan.status = plan.outcome.value
        return plan

    async def _run_once(self, plan: ReproductionPlan) -> tuple[ReproductionOutcome, str]:
        results: list[str] = []
        matched = 0
        observed = 0
        for action in plan.actions:
            if (
                action.method.upper() in {"DELETE", "PUT", "PATCH"}
                and not self.engine.session.scope.lab_mode
            ):
                raise RestrictedActivityError("destructive_reproduction")
            try:
                exchange = await self.engine.http("reproduce").request(
                    action.method,
                    action.url,
                    headers=action.headers or None,
                    content=action.content,
                    active=True,
                    destructive=False,
                )
            except AuthorizationDeniedError:
                return ReproductionOutcome.BLOCKED, "blocked by ScopeGuard/SafetyController"
            if isinstance(exchange, ToolExecutionResult):
                if exchange.state.value in {"timeout"}:
                    return ReproductionOutcome.ENVIRONMENT_ERROR, exchange.detail or "timeout"
                if exchange.state.value in {"dry_run"}:
                    return ReproductionOutcome.INCONCLUSIVE, "dry-run; not executed"
                results.append(exchange.state.value)
                continue
            status = exchange.response_status
            body = redact_text(str(exchange.response_body or ""))[:2000]
            results.append(f"{action.method} {action.url} -> {status}")
            observed += 1
            req_material = f"{action.method} {action.url} {action.content or ''}"
            plan.request_hashes = plan.request_hashes + (artifact_hash(req_material),)
            plan.response_hashes = plan.response_hashes + (artifact_hash(f"{status}:{body}"),)
            oracle_hit = _oracle_match(
                plan,
                action,
                status=status,
                body=body,
            )
            if oracle_hit is False:
                return ReproductionOutcome.NOT_REPRODUCED, "; ".join(results)
            if oracle_hit is True:
                matched += 1
        actual = wrap_untrusted("reproduction", "; ".join(results) or "no actions")
        if not _has_oracle(plan):
            return ReproductionOutcome.INCONCLUSIVE, actual
        if matched > 0 and matched == observed:
            return ReproductionOutcome.REPRODUCED, actual
        if observed == 0:
            return ReproductionOutcome.ENVIRONMENT_ERROR, actual
        return ReproductionOutcome.INCONCLUSIVE, actual


async def execute_reproduction(
    plan: ReproductionPlan,
    engine: SecurityTestEngine,
) -> ReproductionPlan:
    """Run the plan through the existing safety architecture. Never mutate real data."""
    return await ReproductionEngine(engine).execute(plan)


def _has_oracle(plan: ReproductionPlan) -> bool:
    if (plan.oracle_type or "").strip():
        return True
    if (plan.expected_result or "").strip():
        return True
    return any(
        action.expected_body_contains
        or (action.expected_status is not None and plan.expected_result)
        or action.expected_body_contains
        for action in plan.actions
    ) or any(bool(action.expected_body_contains) for action in plan.actions)


def _oracle_match(
    plan: ReproductionPlan,
    action: ReproductionAction,
    *,
    status: int | None,
    body: str,
) -> bool | None:
    """True = matched oracle, False = contradicted, None = status-only / no oracle."""
    expected_body = action.expected_body_contains or plan.expected_result
    if plan.oracle_type:
        try:
            oracle = OracleType(plan.oracle_type)
        except ValueError:
            oracle = None
        if oracle is not None:
            actual: Any = body
            expected: Any = expected_body
            if oracle is OracleType.EXACT_RESPONSE:
                actual = body
            if oracle is OracleType.AUTHORIZATION_DIFFERENCE:
                actual = status
                expected = action.expected_status
            return evaluate_oracle(oracle, expected=expected, actual=actual)
    if expected_body:
        present = expected_body.lower() in (body or "").lower()
        if action.expected_status is not None and status != action.expected_status:
            return False
        return present
    if action.expected_status is not None:
        # Status-only is not sufficient for REPRODUCED.
        return None if status == action.expected_status else False
    return None
