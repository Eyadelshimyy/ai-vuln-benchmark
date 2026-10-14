"""Tool cost estimates. Estimates are not guarantees; units match budget consume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.security_agent.states import ToolRiskLevel


class _ToolSpecLike(Protocol):
    @property
    def budget_kind(self) -> str: ...

    @property
    def risk_level(self) -> ToolRiskLevel: ...


@dataclass(frozen=True)
class CostEstimate:
    """Planned cost for a proposed operation.

    ``estimated_units`` uses the same unit that :class:`SessionBudget.consume`
    will debit. ``estimated_requests`` is the request-equivalent used by
    next-action review. Values are estimates, not guarantees.
    """

    estimated_units: int
    unit: str
    estimated_requests: int
    is_estimate: bool = True
    explanation: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "estimated_units": self.estimated_units,
            "unit": self.unit,
            "estimated_requests": self.estimated_requests,
            "is_estimate": self.is_estimate,
            "explanation": self.explanation,
        }


def estimate_operation_cost(
    tool: str,
    arguments: dict[str, Any] | None,
    spec: _ToolSpecLike | None = None,
    *,
    remaining_requests: int | None = None,
) -> CostEstimate:
    args = arguments or {}
    kind = spec.budget_kind if spec is not None else "tool"
    if tool == "http_request" or kind == "request" and tool not in {"api_test", "reproduce"}:
        return CostEstimate(
            estimated_units=1,
            unit="requests",
            estimated_requests=1,
            explanation="One HTTP request through GatedHttpClient.",
        )
    if tool == "fuzz" or kind == "fuzz":
        count = max(1, int(args.get("count") or 1))
        return CostEstimate(
            estimated_units=count,
            unit="fuzz_requests",
            estimated_requests=count,
            explanation=f"Fuzz count={count}; each mutation is one request-equivalent.",
        )
    if tool == "api_test":
        if args.get("execute", True) is False:
            return CostEstimate(
                estimated_units=0,
                unit="requests",
                estimated_requests=0,
                explanation="Plan-only API import; no requests until execute=true.",
            )
        count = max(1, int(args.get("max_tests") or 8))
        return CostEstimate(
            estimated_units=count,
            unit="requests",
            estimated_requests=count,
            explanation=f"API test envelope of {count} candidate requests.",
        )
    if tool == "reproduce":
        actions = args.get("actions") or []
        count = max(1, len(actions) if isinstance(actions, list) and actions else 1)
        repeats = max(1, int(args.get("reproducibility_count") or 1))
        total = count * repeats
        return CostEstimate(
            estimated_units=total,
            unit="requests",
            estimated_requests=total,
            explanation=f"Reproduction plan: {count} action(s) × {repeats} run(s).",
        )
    if tool in {"zap_scan", "nuclei_scan"}:
        envelope = remaining_requests if remaining_requests is not None else 20
        envelope = max(1, int(envelope))
        return CostEstimate(
            estimated_units=1,
            unit="tool_calls",
            estimated_requests=envelope,
            explanation=(
                f"Scanner envelope ~{envelope} requests (estimate for review). "
                "Session budget consumes one tool call plus scan duration after execution, "
                "not the full envelope up front."
            ),
        )
    if tool == "exploratory_test" or kind == "exploratory":
        return CostEstimate(
            estimated_units=1,
            unit="exploratory_tests",
            estimated_requests=0,
            explanation="One temporary generated test. Network stays disabled.",
        )
    if tool == "browser_navigate" or kind == "browser":
        extra = 1
        if args.get("capture_screenshot", True):
            extra += 1
        if args.get("capture_console", True):
            extra += 1
        return CostEstimate(
            estimated_units=1,
            unit="browser_actions",
            estimated_requests=extra,
            explanation="One browser navigation plus estimated network captures.",
        )
    return CostEstimate(
        estimated_units=1,
        unit="tool_calls",
        estimated_requests=0,
        explanation="Passive/local tool; no network request budget.",
    )


def consume_units_for(estimate: CostEstimate) -> tuple[str, int]:
    """Map an estimate onto SessionBudget.consume(kind, amount=...)."""
    mapping = {
        "requests": "request",
        "fuzz_requests": "fuzz",
        "browser_actions": "browser",
        "tool_calls": "tool",
        "scan_seconds": "scan",
        "exploratory_tests": "exploratory",
    }
    return mapping.get(estimate.unit, "tool"), estimate.estimated_units


def risk_label(spec: _ToolSpecLike | None) -> str:
    if spec is None:
        return ToolRiskLevel.PASSIVE.value
    return spec.risk_level.value
