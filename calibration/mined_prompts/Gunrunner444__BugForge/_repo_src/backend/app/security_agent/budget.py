"""Session budgets. The AI may request more budget but cannot grant it."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.security_testing.errors import SafetyLimitExceededError


@dataclass
class SessionBudget:
    max_tool_calls: int = 40
    max_requests: int = 50
    max_browser_actions: int = 20
    max_fuzz_requests: int = 20
    max_iterations: int = 20
    max_tokens: int = 200_000
    max_scan_seconds: float = 300.0
    max_identical_calls: int = 2
    identical_call_window_seconds: float = 120.0
    max_exploratory_tests: int = 4
    max_exploratory_iterations: int = 3
    max_exploratory_seconds: float = 60.0
    tool_calls: int = 0
    requests: int = 0
    browser_actions: int = 0
    fuzz_requests: int = 0
    iterations: int = 0
    tokens: int = 0
    scan_seconds: float = 0.0
    exploratory_tests: int = 0
    exploratory_iterations: int = 0
    exploratory_seconds: float = 0.0
    extra_granted_by: str | None = None
    planned_requests: int = 0
    reserved_requests: int = 0
    planned_fuzz_requests: int = 0
    reserved_fuzz_requests: int = 0
    planned_browser_actions: int = 0
    reserved_browser_actions: int = 0
    planned_tool_calls: int = 0
    reserved_tool_calls: int = 0
    planned_exploratory_tests: int = 0
    reserved_exploratory_tests: int = 0
    planned_exploratory_iterations: int = 0
    reserved_exploratory_iterations: int = 0
    planned_exploratory_seconds: float = 0.0
    reserved_exploratory_seconds: float = 0.0

    @classmethod
    def from_settings(cls) -> SessionBudget:
        settings = get_settings()
        return cls(
            max_tool_calls=settings.security_agent_max_tool_calls,
            max_requests=settings.security_agent_max_requests,
            max_browser_actions=settings.security_agent_max_browser_actions,
            max_fuzz_requests=settings.security_agent_max_fuzz_requests,
            max_iterations=settings.security_agent_max_iterations,
            max_tokens=settings.security_agent_max_tokens,
            max_scan_seconds=settings.security_agent_max_scan_seconds,
            max_identical_calls=settings.security_agent_max_identical_tool_calls,
            identical_call_window_seconds=settings.security_agent_identical_call_window_seconds,
            max_exploratory_tests=settings.security_agent_max_exploratory_tests,
            max_exploratory_iterations=settings.security_agent_max_exploratory_iterations,
            max_exploratory_seconds=settings.security_agent_max_exploratory_seconds,
        )

    def _bucket(self, unit: str) -> str:
        if unit in {"requests", "request"}:
            return "requests"
        if unit in {"fuzz_requests", "fuzz"}:
            return "fuzz_requests"
        if unit in {"browser_actions", "browser"}:
            return "browser_actions"
        if unit in {"exploratory_tests", "exploratory"}:
            return "exploratory_tests"
        if unit in {"exploratory_iterations"}:
            return "exploratory_iterations"
        if unit in {"exploratory_seconds"}:
            return "exploratory_seconds"
        return "tool_calls"

    def plan(self, unit: str, amount: int | float) -> None:
        """Record a planner estimate. Does not consume budget."""
        bucket = self._bucket(unit)
        if bucket == "requests":
            self.planned_requests += max(0, int(amount))
        elif bucket == "fuzz_requests":
            self.planned_fuzz_requests += max(0, int(amount))
        elif bucket == "browser_actions":
            self.planned_browser_actions += max(0, int(amount))
        elif bucket == "exploratory_tests":
            self.planned_exploratory_tests += max(0, int(amount))
        elif bucket == "exploratory_iterations":
            self.planned_exploratory_iterations += max(0, int(amount))
        elif bucket == "exploratory_seconds":
            self.planned_exploratory_seconds += max(0.0, float(amount))
        else:
            self.planned_tool_calls += max(0, int(amount))

    def reserve(self, unit: str, amount: int | float) -> None:
        """Reserve estimated units before execution. Same units as consume()."""
        bucket = self._bucket(unit)
        if bucket == "requests":
            self.reserved_requests = max(0, int(amount))
        elif bucket == "fuzz_requests":
            self.reserved_fuzz_requests = max(0, int(amount))
        elif bucket == "browser_actions":
            self.reserved_browser_actions = max(0, int(amount))
        elif bucket == "exploratory_tests":
            self.reserved_exploratory_tests = max(0, int(amount))
        elif bucket == "exploratory_iterations":
            self.reserved_exploratory_iterations = max(0, int(amount))
        elif bucket == "exploratory_seconds":
            self.reserved_exploratory_seconds = max(0.0, float(amount))
        else:
            self.reserved_tool_calls = max(0, int(amount))

    def consume(self, kind: str, *, amount: int = 1) -> None:
        if kind == "tool":
            self.tool_calls += amount
            if self.tool_calls > self.max_tool_calls:
                raise SafetyLimitExceededError("tool call budget exhausted")
        elif kind == "request":
            self.requests += amount
            if self.requests > self.max_requests:
                raise SafetyLimitExceededError("request budget exhausted")
        elif kind == "browser":
            self.browser_actions += amount
            if self.browser_actions > self.max_browser_actions:
                raise SafetyLimitExceededError("browser action budget exhausted")
        elif kind == "fuzz":
            self.fuzz_requests += amount
            if self.fuzz_requests > self.max_fuzz_requests:
                raise SafetyLimitExceededError("fuzz request budget exhausted")
        elif kind == "iteration":
            self.iterations += amount
            if self.iterations > self.max_iterations:
                raise SafetyLimitExceededError("reasoning iteration budget exhausted")
        elif kind == "token":
            self.tokens += amount
            if self.tokens > self.max_tokens:
                raise SafetyLimitExceededError("token budget exhausted")
        elif kind == "scan":
            self.scan_seconds += float(amount)
            if self.scan_seconds > self.max_scan_seconds:
                raise SafetyLimitExceededError("scan duration budget exhausted")
        elif kind == "exploratory":
            if self.exploratory_seconds >= self.max_exploratory_seconds:
                raise SafetyLimitExceededError("exploratory time budget exhausted")
            self.exploratory_tests += amount
            self.exploratory_iterations += amount
            if self.exploratory_tests > self.max_exploratory_tests:
                raise SafetyLimitExceededError("exploratory test budget exhausted")
            if self.exploratory_iterations > self.max_exploratory_iterations:
                raise SafetyLimitExceededError("exploratory iteration budget exhausted")

    def consume_scan(self, seconds: float) -> None:
        self.scan_seconds += max(0.0, seconds)
        if self.scan_seconds > self.max_scan_seconds:
            raise SafetyLimitExceededError("scan duration budget exhausted")

    def note_exploratory_seconds(self, seconds: float) -> None:
        """Record sandbox time. The model cannot raise the cap."""
        self.exploratory_seconds += max(0.0, seconds)
        if self.exploratory_seconds > self.max_exploratory_seconds:
            raise SafetyLimitExceededError("exploratory time budget exhausted")

    def remaining(self) -> dict[str, int | float]:
        return {
            "tool_calls": self.max_tool_calls - self.tool_calls,
            "requests": self.max_requests - self.requests,
            "browser_actions": self.max_browser_actions - self.browser_actions,
            "fuzz_requests": self.max_fuzz_requests - self.fuzz_requests,
            "iterations": self.max_iterations - self.iterations,
            "tokens": self.max_tokens - self.tokens,
            "scan_seconds": self.max_scan_seconds - self.scan_seconds,
            "exploratory_tests": self.max_exploratory_tests - self.exploratory_tests,
            "exploratory_iterations": self.max_exploratory_iterations - self.exploratory_iterations,
            "exploratory_seconds": self.max_exploratory_seconds - self.exploratory_seconds,
        }

    def reallocate(self, *, source: str, destination: str, amount: int) -> None:
        """Move unused budget between categories. Cannot increase the total cap."""
        if amount <= 0:
            raise SafetyLimitExceededError("reallocate amount must be positive")
        mapping = {
            "tool": ("max_tool_calls", "tool_calls"),
            "request": ("max_requests", "requests"),
            "browser": ("max_browser_actions", "browser_actions"),
            "fuzz": ("max_fuzz_requests", "fuzz_requests"),
            "iteration": ("max_iterations", "iterations"),
        }
        if source not in mapping or destination not in mapping or source == destination:
            raise SafetyLimitExceededError("invalid budget reallocation")
        src_max, src_used = mapping[source]
        dst_max, _dst_used = mapping[destination]
        available = int(getattr(self, src_max)) - int(getattr(self, src_used))
        if amount > available:
            raise SafetyLimitExceededError("cannot reallocate more than remaining source budget")
        setattr(self, src_max, int(getattr(self, src_max)) - amount)
        setattr(self, dst_max, int(getattr(self, dst_max)) + amount)

    def exhausted(self) -> bool:
        rem = self.remaining()
        return any(float(value) <= 0 for value in rem.values())

    def grant_more(self, *, operator: str, extra_tool_calls: int = 0) -> None:
        from app.security_testing.approvals import is_ai_operator
        from app.security_testing.errors import RestrictedActivityError

        if is_ai_operator(operator):
            raise RestrictedActivityError("The AI cannot grant itself additional budget")
        self.max_tool_calls += max(0, extra_tool_calls)
        self.extra_granted_by = operator

    def restore(self, payload: dict[str, object] | None) -> None:
        if not payload:
            return
        maximum = payload.get("max") if isinstance(payload.get("max"), dict) else payload
        used = payload.get("used") if isinstance(payload.get("used"), dict) else payload
        if isinstance(maximum, dict):
            self.max_tool_calls = int(maximum.get("tool_calls", self.max_tool_calls))
            self.max_requests = int(maximum.get("requests", self.max_requests))
            self.max_browser_actions = int(maximum.get("browser_actions", self.max_browser_actions))
            self.max_fuzz_requests = int(maximum.get("fuzz_requests", self.max_fuzz_requests))
            self.max_iterations = int(maximum.get("iterations", self.max_iterations))
            self.max_tokens = int(maximum.get("tokens", self.max_tokens))
            self.max_scan_seconds = float(maximum.get("scan_seconds", self.max_scan_seconds))
            self.max_identical_calls = int(maximum.get("identical_calls", self.max_identical_calls))
            self.max_exploratory_tests = int(
                maximum.get("exploratory_tests", self.max_exploratory_tests)
            )
            self.max_exploratory_iterations = int(
                maximum.get("exploratory_iterations", self.max_exploratory_iterations)
            )
            self.max_exploratory_seconds = float(
                maximum.get("exploratory_seconds", self.max_exploratory_seconds)
            )
        if isinstance(used, dict):
            self.tool_calls = int(used.get("tool_calls", 0))
            self.requests = int(used.get("requests", 0))
            self.browser_actions = int(used.get("browser_actions", 0))
            self.fuzz_requests = int(used.get("fuzz_requests", 0))
            self.iterations = int(used.get("iterations", 0))
            self.tokens = int(used.get("tokens", 0))
            self.scan_seconds = float(used.get("scan_seconds", 0))
            self.exploratory_tests = int(used.get("exploratory_tests", 0))
            self.exploratory_iterations = int(used.get("exploratory_iterations", 0))
            self.exploratory_seconds = float(used.get("exploratory_seconds", 0))

    def snapshot(self) -> dict[str, object]:
        return {
            "max": {
                "tool_calls": self.max_tool_calls,
                "requests": self.max_requests,
                "browser_actions": self.max_browser_actions,
                "fuzz_requests": self.max_fuzz_requests,
                "iterations": self.max_iterations,
                "tokens": self.max_tokens,
                "scan_seconds": self.max_scan_seconds,
                "identical_calls": self.max_identical_calls,
                "exploratory_tests": self.max_exploratory_tests,
                "exploratory_iterations": self.max_exploratory_iterations,
                "exploratory_seconds": self.max_exploratory_seconds,
            },
            "used": {
                "tool_calls": self.tool_calls,
                "requests": self.requests,
                "browser_actions": self.browser_actions,
                "fuzz_requests": self.fuzz_requests,
                "iterations": self.iterations,
                "tokens": self.tokens,
                "scan_seconds": self.scan_seconds,
                "exploratory_tests": self.exploratory_tests,
                "exploratory_iterations": self.exploratory_iterations,
                "exploratory_seconds": self.exploratory_seconds,
            },
            "remaining": self.remaining(),
            "extra_granted_by": self.extra_granted_by,
            "planned": {
                "requests": self.planned_requests,
                "fuzz_requests": self.planned_fuzz_requests,
                "browser_actions": self.planned_browser_actions,
                "tool_calls": self.planned_tool_calls,
                "exploratory_tests": self.planned_exploratory_tests,
                "exploratory_iterations": self.planned_exploratory_iterations,
                "exploratory_seconds": self.planned_exploratory_seconds,
            },
            "reserved": {
                "requests": self.reserved_requests,
                "fuzz_requests": self.reserved_fuzz_requests,
                "browser_actions": self.reserved_browser_actions,
                "tool_calls": self.reserved_tool_calls,
                "exploratory_tests": self.reserved_exploratory_tests,
                "exploratory_iterations": self.reserved_exploratory_iterations,
                "exploratory_seconds": self.reserved_exploratory_seconds,
            },
            "consumed": {
                "tool_calls": self.tool_calls,
                "requests": self.requests,
                "browser_actions": self.browser_actions,
                "fuzz_requests": self.fuzz_requests,
                "iterations": self.iterations,
                "tokens": self.tokens,
                "scan_seconds": self.scan_seconds,
                "exploratory_tests": self.exploratory_tests,
                "exploratory_iterations": self.exploratory_iterations,
                "exploratory_seconds": self.exploratory_seconds,
            },
        }
