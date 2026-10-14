"""Per-process token-bucket / sliding-window rate limiter for active testing.

This limiter is **in-memory and per process**. Multiple API workers do not
share a global per-target budget. Production/live deployments must either:

* run a single worker, or
* put a shared limiter (Redis, database, or reverse-proxy) in front of BugForge.

Do not treat these counters as cluster-wide guarantees.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from app.security_testing.errors import SafetyLimitExceededError
from app.security_testing.safety import SafetyLimits


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    reason: str
    retry_after_seconds: float = 0.0


class RateLimiter:
    def __init__(
        self,
        limits: SafetyLimits | None = None,
        *,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.limits = limits or SafetyLimits.conservative()
        self._times: deque[float] = deque()
        self._monotonic = monotonic or time.monotonic

    def check(self) -> RateLimitDecision:
        now = self._monotonic()
        window = 1.0
        while self._times and now - self._times[0] >= window:
            self._times.popleft()
        max_per_window = max(1, int(self.limits.requests_per_second))
        # Allow fractional rps by scaling: rps=0.5 → 1 request / 2s handled via retry.
        if self.limits.requests_per_second < 1:
            if self._times and now - self._times[-1] < (1.0 / self.limits.requests_per_second):
                wait = (1.0 / self.limits.requests_per_second) - (now - self._times[-1])
                return RateLimitDecision(
                    False,
                    "Rate limit exceeded",
                    retry_after_seconds=max(0.0, wait),
                )
            return RateLimitDecision(True, "Rate limit ok")
        if len(self._times) >= max_per_window:
            wait = window - (now - self._times[0])
            return RateLimitDecision(
                False, "Rate limit exceeded", retry_after_seconds=max(0.0, wait)
            )
        return RateLimitDecision(True, "Rate limit ok")

    def acquire(self) -> RateLimitDecision:
        decision = self.check()
        if decision.allowed:
            self._times.append(self._monotonic())
        return decision

    def require(self) -> None:
        decision = self.acquire()
        if not decision.allowed:
            raise SafetyLimitExceededError(
                f"{decision.reason} (retry after {decision.retry_after_seconds:.2f}s)"
            )
