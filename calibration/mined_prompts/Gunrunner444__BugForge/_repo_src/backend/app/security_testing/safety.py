# BugForge's per-request RateLimiter applies to GatedHttpClient traffic.
# External scanners have their own ScannerExecutionPolicy envelope; see
# app.security_testing.scanner_policy.

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from app.security_testing.errors import RestrictedActivityError, SafetyLimitExceededError
from app.security_testing.scope_model import AuthorizationDecision


class RestrictedActivity(StrEnum):
    DENIAL_OF_SERVICE = "denial_of_service"
    SOCIAL_ENGINEERING = "social_engineering"
    NOTIFICATION_SPAM = "notification_spam"
    PHYSICAL_SECURITY = "physical_security"
    OUT_OF_SCOPE = "out_of_scope"
    HACKERONE_SUBMISSION = "hackerone_submission"


_ALWAYS_FORBIDDEN = frozenset(RestrictedActivity)


@dataclass(frozen=True)
class SafetyLimits:
    max_requests: int = 50
    requests_per_second: float = 1.0
    max_concurrent: int = 2
    timeout_seconds: float = 10.0
    max_scan_duration_seconds: float = 300.0
    max_targets: int = 10
    allowed_http_methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    max_response_bytes: int = 1_000_000
    max_payload_bytes: int = 8_192
    max_redirects: int = 3
    allow_destructive: bool = False
    max_payload_count: int = 20

    @classmethod
    def conservative(cls) -> SafetyLimits:
        return cls()

    @classmethod
    def lab(cls) -> SafetyLimits:
        return cls(
            max_requests=100,
            requests_per_second=5.0,
            max_concurrent=4,
            timeout_seconds=5.0,
            max_scan_duration_seconds=120.0,
            max_targets=5,
            allowed_http_methods=("GET", "HEAD", "OPTIONS", "POST", "PUT"),
            max_response_bytes=500_000,
            max_payload_bytes=4_096,
        )


@dataclass
class SafetyDecision:
    allowed: bool
    reason: str
    dry_run: bool = False
    would_execute: str = ""


class SafetyController:
    """Gate every active operation. Dry-run explains without sending traffic."""

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.limits = limits or SafetyLimits.conservative()
        self.dry_run = dry_run
        self._request_count = 0
        self._in_flight = 0
        self._seen_targets: set[str] = set()
        self._started_at: datetime | None = None
        self._log: list[tuple[datetime, str, bool, str]] = []

    def forbid(self, activity: RestrictedActivity | str) -> None:
        name = activity.value if isinstance(activity, RestrictedActivity) else activity
        if name in {item.value for item in _ALWAYS_FORBIDDEN} or name in _ALWAYS_FORBIDDEN:
            raise RestrictedActivityError(name)
        raise RestrictedActivityError(str(name))

    def check_activity(self, activity: RestrictedActivity | str) -> None:
        self.forbid(activity)

    def begin_scan(self) -> None:
        if self._started_at is None:
            self._started_at = datetime.now(UTC)

    def check_request(
        self,
        *,
        target: str,
        method: str,
        tool: str,
        payload_bytes: int = 0,
        destructive: bool = False,
        scope: AuthorizationDecision | None = None,
    ) -> SafetyDecision:
        self.begin_scan()
        method_u = method.upper()
        if scope is not None and not scope.allowed:
            decision = SafetyDecision(False, f"Scope denied: {scope.reason}", dry_run=self.dry_run)
            self._record(target, tool, False, decision.reason)
            return decision
        if method_u not in {m.upper() for m in self.limits.allowed_http_methods}:
            decision = SafetyDecision(
                False,
                f"HTTP method {method_u} is outside safety-allowed methods",
                dry_run=self.dry_run,
            )
            self._record(target, tool, False, decision.reason)
            return decision
        if destructive and not self.limits.allow_destructive:
            decision = SafetyDecision(
                False, "Destructive operations are disabled", dry_run=self.dry_run
            )
            self._record(target, tool, False, decision.reason)
            return decision
        if payload_bytes > self.limits.max_payload_bytes:
            decision = SafetyDecision(
                False,
                f"Payload {payload_bytes} exceeds max_payload_bytes={self.limits.max_payload_bytes}",
                dry_run=self.dry_run,
            )
            self._record(target, tool, False, decision.reason)
            return decision
        if self._request_count >= self.limits.max_requests:
            decision = SafetyDecision(
                False,
                f"Maximum request count {self.limits.max_requests} exceeded",
                dry_run=self.dry_run,
            )
            self._record(target, tool, False, decision.reason)
            return decision
        if self._started_at is not None:
            elapsed = datetime.now(UTC) - self._started_at
            if elapsed > timedelta(seconds=self.limits.max_scan_duration_seconds):
                decision = SafetyDecision(
                    False,
                    f"Scan duration limit {self.limits.max_scan_duration_seconds}s exceeded",
                    dry_run=self.dry_run,
                )
                self._record(target, tool, False, decision.reason)
                return decision
        if self._in_flight >= self.limits.max_concurrent:
            decision = SafetyDecision(
                False,
                f"Concurrency limit {self.limits.max_concurrent} exceeded",
                dry_run=self.dry_run,
            )
            self._record(target, tool, False, decision.reason)
            return decision
        seen = set(self._seen_targets)
        seen.add(target)
        if len(seen) > self.limits.max_targets:
            decision = SafetyDecision(
                False,
                f"Target count limit {self.limits.max_targets} exceeded",
                dry_run=self.dry_run,
            )
            self._record(target, tool, False, decision.reason)
            return decision
        planned = f"{method_u} {target} via {tool}"
        if self.dry_run:
            decision = SafetyDecision(
                True,
                "Dry-run: request would be allowed but will not be sent",
                dry_run=True,
                would_execute=planned,
            )
            self._record(target, tool, True, decision.reason)
            return decision
        decision = SafetyDecision(
            True, "Safety checks passed", dry_run=False, would_execute=planned
        )
        self._record(target, tool, True, decision.reason)
        return decision

    def acquire(self, target: str) -> None:
        self._request_count += 1
        self._in_flight += 1
        self._seen_targets.add(target)

    def release(self) -> None:
        if self._in_flight:
            self._in_flight -= 1

    def remaining_requests(self) -> int:
        return max(0, self.limits.max_requests - self._request_count)

    def request_count(self) -> int:
        return self._request_count

    def with_dry_run(self, enabled: bool) -> SafetyController:
        clone = SafetyController(self.limits, dry_run=enabled)
        clone._request_count = self._request_count
        clone._in_flight = self._in_flight
        clone._seen_targets = set(self._seen_targets)
        clone._started_at = self._started_at
        clone._log = list(self._log)
        return clone

    def tighten(self, **changes: object) -> None:
        self.limits = replace(self.limits, **changes)  # type: ignore[arg-type]

    def decisions(self) -> list[tuple[datetime, str, bool, str]]:
        return list(self._log)

    def _record(self, target: str, tool: str, allowed: bool, reason: str) -> None:
        self._log.append((datetime.now(UTC), f"{tool}:{target}", allowed, reason))


def raise_if_denied(decision: SafetyDecision) -> None:
    if not decision.allowed:
        raise SafetyLimitExceededError(decision.reason)
