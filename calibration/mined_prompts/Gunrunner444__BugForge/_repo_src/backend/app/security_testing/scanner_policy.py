"""Scanner execution envelope.

Third-party scanners (ZAP, Nuclei, …) generate their own network traffic.
BugForge's per-request :class:`RateLimiter` does **not** govern every request
made inside an external scanner process.

BugForge remains the authority over:

* whether execution is allowed
* which targets are allowed
* whether active testing is enabled
* the maximum permitted execution envelope

The external tool must be configured with limits no less restrictive than this
envelope (min of scanner-specific and SafetyController limits).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.security_testing.safety import SafetyLimits


@dataclass(frozen=True)
class ScannerExecutionPolicy:
    """Maximum envelope BugForge will hand to an external scanner."""

    max_targets: int = 5
    max_runtime_seconds: float = 300.0
    requests_per_second: float = 1.0
    concurrency: int = 1
    allowed_templates: tuple[str, ...] = ()
    denied_templates: tuple[str, ...] = ()
    allowed_tags: tuple[str, ...] = ()
    denied_tags: tuple[str, ...] = ("dos", "intrusive", "takeover")
    allowed_methods: tuple[str, ...] = ("GET", "HEAD", "OPTIONS")
    allow_destructive: bool = False

    def tighten(self, safety: SafetyLimits) -> ScannerExecutionPolicy:
        """Intersect with SafetyController limits (never looser than safety)."""
        methods = tuple(
            method
            for method in self.allowed_methods
            if method.upper() in {item.upper() for item in safety.allowed_http_methods}
        )
        if not methods:
            methods = safety.allowed_http_methods
        return replace(
            self,
            max_targets=min(self.max_targets, safety.max_targets),
            max_runtime_seconds=min(self.max_runtime_seconds, safety.max_scan_duration_seconds),
            requests_per_second=min(self.requests_per_second, safety.requests_per_second),
            concurrency=min(self.concurrency, safety.max_concurrent),
            allowed_methods=methods,
            allow_destructive=self.allow_destructive and safety.allow_destructive,
        )

    def describe(self) -> str:
        return (
            "ScannerExecutionPolicy governs the envelope passed to an external "
            "scanner. It does not claim BugForge's per-request RateLimiter sees "
            "every request the scanner process makes."
        )
