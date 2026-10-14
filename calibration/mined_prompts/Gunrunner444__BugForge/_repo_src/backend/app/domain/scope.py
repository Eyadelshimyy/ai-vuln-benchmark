"""Authorized-testing scope. Domain types only — no HackerOne API types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeConstraint:
    """What future security testing is allowed to touch.

    An empty ``allowed_hosts`` list means nothing is in scope. Active testing
    is opted in explicitly; Phase 1 never performs it.
    """

    allowed_hosts: tuple[str, ...] = ()
    excluded_hosts: tuple[str, ...] = ()
    allowed_methods: tuple[str, ...] = ()
    rate_limit_per_minute: int | None = None
    instructions: str = ""
    allow_active_testing: bool = False
    program_name: str | None = None

    def allows_host(self, host: str) -> bool:
        normalized = host.strip().lower()
        if not normalized:
            return False
        if _host_matches(normalized, self.excluded_hosts):
            return False
        if not self.allowed_hosts:
            return False
        return _host_matches(normalized, self.allowed_hosts)

    def allows_method(self, method: str) -> bool:
        if not self.allowed_methods:
            return True
        return method.strip().upper() in {m.strip().upper() for m in self.allowed_methods}

    def permits_active_testing(self) -> bool:
        """Whether active testing is opted in. Independent of host allow-lists."""
        return self.allow_active_testing


def _host_matches(host: str, patterns: tuple[str, ...]) -> bool:
    from app.security_testing.target import hostname_matches

    return hostname_matches(host, patterns)
