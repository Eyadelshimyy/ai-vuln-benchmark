"""Scope authorization helpers for future active operations.

Being in scope does not authorize active testing. Browser navigation,
fuzzing, and scanner active scans must call :func:`require_active_testing`
before generating traffic. Passive collectors may use :func:`require_in_scope`
or :func:`filter_in_scope` without opting into active testing.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from app.domain.http import HttpExchange
from app.domain.scope import ScopeConstraint
from app.plugins.errors import ActiveTestingNotPermittedError, OutOfScopeError


def host_from_target(target: str) -> str:
    raw = target.strip()
    if "://" in raw:
        parsed = urlparse(raw)
        return (parsed.hostname or "").lower()
    return raw.split("/", 1)[0].split(":", 1)[0].lower()


def target_is_in_scope(scope: ScopeConstraint, target: str, *, method: str | None = None) -> bool:
    host = host_from_target(target)
    if not scope.allows_host(host):
        return False
    if method is not None and not scope.allows_method(method):
        return False
    return True


def require_in_scope(scope: ScopeConstraint, target: str, *, method: str | None = None) -> None:
    if not target_is_in_scope(scope, target, method=method):
        raise OutOfScopeError(target)


def require_active_testing(
    scope: ScopeConstraint, target: str, *, method: str | None = None
) -> None:
    require_in_scope(scope, target, method=method)
    if not scope.permits_active_testing():
        raise ActiveTestingNotPermittedError(target)


def filter_in_scope(
    scope: ScopeConstraint, exchanges: Iterable[HttpExchange]
) -> tuple[HttpExchange, ...]:
    """Keep captured HTTP evidence whose URL/method is in scope. Does not send traffic."""
    return tuple(
        exchange
        for exchange in exchanges
        if target_is_in_scope(scope, exchange.url, method=exchange.method)
    )
