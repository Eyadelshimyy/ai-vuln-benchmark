"""In-memory / operator-supplied testing scope."""

from __future__ import annotations

from app.adapters.scope.base import ScopeProvider
from app.domain.scope import ScopeConstraint


class ManualScopeProvider(ScopeProvider):
    """Explicit allow/deny lists. Empty allow-list denies every target."""

    def __init__(self, scope: ScopeConstraint | None = None) -> None:
        self._scope = scope or ScopeConstraint()

    @property
    def provider_id(self) -> str:
        return "manual"

    def get_scope(self) -> ScopeConstraint:
        return self._scope
