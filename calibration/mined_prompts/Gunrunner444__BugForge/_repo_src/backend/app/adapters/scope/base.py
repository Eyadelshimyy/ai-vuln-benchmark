"""Scope provider contract for authorized testing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.adapters.scope.authorization import (
    require_active_testing,
    require_in_scope,
    target_is_in_scope,
)
from app.domain.scope import ScopeConstraint
from app.plugins.errors import AdapterNotImplementedError


class ScopeProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str: ...

    @abstractmethod
    def get_scope(self) -> ScopeConstraint: ...

    def is_in_scope(self, target: str, *, method: str | None = None) -> bool:
        """Return True when the host (and optional method) is in the allow-list.

        This does **not** authorize active testing. Use
        :meth:`is_active_testing_permitted` or :meth:`require_active_testing`.
        """
        return target_is_in_scope(self.get_scope(), target, method=method)

    def is_method_permitted(self, method: str) -> bool:
        return self.get_scope().allows_method(method)

    def is_active_testing_permitted(self, target: str, *, method: str | None = None) -> bool:
        scope = self.get_scope()
        return scope.permits_active_testing() and target_is_in_scope(scope, target, method=method)

    def require_in_scope(self, target: str, *, method: str | None = None) -> None:
        require_in_scope(self.get_scope(), target, method=method)

    def require_active_testing(self, target: str, *, method: str | None = None) -> None:
        require_active_testing(self.get_scope(), target, method=method)

    def fetch_remote_scope(self) -> ScopeConstraint:
        raise AdapterNotImplementedError(
            f"{self.provider_id} remote scope retrieval is reserved for a later phase."
        )
