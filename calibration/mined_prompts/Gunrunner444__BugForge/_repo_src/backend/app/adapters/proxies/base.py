"""HTTP proxy adapter contract for ingesting captured traffic as evidence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.adapters.scope.authorization import filter_in_scope
from app.domain.http import HttpExchange
from app.domain.scope import ScopeConstraint
from app.plugins.errors import AdapterNotImplementedError


class ProxyAdapter(ABC):
    @property
    @abstractmethod
    def adapter_id(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    def fetch_exchanges(self, *, scope: ScopeConstraint) -> Sequence[HttpExchange]:
        """Return previously captured HTTP evidence filtered to ``scope``.

        Must not send live requests. Out-of-scope captures are dropped rather
        than replayed.
        """
        return filter_in_scope(scope, self._fetch_captured_exchanges())

    def _fetch_captured_exchanges(self) -> Sequence[HttpExchange]:
        raise AdapterNotImplementedError(
            f"{self.adapter_id} proxy ingestion is reserved for a later phase."
        )
