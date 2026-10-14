"""Fuzzing adapter contract.

No live fuzzing engine is implemented in this phase. Public :meth:`fuzz`
always enforces scope and active-testing authorization before delegating.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from app.adapters.scope.authorization import require_active_testing
from app.domain.evidence import Evidence
from app.domain.scope import ScopeConstraint
from app.plugins.errors import OutOfScopeError, UnsupportedCapabilityError


class FuzzingTarget(StrEnum):
    PARAMETER = "parameter"
    JSON_BODY = "json_body"
    HEADER = "header"
    PATH = "path"
    API = "api"


class FuzzingAdapter(ABC):
    @property
    @abstractmethod
    def adapter_id(self) -> str: ...

    def capabilities(self) -> frozenset[FuzzingTarget]:
        return frozenset()

    def fuzz(
        self,
        *,
        target: FuzzingTarget,
        scope: ScopeConstraint,
        target_url: str = "",
    ) -> Sequence[Evidence]:
        url = target_url.strip()
        if url:
            require_active_testing(scope, url)
        else:
            if not scope.allowed_hosts:
                raise OutOfScopeError("(unspecified target)")
            require_active_testing(scope, scope.allowed_hosts[0])
        return self._fuzz(target=target, scope=scope, target_url=url)

    def _fuzz(
        self,
        *,
        target: FuzzingTarget,
        scope: ScopeConstraint,
        target_url: str,
    ) -> Sequence[Evidence]:
        raise UnsupportedCapabilityError(
            self.adapter_id,
            target,
            detail="Live fuzzing is reserved for a later phase and must stay in-scope.",
        )
