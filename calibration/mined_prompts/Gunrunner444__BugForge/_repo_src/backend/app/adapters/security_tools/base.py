"""Security tool adapter contract.

Live scanning of external targets is out of scope for this phase. Adapters
exist so a future security engine can collect evidence through one interface.
Active scans must go through :meth:`active_scan`, which enforces authorization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from app.adapters.scope.authorization import require_active_testing
from app.domain.evidence import Evidence
from app.domain.scope import ScopeConstraint
from app.plugins.errors import UnsupportedCapabilityError


class SecurityToolCapability(StrEnum):
    PASSIVE_EVIDENCE = "passive_evidence"
    ACTIVE_SCAN = "active_scan"


class SecurityToolAdapter(ABC):
    """Vendor-neutral security tool boundary."""

    @property
    @abstractmethod
    def tool_id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the backing tool is installed and configured."""

    def capabilities(self) -> frozenset[SecurityToolCapability]:
        return frozenset()

    def collect_passive_evidence(self, *, scope: ScopeConstraint) -> Sequence[Evidence]:
        """Collect previously captured or local evidence. Must not attack targets."""
        return self._collect_passive_evidence(scope=scope)

    def _collect_passive_evidence(self, *, scope: ScopeConstraint) -> Sequence[Evidence]:
        raise UnsupportedCapabilityError(
            self.tool_id,
            SecurityToolCapability.PASSIVE_EVIDENCE,
            detail="Passive evidence collection is not implemented in this phase.",
        )

    def active_scan(self, *, scope: ScopeConstraint, target: str) -> Sequence[Evidence]:
        require_active_testing(scope, target)
        return self._active_scan(scope=scope, target=target)

    def _active_scan(self, *, scope: ScopeConstraint, target: str) -> Sequence[Evidence]:
        raise UnsupportedCapabilityError(
            self.tool_id,
            SecurityToolCapability.ACTIVE_SCAN,
            detail="Active scanning is reserved for a later phase and must stay in-scope.",
        )
