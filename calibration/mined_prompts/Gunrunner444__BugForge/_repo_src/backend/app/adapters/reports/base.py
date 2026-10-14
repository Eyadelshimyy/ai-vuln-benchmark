"""Reporting adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.evidence import EvidenceBundle
from app.domain.findings import SecurityFinding
from app.domain.reports import SecurityReport
from app.plugins.errors import AdapterNotImplementedError


class ReportProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str: ...

    @abstractmethod
    def render(
        self,
        findings: Sequence[SecurityFinding],
        *,
        evidence: EvidenceBundle | None = None,
    ) -> SecurityReport: ...

    def submit(self, report: SecurityReport) -> str:
        """Remote submission. Local providers must raise, never upload."""
        raise AdapterNotImplementedError(
            f"{self.provider_id} remote submission is reserved for a later phase."
        )
