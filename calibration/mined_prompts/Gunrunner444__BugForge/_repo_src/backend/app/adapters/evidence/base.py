"""Evidence collector contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.domain.evidence import Evidence, EvidenceKind, EvidenceSource


class EvidenceCollector(ABC):
    """Convert one kind of artifact into domain evidence."""

    @property
    @abstractmethod
    def collector_id(self) -> str: ...

    @property
    @abstractmethod
    def kinds(self) -> frozenset[EvidenceKind]: ...

    @abstractmethod
    def collect(self, source: EvidenceSource) -> Sequence[Evidence]: ...
