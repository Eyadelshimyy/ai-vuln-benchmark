"""Combine independent collectors into one evidence bundle."""

from __future__ import annotations

from collections.abc import Sequence

from app.adapters.evidence.base import EvidenceCollector
from app.domain.evidence import Evidence, EvidenceBundle, EvidenceKind, EvidenceSource


class CompositeEvidenceCollector(EvidenceCollector):
    """Run multiple collectors and merge their output.

    Collectors are independent: failure of one does not discard the others.
    The composite never promotes a hypothesis to a verified finding.
    """

    def __init__(self, collectors: Sequence[EvidenceCollector] | None = None) -> None:
        self._collectors: list[EvidenceCollector] = list(collectors or [])

    @property
    def collector_id(self) -> str:
        return "composite"

    @property
    def kinds(self) -> frozenset[EvidenceKind]:
        kinds: set[EvidenceKind] = set()
        for collector in self._collectors:
            kinds.update(collector.kinds)
        return frozenset(kinds)

    def add(self, collector: EvidenceCollector) -> None:
        self._collectors.append(collector)

    def collect(self, source: EvidenceSource) -> list[Evidence]:
        items: list[Evidence] = []
        for collector in self._collectors:
            items.extend(collector.collect(source))
        return items

    def collect_bundle(self, source: EvidenceSource) -> EvidenceBundle:
        return EvidenceBundle.from_items(self.collect(source))
