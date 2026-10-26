"""Bulk fact-load results — visible failures, ordered flatten."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models.exposure import ExposureFactNormalized


@dataclass(frozen=True)
class LoadError:
    dataset_id: str
    message: str


@dataclass
class FactBatch:
    """Result of loading many EDMs in parallel."""

    by_id: dict[str, list[ExposureFactNormalized]] = field(default_factory=dict)
    errors: list[LoadError] = field(default_factory=list)

    def flatten(self, order: Sequence[str] | None = None) -> list[ExposureFactNormalized]:
        """Concat facts; preserve ``order`` when given, else sorted keys."""
        out: list[ExposureFactNormalized] = []
        keys = list(order) if order is not None else sorted(self.by_id.keys())
        seen: set[str] = set()
        for k in keys:
            if k in seen:
                continue
            seen.add(k)
            out.extend(self.by_id.get(k) or [])
        # Any keys not in order (shouldn't happen) — append sorted remainder.
        for k in sorted(self.by_id.keys()):
            if k not in seen:
                out.extend(self.by_id.get(k) or [])
        return out

    @property
    def failed_dataset_ids(self) -> list[str]:
        return [e.dataset_id for e in self.errors]
