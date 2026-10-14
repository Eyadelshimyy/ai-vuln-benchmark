"""Combine independent static observations into clusters."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.domain.security import VulnerabilityClass
from app.security.rules.base import SecurityObservation


@dataclass(frozen=True)
class ObservationCluster:
    """Observations that describe the same hypothesized issue."""

    vulnerability_class: VulnerabilityClass
    file_path: str
    observations: tuple[SecurityObservation, ...]

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(obs.rule_id for obs in self.observations))

    @property
    def line(self) -> int:
        return min(obs.line for obs in self.observations)

    @property
    def language(self) -> str:
        return self.observations[0].language

    @property
    def independent_rules(self) -> int:
        """Distinct (sink node, taint path) pairs, not merely distinct rule ids."""
        keys: set[tuple[str, str]] = set()
        for obs in self.observations:
            keys.add((obs.node_id or f"{obs.file_path}:{obs.line}", obs.taint_path or obs.rule_id))
        return len(keys)

    @property
    def corroborated(self) -> bool:
        return self.independent_rules >= 2


def correlate_observations(observations: list[SecurityObservation]) -> list[ObservationCluster]:
    """Group observations by vulnerability class and file.

    Multiple distinct rules hitting the same class+file become one cluster so
    the AI and finding model can treat them as linked evidence.
    """
    grouped: dict[tuple[str, str], list[SecurityObservation]] = defaultdict(list)
    for obs in observations:
        grouped[(obs.vulnerability_class.value, obs.file_path)].append(obs)
    clusters: list[ObservationCluster] = []
    for (class_name, file_path), items in grouped.items():
        items.sort(key=lambda o: (o.line, o.rule_id))
        clusters.append(
            ObservationCluster(
                vulnerability_class=VulnerabilityClass(class_name),
                file_path=file_path,
                observations=tuple(items),
            )
        )
    clusters.sort(key=lambda c: (c.file_path, c.line, c.vulnerability_class.value))
    return clusters
