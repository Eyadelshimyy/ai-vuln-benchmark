"""Shared portal knowledge: local registry + tracked seed + built-in host rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from jobbot.config import JobbotConfig
from jobbot.portals.detect import AtsKind, detect_ats
from jobbot.portals.registry import (
    PortalEntry,
    default_portals_path,
    domain_from_url,
    load_registry,
)
from jobbot.workspace import repo_root


class PortalKnowledgeSource(StrEnum):
    LOCAL = "local"
    SEED = "seed"
    BUILTIN = "builtin"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PortalKnowledge:
    """Whether a hard job link's host is a known employment platform."""

    url: str
    domain: str
    ats_kind: AtsKind
    source: PortalKnowledgeSource
    entry: PortalEntry | None = None

    @property
    def known(self) -> bool:
        return self.source != PortalKnowledgeSource.UNKNOWN


def seed_portals_path() -> Path:
    """Tracked seed of ATS domains (`data/portals.example.yaml` in the checkout)."""
    return repo_root() / "data" / "portals.example.yaml"


def lookup_portal(config: JobbotConfig, url: str) -> PortalKnowledge:
    """Resolve a URL against local portals, the distributed seed, then host rules."""
    raw = url.strip()
    if not raw:
        return PortalKnowledge(
            url=raw,
            domain="",
            ats_kind=AtsKind.UNKNOWN,
            source=PortalKnowledgeSource.UNKNOWN,
        )
    if not raw.lower().startswith(("http://", "https://", "mailto:")):
        raw = "https://" + raw
    domain = domain_from_url(raw)
    kind = detect_ats(raw)

    local = load_registry(default_portals_path(config.root))
    entry = local.find(domain) if domain else None
    if entry is not None:
        return PortalKnowledge(
            url=raw,
            domain=domain,
            ats_kind=entry.ats_kind if entry.ats_kind != AtsKind.UNKNOWN else kind,
            source=PortalKnowledgeSource.LOCAL,
            entry=entry,
        )

    seed_path = seed_portals_path()
    if seed_path.is_file() and domain:
        seed = load_registry(seed_path)
        entry = seed.find(domain)
        if entry is not None:
            return PortalKnowledge(
                url=raw,
                domain=domain,
                ats_kind=entry.ats_kind if entry.ats_kind != AtsKind.UNKNOWN else kind,
                source=PortalKnowledgeSource.SEED,
                entry=entry,
            )

    if kind != AtsKind.UNKNOWN and domain:
        return PortalKnowledge(
            url=raw,
            domain=domain,
            ats_kind=kind,
            source=PortalKnowledgeSource.BUILTIN,
        )

    return PortalKnowledge(
        url=raw,
        domain=domain,
        ats_kind=kind,
        source=PortalKnowledgeSource.UNKNOWN,
    )
