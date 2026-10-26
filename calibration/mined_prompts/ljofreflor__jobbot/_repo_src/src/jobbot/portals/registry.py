"""Local registry of recruitment portals (where the user applies / is registered)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from jobbot.portals.detect import AtsKind, detect_ats


class PortalEntry(BaseModel):
    domain: str = Field(min_length=1)
    ats_kind: AtsKind = AtsKind.UNKNOWN
    registered: bool = False
    last_seen: datetime | None = None
    notes: str | None = None
    example_url: str | None = None


class PortalRegistry(BaseModel):
    portals: list[PortalEntry] = Field(default_factory=list)

    def find(self, domain: str) -> PortalEntry | None:
        key = domain.casefold()
        for entry in self.portals:
            if entry.domain.casefold() == key:
                return entry
        return None

    def upsert(
        self,
        *,
        domain: str,
        ats_kind: AtsKind | None = None,
        registered: bool | None = None,
        notes: str | None = None,
        example_url: str | None = None,
        seen: bool = True,
    ) -> PortalEntry:
        entry = self.find(domain)
        kind = ats_kind or (detect_ats(example_url or domain) if example_url else AtsKind.UNKNOWN)
        if entry is None:
            entry = PortalEntry(
                domain=domain,
                ats_kind=kind if kind != AtsKind.UNKNOWN else detect_ats(f"https://{domain}/"),
                registered=bool(registered),
                notes=notes,
                example_url=example_url,
                last_seen=datetime.now(UTC) if seen else None,
            )
            self.portals.append(entry)
            return entry
        if ats_kind is not None:
            entry.ats_kind = ats_kind
        if registered is not None:
            entry.registered = registered
        if notes is not None:
            entry.notes = notes
        if example_url is not None:
            entry.example_url = example_url
        if seen:
            entry.last_seen = datetime.now(UTC)
        return entry


def default_portals_path(root: Path | None = None) -> Path:
    """Portals YAML: config ``paths.portals`` when available, else ``data/portals.yaml``."""
    if root is None:
        from jobbot.config import load_config

        return load_config().portals_path
    from jobbot.config import load_config

    return load_config(root=root).portals_path


def load_registry(path: Path) -> PortalRegistry:
    if not path.is_file():
        return PortalRegistry()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, list):
        return PortalRegistry(portals=raw)
    return PortalRegistry.model_validate(raw)


def save_registry(registry: PortalRegistry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.model_dump(mode="json")
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def domain_from_url(url: str) -> str:
    from urllib.parse import urlparse

    raw = url if "://" in url else f"https://{url}"
    host = (urlparse(raw).hostname or "").lower()
    return host.removeprefix("www.")
