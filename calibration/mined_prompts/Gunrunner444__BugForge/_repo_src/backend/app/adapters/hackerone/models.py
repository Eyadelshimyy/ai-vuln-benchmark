"""HackerOne-normalized records. These are adapter types, not core domain imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.security_testing.target import AssetType


class ProgramSyncStatus(StrEnum):
    NEVER = "never"
    OK = "ok"
    ERROR = "error"
    INCOMPLETE = "incomplete"


class ScopeMode(StrEnum):
    CLOSED = "closed"
    OPEN = "open"


HACKERONE_ASSET_MAP: dict[str, AssetType] = {
    "domain": AssetType.DOMAIN,
    "url": AssetType.URL,
    "wildcard": AssetType.WILDCARD,
    "ip_address": AssetType.IP,
    "ip": AssetType.IP,
    "cidr": AssetType.CIDR,
    "source_code": AssetType.SOURCE_CODE,
    "executable": AssetType.EXECUTABLE,
    "android_app": AssetType.ANDROID_APP,
    "ios_app": AssetType.IOS_APP,
    "hardware": AssetType.HARDWARE,
    "other": AssetType.OTHER_ASSET,
    "ai_model": AssetType.AI_MODEL,
    "other_asset": AssetType.OTHER_ASSET,
}


def asset_type_from_hackerone(raw: str) -> AssetType:
    key = (raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "hardware_iot": AssetType.HARDWARE,
        "hardware/iot": AssetType.HARDWARE,
        "wildcard": AssetType.WILDCARD,
        "ai_model": AssetType.AI_MODEL,
        "ai-model": AssetType.AI_MODEL,
    }
    if key in aliases:
        return aliases[key]
    return HACKERONE_ASSET_MAP.get(key, AssetType.UNSUPPORTED)


@dataclass(frozen=True)
class StructuredScopeRecord:
    id: str
    asset_type_raw: str
    asset_type: AssetType
    asset_identifier: str
    instruction: str = ""
    eligible_for_bounty: bool = False
    eligible_for_submission: bool = True
    reference: str | None = None
    original: dict[str, Any] = field(default_factory=dict)

    @property
    def numeric_id(self) -> int | None:
        return parse_hackerone_id(self.id)


@dataclass(frozen=True)
class ScopeExclusionRecord:
    """HackerOne report-category / reward exclusion. Not a target deny-list."""

    id: str
    category: str
    details: str
    created_at: str | None = None
    updated_at: str | None = None
    original: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeaknessRecord:
    """Program-specific HackerOne weakness. ``id`` is the numeric API id."""

    id: int
    name: str
    description: str = ""
    external_id: str = ""
    original: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeSnapshot:
    """Exact structured-scope snapshot that authorized a draft."""

    program_handle: str
    scope_version: int
    structured_scope_id: int | None
    asset_identifier: str
    eligible_for_submission: bool
    eligible_for_bounty: bool
    in_scope: bool
    matched_instructions: str = ""
    snapshot_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "program_handle": self.program_handle,
            "scope_version": self.scope_version,
            "structured_scope_id": self.structured_scope_id,
            "asset_identifier": self.asset_identifier,
            "eligible_for_submission": self.eligible_for_submission,
            "eligible_for_bounty": self.eligible_for_bounty,
            "in_scope": self.in_scope,
            "matched_instructions": self.matched_instructions,
        }


@dataclass
class HackerOneProgram:
    handle: str
    name: str = ""
    program_id: str | None = None
    program_url: str | None = None
    fetched_at: datetime | None = None
    sync_status: ProgramSyncStatus = ProgramSyncStatus.NEVER
    error: str | None = None
    scope_mode: ScopeMode = ScopeMode.CLOSED
    offers_bounties: bool | None = None
    requires_severity: bool = True
    instructions: str = ""
    structured_scopes: tuple[StructuredScopeRecord, ...] = ()
    exclusions: tuple[ScopeExclusionRecord, ...] = ()
    weaknesses: tuple[WeaknessRecord, ...] = ()
    open_scope_policy: str = ""
    open_scope_acknowledged: bool = False
    active_testing_approved: bool = False
    scope_version: int = 0
    scope_sync_complete: bool = False
    scope_count: int = 0
    last_scope_id: str | None = None
    continuation_state: str | None = None
    scope_content_hash: str = ""
    weaknesses_synced_at: datetime | None = None
    scope_pages_fetched: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "id": self.program_id,
            "name": self.name,
            "program_url": self.program_url,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "sync_status": self.sync_status.value,
            "error": self.error,
            "scope_mode": self.scope_mode.value,
            "offers_bounties": self.offers_bounties,
            "requires_severity": self.requires_severity,
            "scope_count": len(self.structured_scopes),
            "exclusion_count": len(self.exclusions),
            "weakness_count": len(self.weaknesses),
            "scope_version": self.scope_version,
            "scope_sync_complete": self.scope_sync_complete,
            "last_scope_id": self.last_scope_id,
            "active_testing_approved": self.active_testing_approved,
            "open_scope_acknowledged": self.open_scope_acknowledged,
            "scope_content_hash": self.scope_content_hash,
            "weaknesses_synced_at": (
                self.weaknesses_synced_at.isoformat() if self.weaknesses_synced_at else None
            ),
            "scope_pages_fetched": self.scope_pages_fetched,
        }


def attributes(node: object) -> dict[str, Any]:
    if isinstance(node, dict):
        attrs = node.get("attributes")
        if isinstance(attrs, dict):
            return attrs
    return {}


def node_id(node: object) -> str:
    if isinstance(node, dict):
        return str(node.get("id") or "")
    return ""


def parse_hackerone_id(value: object) -> int | None:
    """Parse a HackerOne numeric resource id. CWE strings are never valid ids."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value).strip()
    if not text.isdigit():
        return None
    parsed = int(text)
    return parsed if parsed > 0 else None
