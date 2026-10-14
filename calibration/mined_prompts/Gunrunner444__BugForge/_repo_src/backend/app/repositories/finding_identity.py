"""Authoritative persistent finding identity.

The dedicated ``security_findings.finding_key`` column is the identity. A
value stored only inside ``intelligence_json`` is not a usable key. Duplicate
legacy rows keep their history but lose the key so a later scan cannot
resurrect a second logical finding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FindingKeyRecord:
    id: Any
    project_id: Any
    created_at: Any
    intelligence_json: str
    finding_key: str | None = None


def json_finding_key(intelligence_json: str | None) -> str:
    try:
        loaded = json.loads(intelligence_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(loaded, dict):
        return ""
    return str(loaded.get("finding_key") or "").strip()


def intelligence_with_column_key(intelligence_json: str | None, finding_key: str | None) -> str:
    """Rewrite JSON so it cannot disagree with the dedicated column."""
    try:
        loaded = json.loads(intelligence_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    key = (finding_key or "").strip()
    if key:
        loaded["finding_key"] = key
    else:
        loaded.pop("finding_key", None)
    return json.dumps(loaded, sort_keys=True)


def canonicalize_finding_key_rows(rows: list[FindingKeyRecord]) -> list[FindingKeyRecord]:
    """Keep one canonical row per ``(project_id, key)``; strip the rest.

    Canonical choice is deterministic: earliest ``created_at``, then ``id``.
    Duplicate rows remain in the table but ``finding_key`` is NULL and the
    JSON payload no longer presents the old key.
    """
    ordered = sorted(rows, key=lambda row: (_sort_ts(row.created_at), str(row.id)))
    seen: set[tuple[Any, str]] = set()
    updated: list[FindingKeyRecord] = []
    for row in ordered:
        key = (row.finding_key or "").strip() or json_finding_key(row.intelligence_json)
        if not key:
            updated.append(
                FindingKeyRecord(
                    id=row.id,
                    project_id=row.project_id,
                    created_at=row.created_at,
                    intelligence_json=intelligence_with_column_key(row.intelligence_json, None),
                    finding_key=None,
                )
            )
            continue
        pair = (row.project_id, key)
        if pair in seen:
            updated.append(
                FindingKeyRecord(
                    id=row.id,
                    project_id=row.project_id,
                    created_at=row.created_at,
                    intelligence_json=intelligence_with_column_key(row.intelligence_json, None),
                    finding_key=None,
                )
            )
            continue
        seen.add(pair)
        updated.append(
            FindingKeyRecord(
                id=row.id,
                project_id=row.project_id,
                created_at=row.created_at,
                intelligence_json=intelligence_with_column_key(row.intelligence_json, key),
                finding_key=key,
            )
        )
    return updated


def _sort_ts(value: Any) -> str:
    return "" if value is None else str(value)
