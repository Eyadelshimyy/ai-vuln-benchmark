"""Phase 20 persistent finding identity.

Revision ID: 022
Revises: 021
Create Date: 2026-09-22 00:00:00.000000

The duplicate-key cleanup is copied here on purpose. Historical migrations
must not import live application helpers, because those helpers can change
after this revision has already been applied to a database.
"""

import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa

from alembic import op

revision: str = "022"
down_revision: str | None = "021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("security_findings", sa.Column("finding_key", sa.Text(), nullable=True))
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, project_id, created_at, intelligence_json FROM security_findings")
    )
    records = [
        {
            "id": row[0],
            "project_id": row[1],
            "created_at": row[2],
            "intelligence_json": row[3] or "{}",
            "finding_key": None,
        }
        for row in rows
    ]
    for updated in canonicalize_legacy_finding_keys(records):
        conn.execute(
            sa.text(
                "UPDATE security_findings "
                "SET finding_key = :key, intelligence_json = :intel "
                "WHERE id = :id"
            ),
            {
                "key": updated["finding_key"],
                "intel": updated["intelligence_json"],
                "id": updated["id"],
            },
        )
    op.create_index(
        "ix_security_findings_finding_key",
        "security_findings",
        ["finding_key"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_security_findings_project_finding_key",
        "security_findings",
        ["project_id", "finding_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_security_findings_project_finding_key", "security_findings", type_="unique"
    )
    op.drop_index("ix_security_findings_finding_key", table_name="security_findings")
    op.drop_column("security_findings", "finding_key")


def canonicalize_legacy_finding_keys(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the earliest row for each ``(project_id, key)`` and strip the rest."""
    ordered = sorted(rows, key=lambda row: (str(row.get("created_at") or ""), str(row.get("id"))))
    seen: set[tuple[Any, str]] = set()
    updated: list[dict[str, Any]] = []
    for row in ordered:
        key = str(row.get("finding_key") or "").strip() or _json_key(str(row.get("intelligence_json") or ""))
        intel = _json_object(str(row.get("intelligence_json") or ""))
        if not key:
            intel.pop("finding_key", None)
            updated.append({**row, "finding_key": None, "intelligence_json": json.dumps(intel, sort_keys=True)})
            continue
        pair = (row.get("project_id"), key)
        if pair in seen:
            intel.pop("finding_key", None)
            updated.append({**row, "finding_key": None, "intelligence_json": json.dumps(intel, sort_keys=True)})
            continue
        seen.add(pair)
        intel["finding_key"] = key
        updated.append({**row, "finding_key": key, "intelligence_json": json.dumps(intel, sort_keys=True)})
    return updated


def _json_key(raw: str) -> str:
    loaded = _json_object(raw)
    return str(loaded.get("finding_key") or "").strip()


def _json_object(raw: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return dict(loaded)
