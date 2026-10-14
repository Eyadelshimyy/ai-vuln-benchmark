"""Typed research-lead relationships.

Revision ID: 026
Revises: 025
Create Date: 2026-09-25 00:00:00.000000
"""

import json
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "026"
down_revision: str | None = "025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("research_leads", sa.Column("hypothesis_ids", sa.JSON(), nullable=True))
    op.add_column("research_leads", sa.Column("finding_ids", sa.JSON(), nullable=True))
    op.add_column("research_leads", sa.Column("semantic_node_ids", sa.JSON(), nullable=True))
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT id, related_ids, evidence_ids, observation_ids, chain_ids FROM research_leads"
        )
    )
    for row in rows:
        related = _json_list(row.related_ids)
        evidence = _json_list(row.evidence_ids)
        observations = _json_list(row.observation_ids)
        chains = _json_list(row.chain_ids)
        hypotheses = [
            item
            for item in related
            if item and item not in set(evidence) | set(observations) | set(chains)
        ]
        connection.execute(
            sa.text("UPDATE research_leads SET hypothesis_ids = :hypotheses WHERE id = :id"),
            {"hypotheses": json.dumps(hypotheses), "id": row.id},
        )


def _json_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def downgrade() -> None:
    op.drop_column("research_leads", "semantic_node_ids")
    op.drop_column("research_leads", "finding_ids")
    op.drop_column("research_leads", "hypothesis_ids")
