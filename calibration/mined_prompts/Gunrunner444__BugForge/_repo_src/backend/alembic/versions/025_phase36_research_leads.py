"""Persistent research leads.

Revision ID: 025
Revises: 024
Create Date: 2026-09-25 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "025"
down_revision: str | None = "024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_leads",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.String(length=32), nullable=False),
        sa.Column("next_action", sa.Text(), nullable=False),
        sa.Column("kill_reason", sa.Text(), nullable=False),
        sa.Column("evidence_ids", sa.JSON(), nullable=True),
        sa.Column("observation_ids", sa.JSON(), nullable=True),
        sa.Column("related_ids", sa.JSON(), nullable=True),
        sa.Column("chain_ids", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_leads_project_id", "research_leads", ["project_id"])
    op.create_index("ix_research_leads_session_id", "research_leads", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_research_leads_session_id", table_name="research_leads")
    op.drop_index("ix_research_leads_project_id", table_name="research_leads")
    op.drop_table("research_leads")
