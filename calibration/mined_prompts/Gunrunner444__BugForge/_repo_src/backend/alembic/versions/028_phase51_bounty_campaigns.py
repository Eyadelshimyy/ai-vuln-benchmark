"""Persistent bounty campaigns (Phase 51 hardening).

Revision ID: 028
Revises: 027
Create Date: 2026-10-08 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "028"
down_revision: str | None = "027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "discovery_bounty_campaigns",
        sa.Column(
            "campaign_id",
            sa.String(64),
            sa.ForeignKey("discovery_orchestration_campaigns.campaign_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("operator_identity", sa.String(128), nullable=False, server_default=""),
        sa.Column("engine_version", sa.String(32), nullable=False, server_default=""),
        sa.Column("program_context", sa.String(64), nullable=False, server_default=""),
        sa.Column("manifest", sa.Text(), nullable=False),
        sa.Column("spec", sa.Text(), nullable=False),
        sa.Column("approvals", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("control", sa.String(16), nullable=False, server_default="active"),
        sa.Column("control_reason", sa.String(200), nullable=False, server_default=""),
        sa.Column("job", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("artifacts", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_discovery_bounty_campaigns_operator_identity",
        "discovery_bounty_campaigns",
        ["operator_identity"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_discovery_bounty_campaigns_operator_identity",
        table_name="discovery_bounty_campaigns",
    )
    op.drop_table("discovery_bounty_campaigns")
