"""Adaptive research orchestration state.

Revision ID: 027
Revises: 026
Create Date: 2026-10-06 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "027"
down_revision: str | None = "026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "discovery_orchestration_campaigns",
        sa.Column("campaign_id", sa.String(64), primary_key=True),
        sa.Column("project", sa.String(256), nullable=False, server_default=""),
        sa.Column("target_digest", sa.String(32), nullable=False, server_default=""),
        sa.Column("source_snapshot", sa.String(128), nullable=False, server_default=""),
        sa.Column("compiler_configuration", sa.String(128), nullable=False, server_default=""),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("orchestrator_version", sa.String(32), nullable=False, server_default=""),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("stop_reason", sa.String(48), nullable=False, server_default=""),
        sa.Column("round", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("state_hash", sa.String(64), nullable=False),
        sa.Column("document", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_discovery_orchestration_campaigns_project",
        "discovery_orchestration_campaigns",
        ["project"],
    )
    op.create_index(
        "ix_discovery_orchestration_campaigns_target_digest",
        "discovery_orchestration_campaigns",
        ["target_digest"],
    )
    op.create_table(
        "discovery_orchestration_decisions",
        sa.Column("decision_id", sa.String(64), primary_key=True),
        sa.Column(
            "campaign_id",
            sa.String(64),
            sa.ForeignKey("discovery_orchestration_campaigns.campaign_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("round", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("record_hash", sa.String(32), nullable=False),
        sa.Column("document", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("campaign_id", "sequence", name="uq_orchestration_decision_sequence"),
    )
    op.create_index(
        "ix_discovery_orchestration_decisions_campaign_id",
        "discovery_orchestration_decisions",
        ["campaign_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_discovery_orchestration_decisions_campaign_id",
        table_name="discovery_orchestration_decisions",
    )
    op.drop_table("discovery_orchestration_decisions")
    op.drop_index(
        "ix_discovery_orchestration_campaigns_target_digest",
        table_name="discovery_orchestration_campaigns",
    )
    op.drop_index(
        "ix_discovery_orchestration_campaigns_project",
        table_name="discovery_orchestration_campaigns",
    )
    op.drop_table("discovery_orchestration_campaigns")
