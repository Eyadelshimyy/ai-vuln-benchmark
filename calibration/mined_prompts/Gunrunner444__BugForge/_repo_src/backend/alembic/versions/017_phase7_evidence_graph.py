"""Phase 7 evidence graph persistence, privilege snapshots, and termination.

Revision ID: 017
Revises: 016
Create Date: 2026-09-20 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "017"
down_revision: str | None = "016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "security_research_sessions",
        sa.Column("termination_reason", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "security_research_sessions", sa.Column("privilege_snapshot", sa.JSON(), nullable=True)
    )
    op.add_column(
        "security_research_sessions", sa.Column("disabled_tools", sa.JSON(), nullable=True)
    )
    op.add_column(
        "security_research_sessions",
        sa.Column("repo_root", sa.Text(), nullable=False, server_default="."),
    )
    op.add_column(
        "security_research_sessions",
        sa.Column("stopped", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("security_research_sessions", sa.Column("exchanges", sa.JSON(), nullable=True))

    op.add_column(
        "security_research_hypotheses",
        sa.Column("severity", sa.String(length=32), nullable=False, server_default="medium"),
    )
    op.add_column(
        "security_research_hypotheses",
        sa.Column("impact", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "security_research_hypotheses",
        sa.Column(
            "reproducibility", sa.String(length=32), nullable=False, server_default="unknown"
        ),
    )
    op.add_column(
        "security_research_hypotheses",
        sa.Column("evidence_strength", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_table(
        "research_evidence_node",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("provenance", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_research_evidence_node_session_id", "research_evidence_node", ["session_id"]
    )

    op.create_table(
        "research_evidence_edge",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("source_node", sa.String(length=64), nullable=False),
        sa.Column("destination_node", sa.String(length=64), nullable=False),
        sa.Column("relation", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_research_evidence_edge_session_id", "research_evidence_edge", ["session_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_research_evidence_edge_session_id", table_name="research_evidence_edge")
    op.drop_table("research_evidence_edge")
    op.drop_index("ix_research_evidence_node_session_id", table_name="research_evidence_node")
    op.drop_table("research_evidence_node")
    op.drop_column("security_research_hypotheses", "evidence_strength")
    op.drop_column("security_research_hypotheses", "reproducibility")
    op.drop_column("security_research_hypotheses", "impact")
    op.drop_column("security_research_hypotheses", "severity")
    op.drop_column("security_research_sessions", "exchanges")
    op.drop_column("security_research_sessions", "stopped")
    op.drop_column("security_research_sessions", "repo_root")
    op.drop_column("security_research_sessions", "disabled_tools")
    op.drop_column("security_research_sessions", "privilege_snapshot")
    op.drop_column("security_research_sessions", "termination_reason")
