"""Phase 8 research findings, memory, checkpoints, identities, and tool-call quality.

Revision ID: 018
Revises: 017
Create Date: 2026-09-20 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "018"
down_revision: str | None = "017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "security_research_tool_calls",
        sa.Column("execution_state", sa.String(length=64), nullable=False, server_default=""),
    )
    op.add_column(
        "security_research_tool_calls",
        sa.Column("result_quality", sa.String(length=64), nullable=False, server_default=""),
    )
    op.create_table(
        "research_findings",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("hypothesis_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("vulnerability_class", sa.String(length=128), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("verification_state", sa.String(length=64), nullable=False),
        sa.Column("evidence_ids", sa.JSON(), nullable=True),
        sa.Column("reproduction_ids", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("impact", sa.Text(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_findings_session_id", "research_findings", ["session_id"])
    op.create_table(
        "research_memory",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_memory_project_id", "research_memory", ["project_id"])
    op.create_index("ix_research_memory_session_id", "research_memory", ["session_id"])
    op.create_table(
        "research_checkpoints",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_checkpoints_session_id", "research_checkpoints", ["session_id"])
    op.create_table(
        "research_identities",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("cookies", sa.JSON(), nullable=True),
        sa.Column("storage", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_research_identities_session_id", "research_identities", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_research_identities_session_id", table_name="research_identities")
    op.drop_table("research_identities")
    op.drop_index("ix_research_checkpoints_session_id", table_name="research_checkpoints")
    op.drop_table("research_checkpoints")
    op.drop_index("ix_research_memory_session_id", table_name="research_memory")
    op.drop_index("ix_research_memory_project_id", table_name="research_memory")
    op.drop_table("research_memory")
    op.drop_index("ix_research_findings_session_id", table_name="research_findings")
    op.drop_table("research_findings")
    op.drop_column("security_research_tool_calls", "result_quality")
    op.drop_column("security_research_tool_calls", "execution_state")
