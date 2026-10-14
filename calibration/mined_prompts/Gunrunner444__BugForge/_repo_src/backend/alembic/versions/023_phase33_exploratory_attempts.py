"""Persist exploratory sandbox attempts.

Revision ID: 023
Revises: 022
Create Date: 2026-09-23 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "023"
down_revision: str | None = "022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_exploratory_attempts",
        sa.Column("attempt_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("hypothesis_id", sa.String(length=64), nullable=False),
        sa.Column("parent_attempt_id", sa.String(length=64), nullable=False),
        sa.Column("candidate_id", sa.String(length=64), nullable=False),
        sa.Column("test_hash", sa.String(length=64), nullable=False),
        sa.Column("target_file", sa.Text(), nullable=False),
        sa.Column("target_symbol", sa.String(length=255), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("framework", sa.String(length=32), nullable=False),
        sa.Column("test_code", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("expected_behavior", sa.Text(), nullable=False),
        sa.Column("oracle", sa.String(length=40), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("repository_snapshot", sa.String(length=64), nullable=False),
        sa.Column("repository_commit", sa.String(length=64), nullable=False),
        sa.Column("execution_profile", sa.String(length=64), nullable=False),
        sa.Column("command_identity", sa.JSON(), nullable=True),
        sa.Column("classification", sa.String(length=32), nullable=False),
        sa.Column("stdout_summary", sa.Text(), nullable=False),
        sa.Column("stderr_summary", sa.Text(), nullable=False),
        sa.Column("artifacts", sa.JSON(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=False),
        sa.Column("timeout", sa.Boolean(), nullable=False),
        sa.Column("executed", sa.Boolean(), nullable=False),
        sa.Column("meaningful", sa.Boolean(), nullable=False),
        sa.Column("follow_up_recommended", sa.Boolean(), nullable=False),
        sa.Column("repeat_classification", sa.String(length=64), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False),
        sa.Column("follow_up", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("attempt_id"),
    )
    op.create_index(
        "ix_research_exploratory_attempts_session_id",
        "research_exploratory_attempts",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_exploratory_attempts_session_id", table_name="research_exploratory_attempts"
    )
    op.drop_table("research_exploratory_attempts")
