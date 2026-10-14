"""Add bug_reproduction_sessions and bug_reproduction_attempts tables (Phase 6)

Revision ID: 006
Revises: 005
Create Date: 2024-01-06 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "bug_reproduction_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("debugging_session_id", sa.Uuid(), nullable=True),
        sa.Column("hypothesis_id", sa.Uuid(), nullable=True),
        sa.Column("generated_test_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("successful_attempts", sa.Integer(), nullable=False),
        sa.Column("total_attempts", sa.Integer(), nullable=False),
        sa.Column("reproducibility_rate", sa.Float(), nullable=True),
        sa.Column("final_classification", sa.String(50), nullable=True),
        sa.Column("strategy_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["debugging_session_id"], ["debugging_sessions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["hypothesis_id"], ["debugging_hypotheses.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["generated_test_id"], ["generated_tests.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bug_repro_project_id", "bug_reproduction_sessions", ["project_id"])
    op.create_index("ix_bug_repro_hypothesis_id", "bug_reproduction_sessions", ["hypothesis_id"])
    op.create_index("ix_bug_repro_debug_session_id", "bug_reproduction_sessions", ["debugging_session_id"])

    op.create_table(
        "bug_reproduction_attempts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("command", sa.Text(), nullable=True),
        sa.Column("input_description", sa.Text(), nullable=True),
        sa.Column("reproducer_code", sa.Text(), nullable=True),
        sa.Column("exit_code", sa.Integer(), nullable=True),
        sa.Column("stdout", sa.Text(), nullable=True),
        sa.Column("stderr", sa.Text(), nullable=True),
        sa.Column("traceback", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("timed_out", sa.Boolean(), nullable=False),
        sa.Column("reproduced", sa.Boolean(), nullable=False),
        sa.Column("classification", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["bug_reproduction_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bug_repro_attempts_session_id", "bug_reproduction_attempts", ["session_id"])


def downgrade() -> None:
    op.drop_table("bug_reproduction_attempts")
    op.drop_table("bug_reproduction_sessions")
