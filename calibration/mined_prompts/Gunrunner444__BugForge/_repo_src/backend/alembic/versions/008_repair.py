"""Add repair_sessions and patch_candidates tables (Phase 7)

Revision ID: 008
Revises: 007
Create Date: 2024-01-08 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create repair_sessions first (without best_candidate_id FK to avoid cycle)
    op.create_table(
        "repair_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("debugging_session_id", sa.Uuid(), nullable=True),
        sa.Column("hypothesis_id", sa.Uuid(), nullable=True),
        sa.Column("reproduction_session_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("total_candidates", sa.Integer(), nullable=False),
        sa.Column("best_candidate_id", sa.Uuid(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["debugging_session_id"], ["debugging_sessions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["hypothesis_id"], ["debugging_hypotheses.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["reproduction_session_id"],
            ["bug_reproduction_sessions.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_repair_sessions_project_id", "repair_sessions", ["project_id"])

    op.create_table(
        "patch_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("patch_provider", sa.String(50), nullable=True),
        sa.Column("patch_model", sa.String(100), nullable=True),
        sa.Column("patch_plan", sa.Text(), nullable=True),
        sa.Column("patch_diff", sa.Text(), nullable=True),
        sa.Column("changed_files_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("validation_status", sa.String(50), nullable=True),
        sa.Column("validation_error", sa.Text(), nullable=True),
        sa.Column("pre_patch_reproduced", sa.Boolean(), nullable=True),
        sa.Column("pre_patch_stdout", sa.Text(), nullable=True),
        sa.Column("post_patch_reproduced", sa.Boolean(), nullable=True),
        sa.Column("post_patch_stdout", sa.Text(), nullable=True),
        sa.Column("bug_fixed", sa.Boolean(), nullable=True),
        sa.Column("existing_tests_total", sa.Integer(), nullable=True),
        sa.Column("existing_tests_passed", sa.Integer(), nullable=True),
        sa.Column("existing_tests_failed", sa.Integer(), nullable=True),
        sa.Column("no_regressions", sa.Boolean(), nullable=True),
        sa.Column("regression_count", sa.Integer(), nullable=False),
        sa.Column("new_static_findings", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("disposition", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["session_id"], ["repair_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_patch_candidates_session_id", "patch_candidates", ["session_id"])

    # Now add the circular FK from repair_sessions.best_candidate_id → patch_candidates
    op.create_foreign_key(
        "fk_repair_sessions_best_candidate",
        "repair_sessions",
        "patch_candidates",
        ["best_candidate_id"],
        ["id"],
        ondelete="SET NULL",
        use_alter=True,
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_repair_sessions_best_candidate", "repair_sessions", type_="foreignkey"
    )
    op.drop_table("patch_candidates")
    op.drop_table("repair_sessions")
