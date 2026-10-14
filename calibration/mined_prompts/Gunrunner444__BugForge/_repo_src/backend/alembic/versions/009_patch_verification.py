"""Add patch_verifications table (Phase 8)

Revision ID: 009
Revises: 008
Create Date: 2024-01-09 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "patch_verifications",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        # baseline
        sa.Column("baseline_reproduced", sa.Boolean(), nullable=True),
        sa.Column("baseline_reproduction_evidence", sa.Text(), nullable=True),
        sa.Column("baseline_tests_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("baseline_tests_passed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("baseline_tests_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("baseline_tests_error", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("baseline_tests_skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("baseline_passing_ids_json", sa.Text(), nullable=True),
        sa.Column("baseline_failing_ids_json", sa.Text(), nullable=True),
        sa.Column("baseline_static_findings", sa.Integer(), nullable=False, server_default="0"),
        # patch application
        sa.Column("patch_applied", sa.Boolean(), nullable=True),
        sa.Column("patch_apply_error", sa.Text(), nullable=True),
        # post-patch
        sa.Column("post_patch_reproduced", sa.Boolean(), nullable=True),
        sa.Column("post_patch_reproduction_evidence", sa.Text(), nullable=True),
        sa.Column("post_tests_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("post_tests_passed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("post_tests_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("post_tests_error", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("post_tests_skipped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("post_passing_ids_json", sa.Text(), nullable=True),
        sa.Column("post_failing_ids_json", sa.Text(), nullable=True),
        sa.Column("post_static_findings", sa.Integer(), nullable=False, server_default="0"),
        # comparison
        sa.Column("target_bug_fixed", sa.Boolean(), nullable=True),
        sa.Column("newly_failing_ids_json", sa.Text(), nullable=True),
        sa.Column("recovered_ids_json", sa.Text(), nullable=True),
        sa.Column("regression_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("new_static_introduced", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("static_resolved", sa.Integer(), nullable=False, server_default="0"),
        # security
        sa.Column("security_passed", sa.Boolean(), nullable=True),
        sa.Column("security_issues_json", sa.Text(), nullable=True),
        # decision
        sa.Column("verification_score", sa.Float(), nullable=True),
        sa.Column("verification_decision", sa.String(50), nullable=True),
        sa.Column("decision_reasons_json", sa.Text(), nullable=True),
        sa.Column("evidence_summary", sa.Text(), nullable=True),
        # timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["candidate_id"], ["patch_candidates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["repair_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("candidate_id"),
    )
    op.create_index("ix_patch_verifications_session_id", "patch_verifications", ["session_id"])
    op.create_index("ix_patch_verifications_project_id", "patch_verifications", ["project_id"])
    op.create_index("ix_patch_verifications_candidate_id", "patch_verifications", ["candidate_id"])


def downgrade() -> None:
    op.drop_index("ix_patch_verifications_candidate_id", table_name="patch_verifications")
    op.drop_index("ix_patch_verifications_project_id", table_name="patch_verifications")
    op.drop_index("ix_patch_verifications_session_id", table_name="patch_verifications")
    op.drop_table("patch_verifications")
