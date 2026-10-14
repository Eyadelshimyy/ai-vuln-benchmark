"""Add autonomous discovery tables (v1.1.0)

Revision ID: 012
Revises: 011
Create Date: 2025-01-03 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # discovery_runs — records one GitHub search sweep
    # ------------------------------------------------------------------
    op.create_table(
        "discovery_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("search_criteria", sa.Text(), nullable=True),
        sa.Column("discovered_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("eligible_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rejected_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("github_api_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_discovery_runs_status", "discovery_runs", ["status"])

    # ------------------------------------------------------------------
    # repository_candidates — public repos under consideration
    # ------------------------------------------------------------------
    op.create_table(
        "repository_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("github_repo_id", sa.Integer(), nullable=False),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(512), nullable=False),
        sa.Column("html_url", sa.String(1024), nullable=False),
        sa.Column("clone_url", sa.String(1024), nullable=False),
        sa.Column("stars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_fork", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("default_branch", sa.String(255), nullable=False, server_default="main"),
        sa.Column("primary_language", sa.String(100), nullable=True),
        sa.Column("license_key", sa.String(100), nullable=True),
        sa.Column("size_kb", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("open_issues", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("topics", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_pushed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("eligibility_status", sa.String(30), nullable=False, server_default="discovered"),
        sa.Column("eligibility_score", sa.Float(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("safety_classification", sa.String(30), nullable=True),
        sa.Column("safety_detail", sa.Text(), nullable=True),
        sa.Column("analysis_status", sa.String(30), nullable=True),
        sa.Column("last_analyzed_commit", sa.String(40), nullable=True),
        sa.Column("last_analyzed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("github_repo_id", name="uq_candidates_github_repo_id"),
        sa.UniqueConstraint("full_name", name="uq_candidates_full_name"),
    )
    op.create_index("ix_repository_candidates_github_repo_id", "repository_candidates", ["github_repo_id"])
    op.create_index("ix_repository_candidates_full_name", "repository_candidates", ["full_name"])
    op.create_index("ix_repository_candidates_eligibility_status", "repository_candidates", ["eligibility_status"])

    # ------------------------------------------------------------------
    # autonomous_analysis_runs — one pipeline run per candidate commit
    # ------------------------------------------------------------------
    op.create_table(
        "autonomous_analysis_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="queued"),
        sa.Column("current_stage", sa.String(30), nullable=True),
        sa.Column("commit_sha", sa.String(40), nullable=True),
        sa.Column("ai_provider", sa.String(50), nullable=True),
        sa.Column("ai_model", sa.String(100), nullable=True),
        sa.Column("is_local_ai", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("static_findings_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ai_hypotheses_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("validated_findings_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rejected_findings_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tests_generated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tests_executed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("repairs_generated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("repairs_verified", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["candidate_id"], ["repository_candidates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_autonomous_runs_candidate_id", "autonomous_analysis_runs", ["candidate_id"])
    op.create_index("ix_autonomous_runs_project_id", "autonomous_analysis_runs", ["project_id"])
    op.create_index("ix_autonomous_runs_status", "autonomous_analysis_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_autonomous_runs_status", "autonomous_analysis_runs")
    op.drop_index("ix_autonomous_runs_project_id", "autonomous_analysis_runs")
    op.drop_index("ix_autonomous_runs_candidate_id", "autonomous_analysis_runs")
    op.drop_table("autonomous_analysis_runs")

    op.drop_index("ix_repository_candidates_eligibility_status", "repository_candidates")
    op.drop_index("ix_repository_candidates_full_name", "repository_candidates")
    op.drop_index("ix_repository_candidates_github_repo_id", "repository_candidates")
    op.drop_table("repository_candidates")

    op.drop_index("ix_discovery_runs_status", "discovery_runs")
    op.drop_table("discovery_runs")
