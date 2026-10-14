"""Add GitHub integration tables (Phase 9)

Revision ID: 011
Revises: 010
Create Date: 2025-01-02 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # GitHub repositories — one per project
    op.create_table(
        "github_repositories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("repo", sa.String(255), nullable=False),
        sa.Column("github_id", sa.Integer(), nullable=True),
        sa.Column("default_branch", sa.String(255), nullable=False, server_default="main"),
        sa.Column("html_url", sa.String(1024), nullable=False),
        sa.Column("connected", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id"),
    )
    op.create_index("ix_github_repositories_project_id", "github_repositories", ["project_id"])

    # GitHub deliveries — one per delivery attempt per candidate
    op.create_table(
        "github_deliveries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("github_repository_id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("verification_id", sa.Uuid(), nullable=False),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("repo", sa.String(255), nullable=False),
        sa.Column("base_branch", sa.String(255), nullable=False),
        sa.Column("delivery_branch", sa.String(255), nullable=False),
        sa.Column("commit_sha", sa.String(40), nullable=True),
        sa.Column("pull_request_number", sa.Integer(), nullable=True),
        sa.Column("pull_request_url", sa.String(1024), nullable=True),
        sa.Column("verified_patch_hash", sa.String(64), nullable=False),
        sa.Column("delivered_patch_hash", sa.String(64), nullable=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["github_repository_id"], ["github_repositories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["candidate_id"], ["patch_candidates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["verification_id"], ["patch_verifications.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_github_deliveries_project_id", "github_deliveries", ["project_id"])
    op.create_index("ix_github_deliveries_candidate_id", "github_deliveries", ["candidate_id"])
    op.create_index("ix_github_deliveries_github_repository_id", "github_deliveries", ["github_repository_id"])
    op.create_index("ix_github_deliveries_verification_id", "github_deliveries", ["verification_id"])


def downgrade() -> None:
    op.drop_index("ix_github_deliveries_verification_id", "github_deliveries")
    op.drop_index("ix_github_deliveries_github_repository_id", "github_deliveries")
    op.drop_index("ix_github_deliveries_candidate_id", "github_deliveries")
    op.drop_index("ix_github_deliveries_project_id", "github_deliveries")
    op.drop_table("github_deliveries")
    op.drop_index("ix_github_repositories_project_id", "github_repositories")
    op.drop_table("github_repositories")
