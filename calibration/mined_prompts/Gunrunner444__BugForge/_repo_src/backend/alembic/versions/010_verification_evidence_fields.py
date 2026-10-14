"""Add Phase 8 evidence fields to patch_verifications

Revision ID: 010
Revises: 009
Create Date: 2025-01-01 00:00:00.000000
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Execution metadata
    op.add_column("patch_verifications", sa.Column("executor_type", sa.String(20), nullable=True))
    op.add_column("patch_verifications", sa.Column("schema_version", sa.Integer(), nullable=False, server_default="2"))
    # Baseline execution / analysis status
    op.add_column("patch_verifications", sa.Column("baseline_test_execution_status", sa.String(30), nullable=False, server_default="not_run"))
    op.add_column("patch_verifications", sa.Column("baseline_static_analysis_status", sa.String(20), nullable=False, server_default="not_run"))
    op.add_column("patch_verifications", sa.Column("baseline_finding_ids_json", sa.Text(), nullable=True))
    op.add_column("patch_verifications", sa.Column("baseline_duration_seconds", sa.Float(), nullable=True))
    # Post-patch execution / analysis status
    op.add_column("patch_verifications", sa.Column("post_test_execution_status", sa.String(30), nullable=False, server_default="not_run"))
    op.add_column("patch_verifications", sa.Column("post_static_analysis_status", sa.String(20), nullable=False, server_default="not_run"))
    op.add_column("patch_verifications", sa.Column("post_finding_ids_json", sa.Text(), nullable=True))
    op.add_column("patch_verifications", sa.Column("post_duration_seconds", sa.Float(), nullable=True))
    # Identity-based finding comparison
    op.add_column("patch_verifications", sa.Column("new_finding_ids_json", sa.Text(), nullable=True))
    op.add_column("patch_verifications", sa.Column("resolved_finding_ids_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("patch_verifications", "resolved_finding_ids_json")
    op.drop_column("patch_verifications", "new_finding_ids_json")
    op.drop_column("patch_verifications", "post_duration_seconds")
    op.drop_column("patch_verifications", "post_finding_ids_json")
    op.drop_column("patch_verifications", "post_static_analysis_status")
    op.drop_column("patch_verifications", "post_test_execution_status")
    op.drop_column("patch_verifications", "baseline_duration_seconds")
    op.drop_column("patch_verifications", "baseline_finding_ids_json")
    op.drop_column("patch_verifications", "baseline_static_analysis_status")
    op.drop_column("patch_verifications", "baseline_test_execution_status")
    op.drop_column("patch_verifications", "schema_version")
    op.drop_column("patch_verifications", "executor_type")
