"""Add security_findings table for Phase 2 potential/corroborated findings.

Revision ID: 013
Revises: 012
Create Date: 2026-09-20 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "security_findings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("analysis_id", sa.Uuid(), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="potential"),
        sa.Column("vulnerability_class", sa.String(length=100), nullable=True),
        sa.Column(
            "evidence_tier", sa.String(length=50), nullable=False, server_default="static_indicator"
        ),
        sa.Column("confidence", sa.String(length=50), nullable=False, server_default="low"),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("ai_analysis", sa.Text(), nullable=True),
        sa.Column("impact", sa.Text(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("line", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(length=50), nullable=True),
        sa.Column("analyzer", sa.String(length=100), nullable=True),
        sa.Column("rule_ids", sa.Text(), nullable=False, server_default=""),
        sa.Column("observation_refs", sa.Text(), nullable=False, server_default=""),
        sa.Column("evidence_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("asset", sa.Text(), nullable=True),
        sa.Column("report_title", sa.Text(), nullable=True),
        sa.Column("report_description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_security_findings_project_id", "security_findings", ["project_id"])
    op.create_index("ix_security_findings_analysis_id", "security_findings", ["analysis_id"])
    op.create_index("ix_security_findings_status", "security_findings", ["status"])
    op.create_index(
        "ix_security_findings_vulnerability_class", "security_findings", ["vulnerability_class"]
    )


def downgrade() -> None:
    op.drop_index("ix_security_findings_vulnerability_class", table_name="security_findings")
    op.drop_index("ix_security_findings_status", table_name="security_findings")
    op.drop_index("ix_security_findings_analysis_id", table_name="security_findings")
    op.drop_index("ix_security_findings_project_id", table_name="security_findings")
    op.drop_table("security_findings")
