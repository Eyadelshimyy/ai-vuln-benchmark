"""Phase 10 parser metadata and finding catalog fields.

Revision ID: 020
Revises: 019
Create Date: 2026-09-21 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "020"
down_revision: str | None = "019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "findings",
        sa.Column("catalog", sa.String(length=50), nullable=False, server_default="code_quality"),
    )
    op.add_column("findings", sa.Column("language", sa.String(length=50), nullable=True))
    op.add_column("findings", sa.Column("parser_backend", sa.String(length=50), nullable=True))
    op.add_column("findings", sa.Column("node_id", sa.String(length=500), nullable=True))
    op.add_column("findings", sa.Column("start_byte", sa.Integer(), nullable=True))
    op.add_column("findings", sa.Column("end_byte", sa.Integer(), nullable=True))
    op.add_column("findings", sa.Column("start_column", sa.Integer(), nullable=True))
    op.add_column("findings", sa.Column("end_column", sa.Integer(), nullable=True))
    op.create_index("ix_findings_catalog", "findings", ["catalog"])

    op.add_column("repository_files", sa.Column("parser_backend", sa.String(length=50), nullable=True))
    op.add_column("repository_files", sa.Column("parser_tier", sa.String(length=50), nullable=True))
    op.add_column(
        "repository_files",
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
    )

    op.add_column(
        "security_findings", sa.Column("parser_backend", sa.String(length=50), nullable=True)
    )
    op.add_column("security_findings", sa.Column("node_id", sa.String(length=500), nullable=True))
    op.add_column("security_findings", sa.Column("taint_path", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("security_findings", "taint_path")
    op.drop_column("security_findings", "node_id")
    op.drop_column("security_findings", "parser_backend")
    op.drop_column("repository_files", "error_count")
    op.drop_column("repository_files", "parser_tier")
    op.drop_column("repository_files", "parser_backend")
    op.drop_index("ix_findings_catalog", table_name="findings")
    op.drop_column("findings", "end_column")
    op.drop_column("findings", "start_column")
    op.drop_column("findings", "end_byte")
    op.drop_column("findings", "start_byte")
    op.drop_column("findings", "node_id")
    op.drop_column("findings", "parser_backend")
    op.drop_column("findings", "language")
    op.drop_column("findings", "catalog")
