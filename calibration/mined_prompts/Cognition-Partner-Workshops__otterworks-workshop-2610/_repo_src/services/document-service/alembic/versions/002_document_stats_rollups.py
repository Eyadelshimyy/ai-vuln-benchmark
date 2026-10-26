"""Add document_stats_rollups for the usage dashboard.

Revision ID: 002
Revises: 001
Create Date: 2024-09-02 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_stats_rollups",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("documents_total", sa.Integer(), nullable=False),
        sa.Column("versions_total", sa.Integer(), nullable=False),
        sa.Column("words_total", sa.BigInteger(), nullable=False),
        sa.Column("computed_by", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("document_stats_rollups")
