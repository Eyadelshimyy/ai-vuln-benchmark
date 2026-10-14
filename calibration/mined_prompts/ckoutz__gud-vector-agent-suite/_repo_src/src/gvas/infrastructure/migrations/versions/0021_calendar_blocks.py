"""calendar blocks the owner asked for by text

Revision ID: 0021_calendar_blocks
Revises: 0020_owner_dashboard
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0021_calendar_blocks"
down_revision = "0020_owner_dashboard"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "calendar_blocks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "business_id",
            sa.Uuid(),
            sa.ForeignKey("businesses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("start_minute", sa.Integer(), nullable=False),
        sa.Column("end_minute", sa.Integer(), nullable=False),
        sa.Column("reference", sa.String(length=32), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column(
            "previous",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_calendar_blocks_business_state", "calendar_blocks", ["business_id", "state"]
    )


def downgrade() -> None:
    op.drop_index("ix_calendar_blocks_business_state", table_name="calendar_blocks")
    op.drop_table("calendar_blocks")
