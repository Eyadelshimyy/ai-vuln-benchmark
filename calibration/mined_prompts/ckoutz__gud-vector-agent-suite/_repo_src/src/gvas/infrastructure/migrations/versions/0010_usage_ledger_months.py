"""per-business monthly usage totals for transcription and review calls

Revision ID: 0010_usage_ledger_months
Revises: 0009_channel_delivery_receipts
"""

import sqlalchemy as sa
from alembic import op

revision = "0010_usage_ledger_months"
down_revision = "0009_channel_delivery_receipts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_ledger_months",
        sa.Column("business_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("month", sa.Date(), nullable=False),
        sa.Column("units", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("business_id", "kind", "month", name="pk_usage_ledger_months"),
    )


def downgrade() -> None:
    op.drop_table("usage_ledger_months")
