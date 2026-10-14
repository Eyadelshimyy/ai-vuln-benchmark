"""durable channel delivery receipts shared by the web and worker processes

Revision ID: 0009_channel_delivery_receipts
Revises: 0008_template_sets
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_channel_delivery_receipts"
down_revision = "0008_template_sets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "channel_delivery_receipts",
        sa.Column("delivery_key", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("provider_message_id", sa.String(length=255), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("delivery_key", name="pk_channel_delivery_receipts"),
    )


def downgrade() -> None:
    op.drop_table("channel_delivery_receipts")
