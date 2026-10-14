"""intake conversations: the provider event type the booking was arranged on

Revision ID: 0016_intake_booking_event_type
Revises: 0015_intake_conversations
"""

import sqlalchemy as sa
from alembic import op

revision = "0016_intake_booking_event_type"
down_revision = "0015_intake_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "intake_conversations",
        sa.Column("booking_event_type_uri", sa.String(length=2048), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("intake_conversations", "booking_event_type_uri")
