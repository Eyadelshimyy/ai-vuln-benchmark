"""intake conversations: reschedule custody of the previous booking

A reschedule keeps the customer's existing calendar event until the owner
decides: ``superseded_booking`` holds what it was (cancelled on approve,
restored on decline) and ``reschedule_offered_at`` marks that new times are
on the table while the booking still stands.

Revision ID: 0022_intake_reschedule_custody
Revises: 0021_calendar_blocks
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0022_intake_reschedule_custody"
down_revision = "0021_calendar_blocks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "intake_conversations",
        sa.Column(
            "superseded_booking",
            sa.JSON().with_variant(JSONB, "postgresql"),
            nullable=True,
        ),
    )
    op.add_column(
        "intake_conversations",
        sa.Column("reschedule_offered_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("intake_conversations", "reschedule_offered_at")
    op.drop_column("intake_conversations", "superseded_booking")
