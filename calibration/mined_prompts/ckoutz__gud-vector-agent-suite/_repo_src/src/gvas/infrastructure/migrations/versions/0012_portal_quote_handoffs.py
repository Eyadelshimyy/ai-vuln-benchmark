"""portal quote handoffs, keyed on the delivery idempotency key

Revision ID: 0012_portal_quote_handoffs
Revises: 0011_quote_customer_appointments
"""

import sqlalchemy as sa
from alembic import op

revision = "0012_portal_quote_handoffs"
down_revision = "0011_quote_customer_appointments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "portal_quote_handoffs",
        sa.Column("idempotency_key", sa.String(length=512), primary_key=True),
        sa.Column("portal_quote_id", sa.String(length=255), nullable=False),
        sa.Column("claim_token", sa.String(length=512), nullable=False),
        sa.Column("quote_url", sa.String(length=2048), nullable=False),
        sa.Column("emailed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("portal_quote_handoffs")
