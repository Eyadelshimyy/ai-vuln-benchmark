"""payments: what the customer had done before the owner marked it paid

"Mark unpaid" puts a quote marked paid before the customer accepted it back
to sent or viewed, not accepted. Older rows stay null and reopen as accepted.

Revision ID: 0026_payment_status_before
Revises: 0025_payments_ledger
"""

import sqlalchemy as sa
from alembic import op

revision = "0026_payment_status_before"
down_revision = "0025_payments_ledger"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "payments", sa.Column("customer_status_before", sa.String(length=20), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("payments", "customer_status_before")
