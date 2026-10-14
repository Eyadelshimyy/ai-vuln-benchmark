"""quote customer picked from an appointment, plus pending candidate list

Revision ID: 0011_quote_customer_appointments
Revises: 0010_usage_ledger_months
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0011_quote_customer_appointments"
down_revision = "0010_usage_ledger_months"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("quotes", sa.Column("customer_appointment", json_type, nullable=True))
    op.add_column("quotes", sa.Column("customer_candidates", json_type, nullable=True))


def downgrade() -> None:
    op.drop_column("quotes", "customer_candidates")
    op.drop_column("quotes", "customer_appointment")
