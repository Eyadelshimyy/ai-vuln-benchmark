"""quotes: when the customer was first texted the quote link

The owner dashboard counts a quote as sent once it was e-mailed (on the
delivery receipt) or texted (this column) to the customer.

Revision ID: 0024_quote_texted_at
Revises: 0023_business_timezone
"""

import sqlalchemy as sa
from alembic import op

revision = "0024_quote_texted_at"
down_revision = "0023_business_timezone"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("quotes", sa.Column("texted_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("quotes", "texted_at")
