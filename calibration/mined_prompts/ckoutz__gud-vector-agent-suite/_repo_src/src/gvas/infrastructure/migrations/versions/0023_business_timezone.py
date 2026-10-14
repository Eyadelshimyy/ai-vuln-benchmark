"""businesses: the IANA time zone the business works in

Filled from the Calendly schedule the first time availability is read, and
editable on the owner Settings page. Every time label renders in it.

Revision ID: 0023_business_timezone
Revises: 0022_intake_reschedule_custody
"""

import sqlalchemy as sa
from alembic import op

revision = "0023_business_timezone"
down_revision = "0022_intake_reschedule_custody"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("businesses", sa.Column("timezone", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("businesses", "timezone")
