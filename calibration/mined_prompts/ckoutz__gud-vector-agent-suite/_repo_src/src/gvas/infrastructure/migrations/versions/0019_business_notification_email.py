"""businesses: owner notification email

Revision ID: 0019_business_notification_email
Revises: 0018_customer_sms_consent
"""

import sqlalchemy as sa
from alembic import op

revision = "0019_business_notification_email"
down_revision = "0018_customer_sms_consent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "businesses", sa.Column("notification_email", sa.String(length=254), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("businesses", "notification_email")
