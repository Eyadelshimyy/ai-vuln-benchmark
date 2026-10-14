"""customers and intake conversations: SMS consent

Revision ID: 0018_customer_sms_consent
Revises: 0017_business_intake_profile
"""

import sqlalchemy as sa
from alembic import op

revision = "0018_customer_sms_consent"
down_revision = "0017_business_intake_profile"
branch_labels = None
depends_on = None

TABLES = ("customers", "intake_conversations")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("sms_consent", sa.Boolean(), nullable=True))
        op.add_column(table, sa.Column("sms_consent_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    for table in TABLES:
        op.drop_column(table, "sms_consent_at")
        op.drop_column(table, "sms_consent")
