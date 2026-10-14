"""invoices as their own records, and the per-business switch for them

Revision ID: 0030_demo_invoices
Revises: 0029_intake_offer_visit
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0030_demo_invoices"
down_revision = "0029_intake_offer_visit"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("businesses", sa.Column("invoice_prefix", sa.String(10), nullable=True))
    op.create_table(
        "invoices",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "business_id",
            sa.Uuid(),
            sa.ForeignKey("businesses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("quote_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("number", sa.String(32), nullable=False),
        sa.Column("customer_name", sa.String(255), nullable=True),
        sa.Column("customer_email", sa.String(320), nullable=True),
        sa.Column("customer_phone", sa.String(64), nullable=True),
        sa.Column("service_address", sa.Text(), nullable=True),
        sa.Column("line_items", json_type, nullable=False),
        sa.Column("total_cents", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_on", sa.Date(), nullable=False),
        sa.Column("sent_by", sa.String(320), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("business_id", "quote_id", name="uq_invoices_business_id_quote_id"),
        sa.UniqueConstraint("business_id", "sequence", name="uq_invoices_business_id_sequence"),
    )


def downgrade() -> None:
    op.drop_table("invoices")
    op.drop_column("businesses", "invoice_prefix")
