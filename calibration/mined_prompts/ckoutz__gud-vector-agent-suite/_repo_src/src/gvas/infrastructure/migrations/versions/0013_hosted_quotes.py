"""hosted quotes: business site config, quote claim tokens, payments, events

Revision ID: 0013_hosted_quotes
Revises: 0012_portal_quote_handoffs
"""

import sqlalchemy as sa
from alembic import op

revision = "0013_hosted_quotes"
down_revision = "0012_portal_quote_handoffs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("businesses", sa.Column("site_url", sa.String(length=2048), nullable=True))
    op.add_column("businesses", sa.Column("display_name", sa.String(length=255), nullable=True))
    op.add_column("businesses", sa.Column("calendly_url", sa.String(length=2048), nullable=True))
    op.add_column(
        "businesses", sa.Column("stripe_account_id", sa.String(length=255), nullable=True)
    )
    op.add_column("businesses", sa.Column("public_key", sa.String(length=255), nullable=True))
    op.create_index("uq_businesses_public_key", "businesses", ["public_key"], unique=True)
    op.add_column("quotes", sa.Column("claim_token", sa.String(length=512), nullable=True))
    op.add_column("quotes", sa.Column("claim_token_hash", sa.String(length=64), nullable=True))
    op.add_column("quotes", sa.Column("customer_status", sa.String(length=50), nullable=True))
    op.add_column("quotes", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_quotes_claim_token_hash", "quotes", ["claim_token_hash"], unique=True)
    # Rows approved before claim tokens existed: display the approval time.
    op.execute(
        "UPDATE quotes SET approved_at = updated_at "
        "WHERE approved_at IS NULL "
        "AND status IN ('approved', 'delivery_pending', 'delivered')"
    )
    op.create_table(
        "quote_payments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "business_id",
            sa.Uuid(),
            sa.ForeignKey("businesses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("quote_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("checkout_session_id", sa.String(length=255), nullable=False),
        sa.Column("checkout_url", sa.String(length=2048), nullable=False),
        sa.Column("payment_intent_id", sa.String(length=255), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("amount_cents", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("checkout_session_id", name="uq_quote_payments_checkout_session_id"),
    )
    op.create_index(
        "ix_quote_payments_business_id_quote_id",
        "quote_payments",
        ["business_id", "quote_id"],
    )
    op.create_table(
        "payment_provider_events",
        sa.Column("provider", sa.String(length=50), primary_key=True),
        sa.Column("event_id", sa.String(length=255), primary_key=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("payment_provider_events")
    op.drop_index("ix_quote_payments_business_id_quote_id", table_name="quote_payments")
    op.drop_table("quote_payments")
    op.drop_index("ix_quotes_claim_token_hash", table_name="quotes")
    op.drop_column("quotes", "approved_at")
    op.drop_column("quotes", "customer_status")
    op.drop_column("quotes", "claim_token_hash")
    op.drop_column("quotes", "claim_token")
    op.drop_index("uq_businesses_public_key", table_name="businesses")
    op.drop_column("businesses", "public_key")
    op.drop_column("businesses", "stripe_account_id")
    op.drop_column("businesses", "calendly_url")
    op.drop_column("businesses", "display_name")
    op.drop_column("businesses", "site_url")
