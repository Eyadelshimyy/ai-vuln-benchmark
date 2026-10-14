"""customer portal: customers, login tokens, sessions, service requests,
recurring quotes and subscriptions

Revision ID: 0014_customer_portal
Revises: 0013_hosted_quotes
"""

import sqlalchemy as sa
from alembic import op

revision = "0014_customer_portal"
down_revision = "0013_hosted_quotes"
branch_labels = None
depends_on = None


def _tenant_business_column() -> sa.Column[sa.Uuid]:
    return sa.Column(
        "business_id",
        sa.Uuid(),
        sa.ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "customers",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant_business_column(),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("phone", sa.String(length=64), nullable=True),
        sa.Column("stripe_customer_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("business_id", "id", name="uq_customers_business_id_id"),
        sa.UniqueConstraint("business_id", "email", name="uq_customers_business_id_email"),
    )
    op.create_index("ix_customers_business_id", "customers", ["business_id"])

    op.create_table(
        "portal_login_tokens",
        sa.Column("token_hash", sa.String(length=64), primary_key=True),
        _tenant_business_column(),
        sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_portal_login_tokens_business_id_customer_id",
        "portal_login_tokens",
        ["business_id", "customer_id"],
    )

    op.create_table(
        "portal_sessions",
        sa.Column("token_hash", sa.String(length=64), primary_key=True),
        _tenant_business_column(),
        sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_portal_sessions_business_id_customer_id",
        "portal_sessions",
        ["business_id", "customer_id"],
    )

    op.create_table(
        "service_requests",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant_business_column(),
        sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("preferred_dates", sa.String(length=512), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_service_requests_business_id_customer_id",
        "service_requests",
        ["business_id", "customer_id"],
    )

    with op.batch_alter_table("quotes") as batch:
        batch.add_column(sa.Column("customer_id", sa.Uuid(), nullable=True))
        batch.add_column(
            sa.Column("billing", sa.String(length=20), nullable=False, server_default="one_time")
        )
        batch.add_column(sa.Column("billing_interval", sa.String(length=10), nullable=True))
        batch.create_foreign_key(
            "fk_quotes_customer_id_customers",
            "customers",
            ["customer_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index("ix_quotes_business_id_customer_id", "quotes", ["business_id", "customer_id"])

    op.create_table(
        "quote_subscriptions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant_business_column(),
        sa.Column("quote_id", sa.Uuid(), nullable=False),
        sa.Column("customer_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("stripe_subscription_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("interval", sa.String(length=10), nullable=False),
        sa.Column("amount_cents", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_at_period_end", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "stripe_subscription_id", name="uq_quote_subscriptions_stripe_subscription_id"
        ),
    )
    op.create_index(
        "ix_quote_subscriptions_business_id_customer_id",
        "quote_subscriptions",
        ["business_id", "customer_id"],
    )
    op.create_index(
        "ix_quote_subscriptions_business_id_quote_id",
        "quote_subscriptions",
        ["business_id", "quote_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_quote_subscriptions_business_id_quote_id", table_name="quote_subscriptions")
    op.drop_index(
        "ix_quote_subscriptions_business_id_customer_id", table_name="quote_subscriptions"
    )
    op.drop_table("quote_subscriptions")
    op.drop_index("ix_quotes_business_id_customer_id", table_name="quotes")
    with op.batch_alter_table("quotes") as batch:
        batch.drop_constraint("fk_quotes_customer_id_customers", type_="foreignkey")
        batch.drop_column("billing_interval")
        batch.drop_column("billing")
        batch.drop_column("customer_id")
    op.drop_index("ix_service_requests_business_id_customer_id", table_name="service_requests")
    op.drop_table("service_requests")
    op.drop_index("ix_portal_sessions_business_id_customer_id", table_name="portal_sessions")
    op.drop_table("portal_sessions")
    op.drop_index(
        "ix_portal_login_tokens_business_id_customer_id", table_name="portal_login_tokens"
    )
    op.drop_table("portal_login_tokens")
    op.drop_index("ix_customers_business_id", table_name="customers")
    op.drop_table("customers")
