"""payments: one row per settled payment, card or manual

Fills in the card payments already settled. Their settle time was not kept,
so it is taken from when the checkout row was last updated (the webhook that
marked it paid). Plan renewals were never stored and can't be recovered.

Revision ID: 0025_payments_ledger
Revises: 0024_quote_texted_at
"""

import sqlalchemy as sa
from alembic import op

revision = "0025_payments_ledger"
down_revision = "0024_quote_texted_at"
branch_labels = None
depends_on = None

ACTIVE_ONE_OFF = sa.text("kind = 'one_off' AND voided_at IS NULL AND NOT duplicate")


def upgrade() -> None:
    op.create_table(
        "payments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "business_id",
            sa.Uuid(),
            sa.ForeignKey("businesses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("quote_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("method", sa.String(length=20), nullable=False),
        sa.Column("reference", sa.String(length=255), nullable=False),
        sa.Column("amount_cents", sa.BigInteger(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("months_covered", sa.Integer()),
        sa.Column("recorded_by", sa.String(length=320)),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("note", sa.String(length=500)),
        sa.Column("voided_at", sa.DateTime(timezone=True)),
        sa.Column("voided_by", sa.String(length=320)),
        sa.Column("duplicate", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("source", "reference", name="uq_payments_source_reference"),
    )
    op.create_index("ix_payments_business_id_paid_at", "payments", ["business_id", "paid_at"])
    op.execute(
        """
        INSERT INTO payments (
            id, business_id, quote_id, kind, source, method, reference,
            amount_cents, currency, paid_at, months_covered, recorded_at, duplicate
        )
        SELECT
            id, business_id, quote_id,
            CASE WHEN plan_interval IS NULL THEN 'one_off' ELSE 'plan' END,
            'stripe', 'card', checkout_session_id,
            amount_cents, UPPER(currency), updated_at,
            CASE WHEN plan_interval IS NULL THEN NULL
                WHEN plan_interval = 'year' THEN 12 ELSE 1 END,
            updated_at, false
        FROM (
            SELECT qp.*, (
                -- A quote can have had more than one subscription; the latest
                -- one decides the interval, and each checkout stays one row.
                SELECT s.interval FROM quote_subscriptions s
                WHERE s.business_id = qp.business_id AND s.quote_id = qp.quote_id
                ORDER BY s.created_at DESC, s.id
                LIMIT 1
            ) AS plan_interval
            FROM quote_payments qp
            WHERE qp.status = 'paid'
        ) paid
        """
    )
    # A quote paid through two checkouts keeps both rows; only the first counts.
    op.execute(
        """
        UPDATE payments SET duplicate = true
        WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY business_id, quote_id ORDER BY paid_at, id
                ) AS n
                FROM payments WHERE kind = 'one_off'
            ) ranked
            WHERE n > 1
        )
        """
    )
    op.create_index(
        "uq_payments_one_active_one_off",
        "payments",
        ["business_id", "quote_id"],
        unique=True,
        postgresql_where=ACTIVE_ONE_OFF,
        sqlite_where=ACTIVE_ONE_OFF,
    )


def downgrade() -> None:
    op.drop_table("payments")
