from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from gvas.infrastructure.db import json_type
from gvas.infrastructure.models import Base


class QuotePayment(Base):
    """One checkout attempt against one quote.

    ``checkout_session_id`` is unique so two accepts cannot create two rows for
    the same provider session; ``(business_id, quote_id)`` keeps lookups
    tenant-scoped.
    """

    __tablename__ = "quote_payments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("checkout_session_id", name="uq_quote_payments_checkout_session_id"),
        Index("ix_quote_payments_business_id_quote_id", "business_id", "quote_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    quote_id: Mapped[UUID] = mapped_column(nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    checkout_session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    checkout_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    payment_intent_id: Mapped[str | None] = mapped_column(String(255))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    amount_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QuoteSubscription(Base):
    """The recurring agreement a paid recurring quote became; one row per
    provider subscription, tenant-scoped to its quote and customer."""

    __tablename__ = "quote_subscriptions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "stripe_subscription_id", name="uq_quote_subscriptions_stripe_subscription_id"
        ),
        # A manual plan may be for a customer with no e-mail, so no customer row.
        CheckConstraint(
            "customer_id IS NOT NULL OR provider = 'manual'",
            name="ck_quote_subscriptions_customer_unless_manual",
        ),
        Index("ix_quote_subscriptions_business_id_customer_id", "business_id", "customer_id"),
        Index("ix_quote_subscriptions_business_id_quote_id", "business_id", "quote_id"),
        Index(
            "uq_quote_subscriptions_one_manual_plan",
            "business_id",
            "quote_id",
            unique=True,
            postgresql_where=text("provider = 'manual'"),
            sqlite_where=text("provider = 'manual'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    quote_id: Mapped[UUID] = mapped_column(nullable=False)
    customer_id: Mapped[UUID | None] = mapped_column(nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    stripe_subscription_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)
    amount_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    paid_from: Mapped[date | None] = mapped_column(Date)
    paid_through: Mapped[date | None] = mapped_column(Date)


class LedgerPaymentRow(Base):
    """One settled payment, card or manual; see ``LedgerPayment``."""

    __tablename__ = "payments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("source", "reference", name="uq_payments_source_reference"),
        Index("ix_payments_business_id_paid_at", "business_id", "paid_at"),
        Index(
            "uq_payments_one_active_one_off",
            "business_id",
            "quote_id",
            unique=True,
            postgresql_where=text("kind = 'one_off' AND voided_at IS NULL AND NOT duplicate"),
            sqlite_where=text("kind = 'one_off' AND voided_at IS NULL AND NOT duplicate"),
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    quote_id: Mapped[UUID] = mapped_column(nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    method: Mapped[str] = mapped_column(String(20), nullable=False)
    reference: Mapped[str] = mapped_column(String(255), nullable=False)
    amount_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    paid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    months_covered: Mapped[int | None] = mapped_column(Integer)
    recorded_by: Mapped[str | None] = mapped_column(String(320))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    note: Mapped[str | None] = mapped_column(String(500))
    voided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    customer_status_before: Mapped[str | None] = mapped_column(String(20))
    voided_by: Mapped[str | None] = mapped_column(String(320))
    duplicate: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )


class PaymentProviderEvent(Base):
    """Replay ledger for payment webhooks: ``(provider, event_id)`` recorded in
    the same transaction as its effects, so a retried delivery answers from the
    row instead of repeating work."""

    __tablename__ = "payment_provider_events"

    provider: Mapped[str] = mapped_column(String(50), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


__all__ = ["LedgerPaymentRow", "PaymentProviderEvent", "QuotePayment", "QuoteSubscription"]


class InvoiceRow(Base):
    """An invoice sent from an accepted one-off quote; see ``Invoice``."""

    __tablename__ = "invoices"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "quote_id"],
            ["quotes.business_id", "quotes.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "quote_id", name="uq_invoices_business_id_quote_id"),
        UniqueConstraint("business_id", "sequence", name="uq_invoices_business_id_sequence"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    quote_id: Mapped[UUID] = mapped_column(nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    number: Mapped[str] = mapped_column(String(32), nullable=False)
    customer_name: Mapped[str | None] = mapped_column(String(255))
    customer_email: Mapped[str | None] = mapped_column(String(320))
    customer_phone: Mapped[str | None] = mapped_column(String(64))
    service_address: Mapped[str | None] = mapped_column(Text)
    line_items: Mapped[list[dict[str, object]]] = mapped_column(json_type, nullable=False)
    total_cents: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    due_on: Mapped[date] = mapped_column(Date, nullable=False)
    sent_by: Mapped[str | None] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
