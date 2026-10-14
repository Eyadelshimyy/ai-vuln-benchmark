from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from gvas.domain.identifiers import JsonValue
from gvas.domain.outbox import DEFAULT_MAX_ATTEMPTS
from gvas.infrastructure.db import json_type, metadata


class Base(DeclarativeBase):
    metadata = metadata


class Business(Base):
    __tablename__ = "businesses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Hosted-quote config: the client site's public origin, the customer-facing
    # name, a booking link, the publishable key, and the future connected
    # account id (stored now, used when connected payments arrive).
    site_url: Mapped[str | None] = mapped_column(String(2048))
    display_name: Mapped[str | None] = mapped_column(String(255))
    calendly_url: Mapped[str | None] = mapped_column(String(2048))
    stripe_account_id: Mapped[str | None] = mapped_column(String(255))
    public_key: Mapped[str | None] = mapped_column(String(255))
    # The website booking agent's per-business profile.
    intake_brief: Mapped[str | None] = mapped_column(Text)
    intake_questions: Mapped[str | None] = mapped_column(Text)
    intake_opening: Mapped[str | None] = mapped_column(Text)
    intake_offer_line: Mapped[str | None] = mapped_column(Text)
    intake_visit_question: Mapped[str | None] = mapped_column(Text)
    # Owner inbox copied on website-originated owner notices.
    notification_email: Mapped[str | None] = mapped_column(String(254))
    # The address that signs in to the owner dashboard, and the owner's
    # private calendar subscription link (a credential; never sent to a site).
    owner_email: Mapped[str | None] = mapped_column(String(320))
    calendar_feed_url: Mapped[str | None] = mapped_column(Text)
    # IANA zone name the business's times are shown in.
    timezone: Mapped[str | None] = mapped_column(String(64))
    # Set (e.g. "LL") only where invoices are on: numbers read LL-1042.
    invoice_prefix: Mapped[str | None] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("uq_businesses_public_key", "public_key", unique=True),)


class OwnerChannelEndpoint(Base):
    __tablename__ = "owner_channel_endpoints"
    __table_args__ = (
        UniqueConstraint("business_id", "id", name="uq_owner_channel_endpoints_business_id_id"),
        UniqueConstraint(
            "business_id",
            "source_namespace",
            "external_endpoint_id",
            name="uq_owner_channel_endpoints_source_endpoint",
        ),
        Index("ix_owner_channel_endpoints_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    source_namespace: Mapped[str] = mapped_column(String(100), nullable=False)
    external_endpoint_id: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_external_id: Mapped[str | None] = mapped_column(String(255))
    routing: Mapped[dict[str, JsonValue]] = mapped_column(json_type, nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "endpoint_id"],
            ["owner_channel_endpoints.business_id", "owner_channel_endpoints.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id"),
        UniqueConstraint("endpoint_id", "external_conversation_id"),
        Index("ix_conversations_endpoint_id", "endpoint_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    endpoint_id: Mapped[UUID] = mapped_column(nullable=False)
    external_conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    routing: Mapped[dict[str, JsonValue]] = mapped_column(json_type, nullable=False)


class InboundMessage(Base):
    __tablename__ = "inbound_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "endpoint_id"],
            ["owner_channel_endpoints.business_id", "owner_channel_endpoints.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id"),
        UniqueConstraint("endpoint_id", "message_key"),
        Index("ix_inbound_messages_conversation_id", "conversation_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    endpoint_id: Mapped[UUID] = mapped_column(nullable=False)
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    message_key: Mapped[str] = mapped_column(String(255), nullable=False)
    sender_external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    sender_role: Mapped[str] = mapped_column(String(50), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    parts: Mapped[list[JsonValue]] = mapped_column(json_type, nullable=False)
    reply_to: Mapped[dict[str, JsonValue] | None] = mapped_column(json_type)
    routing: Mapped[dict[str, JsonValue]] = mapped_column(json_type, nullable=False)


class OutboundMessage(Base):
    __tablename__ = "outbound_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id"),
        UniqueConstraint("inbound_message_id", "correlation_id"),
        Index("ix_outbound_messages_conversation_id", "conversation_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    inbound_message_id: Mapped[UUID | None] = mapped_column()
    parts: Mapped[list[JsonValue]] = mapped_column(json_type, nullable=False)
    reply_to: Mapped[dict[str, JsonValue] | None] = mapped_column(json_type)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(String(255))
    correlation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_detail: Mapped[str | None] = mapped_column(Text)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("inbound_message_id"),
        Index("ix_workflow_runs_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    inbound_message_id: Mapped[UUID] = mapped_column(nullable=False)
    intent: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    leased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID] = mapped_column(nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class OutboxMessage(Base):
    __tablename__ = "outbox_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "outbound_message_id"],
            ["outbound_messages.business_id", "outbound_messages.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "dedup_key"),
        UniqueConstraint("outbound_message_id"),
        UniqueConstraint("inbound_message_id"),
        Index("ix_outbox_messages_claim", "status", "available_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    outbound_message_id: Mapped[UUID | None] = mapped_column()
    inbound_message_id: Mapped[UUID | None] = mapped_column()
    command_type: Mapped[str] = mapped_column(String(255), nullable=False)
    payload: Mapped[dict[str, JsonValue]] = mapped_column(json_type, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(nullable=False, default=DEFAULT_MAX_ATTEMPTS)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    dedup_key: Mapped[str | None] = mapped_column(String(255))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(255))


class FieldNoteReport(Base):
    __tablename__ = "field_note_reports"
    __table_args__ = (
        UniqueConstraint("business_id", "id", name="uq_field_note_reports_business_id_id"),
        UniqueConstraint(
            "business_id",
            "case_id",
            name="uq_field_note_reports_business_id_case_id",
        ),
        Index("ix_field_note_reports_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[UUID] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    attempts: Mapped[int] = mapped_column(nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    leased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID] = mapped_column(nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    current_version: Mapped[int] = mapped_column(nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class FieldNoteReportVersion(Base):
    __tablename__ = "field_note_report_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "report_id"],
            ["field_note_reports.business_id", "field_note_reports.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "report_id",
            "version",
            name="uq_field_note_report_versions_report_id_version",
        ),
        UniqueConstraint(
            "report_id",
            "source_fingerprint",
            name="uq_field_note_report_versions_report_id_source_fingerprint",
        ),
        Index("ix_field_note_report_versions_report_id", "report_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    business_id: Mapped[UUID] = mapped_column(nullable=False)
    report_id: Mapped[UUID] = mapped_column(nullable=False)
    version: Mapped[int] = mapped_column(nullable=False)
    schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    document: Mapped[dict[str, JsonValue]] = mapped_column(json_type, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Customer(Base):
    """A portal identity: one row per ``(business, lowercased e-mail)``."""

    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("business_id", "id", name="uq_customers_business_id_id"),
        UniqueConstraint("business_id", "email", name="uq_customers_business_id_email"),
        Index("ix_customers_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(64))
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255))
    sms_consent: Mapped[bool | None] = mapped_column(Boolean)
    sms_consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PortalLoginTokenRecord(Base):
    """One magic link; only the SHA-256 digest of the token is stored."""

    __tablename__ = "portal_login_tokens"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
        Index("ix_portal_login_tokens_business_id_customer_id", "business_id", "customer_id"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[UUID] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PortalSessionRecord(Base):
    __tablename__ = "portal_sessions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
        Index("ix_portal_sessions_business_id_customer_id", "business_id", "customer_id"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[UUID] = mapped_column(nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OwnerLoginTokenRecord(Base):
    """One owner magic link; only the SHA-256 digest of the token is stored."""

    __tablename__ = "owner_login_tokens"
    __table_args__ = (Index("ix_owner_login_tokens_business_id", "business_id"),)

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OwnerSessionRecord(Base):
    __tablename__ = "owner_sessions"
    __table_args__ = (Index("ix_owner_sessions_business_id", "business_id"),)

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ServiceRequestRecord(Base):
    """A customer's ask for new work; ``source`` says which surface sent it."""

    __tablename__ = "service_requests"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "customer_id"],
            ["customers.business_id", "customers.id"],
            ondelete="CASCADE",
        ),
        Index("ix_service_requests_business_id_customer_id", "business_id", "customer_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[UUID] = mapped_column(nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    preferred_dates: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QuoteRecord(Base):
    __tablename__ = "quotes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id", name="uq_quotes_business_id_id"),
        UniqueConstraint(
            "business_id",
            "active_conversation_id",
            name="uq_quotes_business_id_active_conversation_id",
        ),
        UniqueConstraint(
            "conversation_id",
            "last_message_key",
            name="uq_quotes_conversation_id_last_message_key",
        ),
        Index("ix_quotes_business_id", "business_id"),
        Index("ix_quotes_conversation_id", "conversation_id"),
        Index(
            "ix_quotes_claim_token_hash",
            "claim_token_hash",
            unique=True,
        ),
        Index("ix_quotes_business_id_customer_id", "business_id", "customer_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    active_conversation_id: Mapped[UUID | None] = mapped_column()
    external_conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    revision: Mapped[int] = mapped_column(nullable=False)
    source_message_key: Mapped[str] = mapped_column(String(255), nullable=False)
    last_message_key: Mapped[str] = mapped_column(String(255), nullable=False)
    pending_request_text: Mapped[str] = mapped_column(Text, nullable=False)
    draft: Mapped[dict[str, JsonValue] | None] = mapped_column(json_type)
    approval_correlation_id: Mapped[str | None] = mapped_column(String(255))
    delivery_receipt: Mapped[dict[str, JsonValue] | None] = mapped_column(json_type)
    customer_appointment: Mapped[dict[str, JsonValue] | None] = mapped_column(json_type)
    customer_candidates: Mapped[list[JsonValue] | None] = mapped_column(json_type)
    claim_token: Mapped[str | None] = mapped_column(String(512))
    claim_token_hash: Mapped[str | None] = mapped_column(String(64))
    customer_status: Mapped[str | None] = mapped_column(String(50))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    texted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # A plain FK: a composite ``SET NULL`` would also null ``business_id``.
    # Repositories always pair it with ``business_id`` in their predicates.
    customer_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL")
    )
    billing: Mapped[str] = mapped_column(String(20), nullable=False, default="one_time")
    billing_interval: Mapped[str | None] = mapped_column(String(10))
    version: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DemoSandbox(Base):
    """A visitor's own copy of the demo business (demo deployments only).

    Only the SHA-256 digest of the visitor's sandbox token is stored. The
    sweep deletes the business, and everything under it, once
    ``last_active_at`` is older than the idle limit.
    """

    __tablename__ = "demo_sandboxes"
    __table_args__ = (Index("ix_demo_sandboxes_last_active_at", "last_active_at"),)

    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), primary_key=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# Workstream models are imported after Base is defined to register metadata.
from gvas.infrastructure import calendar_block_models as calendar_block_models  # noqa: E402, F401
from gvas.infrastructure import completeness_models as completeness_models  # noqa: E402, F401
from gvas.infrastructure import delivery_models as delivery_models  # noqa: E402, F401
from gvas.infrastructure import field_note_models as field_note_models  # noqa: E402, F401
from gvas.infrastructure import intake_models as intake_models  # noqa: E402, F401
from gvas.infrastructure import payment_models as payment_models  # noqa: E402, F401
from gvas.infrastructure import plan_models as plan_models  # noqa: E402, F401
from gvas.infrastructure import template_models as template_models  # noqa: E402, F401
from gvas.infrastructure import usage_models as usage_models  # noqa: E402, F401
from gvas.infrastructure.portal import models as portal_models  # noqa: E402, F401
