"""Intake conversations: the website chat agent's tables.

``intake_conversations`` carries the conversation token's SHA-256 digest, the
collected fields and the state machine; ``intake_messages`` is the persisted
transcript. Both are tenant-scoped — ``(business_id, id)`` is the boundary
and messages reference the composite key so a row can never point across
businesses.
"""

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
from sqlalchemy.orm import Mapped, mapped_column

from gvas.domain.identifiers import JsonValue
from gvas.infrastructure.db import json_type
from gvas.infrastructure.models import Base


class IntakeConversation(Base):
    __tablename__ = "intake_conversations"
    __table_args__ = (
        UniqueConstraint("business_id", "id", name="uq_intake_conversations_business_id_id"),
        UniqueConstraint(
            "business_id", "reference", name="uq_intake_conversations_business_id_reference"
        ),
        Index("ix_intake_conversations_business_id", "business_id"),
        Index("ix_intake_conversations_created", "business_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL")
    )
    reference: Mapped[str] = mapped_column(String(16), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    state: Mapped[str] = mapped_column(String(30), nullable=False)
    collected: Mapped[dict[str, JsonValue]] = mapped_column(json_type, nullable=False)
    proposed_slots: Mapped[list[JsonValue] | None] = mapped_column(json_type)
    requested_slot_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requested_slot_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    booking_kind: Mapped[str | None] = mapped_column(String(20))
    booking_link: Mapped[str | None] = mapped_column(String(2048))
    booking_attempted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    booked_event_uri: Mapped[str | None] = mapped_column(String(2048))
    # The booking a pending reschedule request replaces; cancelled when the
    # owner approves the new time, restored when they decline it.
    superseded_booking: Mapped[dict[str, JsonValue] | None] = mapped_column(json_type)
    reschedule_offered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    booking_event_type_uri: Mapped[str | None] = mapped_column(String(2048))
    decision_reason: Mapped[str | None] = mapped_column(String(500))
    decision_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    owner_notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    escalation_notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sms_consent: Mapped[bool | None] = mapped_column(Boolean)
    sms_consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IntakeMessage(Base):
    __tablename__ = "intake_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["intake_conversations.business_id", "intake_conversations.id"],
            ondelete="CASCADE",
        ),
        Index("ix_intake_messages_conversation_id", "conversation_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
