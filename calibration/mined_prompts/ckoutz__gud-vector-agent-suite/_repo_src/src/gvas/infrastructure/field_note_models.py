from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from gvas.infrastructure.models import Base


class FieldNoteCase(Base):
    __tablename__ = "field_note_cases"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "origin_inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id", name="uq_field_note_cases_business_id_id"),
        UniqueConstraint(
            "origin_inbound_message_id", name="uq_field_note_cases_origin_inbound_message_id"
        ),
        Index("ix_field_note_cases_conversation_id", "conversation_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    origin_inbound_message_id: Mapped[UUID] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FieldNotePartRow(Base):
    __tablename__ = "field_note_parts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "case_id"],
            ["field_note_cases.business_id", "field_note_cases.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "source_inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id", name="uq_field_note_parts_business_id_id"),
        UniqueConstraint("case_id", "sequence", name="uq_field_note_parts_case_id_sequence"),
        UniqueConstraint(
            "case_id",
            "source_inbound_message_id",
            "sequence",
            name="uq_field_note_parts_case_source_sequence",
        ),
        Index("ix_field_note_parts_case_sequence", "case_id", "sequence"),
        Index("ix_field_note_parts_transcription_status", "transcription_status"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[UUID] = mapped_column(nullable=False)
    source_inbound_message_id: Mapped[UUID] = mapped_column(nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    text: Mapped[str | None] = mapped_column(Text)
    attachment_id: Mapped[UUID | None] = mapped_column()
    media_kind: Mapped[str | None] = mapped_column(String(50))
    attachment_locator: Mapped[str | None] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    filename: Mapped[str | None] = mapped_column(String(500))
    byte_size: Mapped[int | None] = mapped_column(Integer)
    transcription_status: Mapped[str] = mapped_column(String(50), nullable=False)
    transcript_text: Mapped[str | None] = mapped_column(Text)
    transcript_language: Mapped[str | None] = mapped_column(String(50))
    transcript_confidence: Mapped[float | None] = mapped_column()
    transcript_duration_seconds: Mapped[float | None] = mapped_column()
    transcript_provider_ref: Mapped[str | None] = mapped_column(String(255))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    leased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID] = mapped_column(nullable=False, default=uuid4)


class FieldNoteConversationState(Base):
    __tablename__ = "field_note_conversation_states"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            ondelete="CASCADE",
        ),
        Index("ix_field_note_conversation_states_business_id", "business_id"),
    )

    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[UUID] = mapped_column(primary_key=True)
    active_case_id: Mapped[UUID | None] = mapped_column(nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
