from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
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


def _utcnow() -> datetime:
    return datetime.now(UTC)


class FieldNoteChecklist(Base):
    __tablename__ = "field_note_checklists"
    __table_args__ = (
        UniqueConstraint(
            "business_id",
            "checklist_key",
            "version",
            name="uq_field_note_checklists_key_version",
        ),
        Index("ix_field_note_checklists_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    checklist_key: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(nullable=False)
    items: Mapped[list[JsonValue]] = mapped_column(json_type, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class FieldNoteReview(Base):
    __tablename__ = "field_note_reviews"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            name="fk_field_note_reviews_business_id_conversations",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            name="fk_field_note_reviews_business_id_inbound_messages",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "active_conversation_id"],
            ["conversations.business_id", "conversations.id"],
            name="fk_field_note_reviews_active_conversation",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "checklist_key", "checklist_version"],
            [
                "field_note_checklists.business_id",
                "field_note_checklists.checklist_key",
                "field_note_checklists.version",
            ],
            name="fk_field_note_reviews_checklist",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["business_id", "template_set_key", "template_set_version"],
            [
                "field_note_template_sets.business_id",
                "field_note_template_sets.template_set_key",
                "field_note_template_sets.version",
            ],
            name="fk_field_note_reviews_template_set",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("business_id", "id", name="uq_field_note_reviews_business_id_id"),
        UniqueConstraint(
            "business_id",
            "inbound_message_id",
            "transcript_fingerprint",
            name="uq_field_note_reviews_business_id_inbound_message_transcript",
        ),
        UniqueConstraint(
            "business_id",
            "inbound_message_id",
            "revision",
            name="uq_field_note_reviews_business_id_inbound_message_revision",
        ),
        UniqueConstraint(
            "business_id",
            "active_conversation_id",
            name="uq_field_note_reviews_active_conversation",
        ),
        Index("ix_field_note_reviews_conversation_id", "conversation_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    active_conversation_id: Mapped[UUID | None] = mapped_column()
    external_conversation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    inbound_message_id: Mapped[UUID] = mapped_column(nullable=False)
    checklist_key: Mapped[str] = mapped_column(String(255), nullable=False)
    checklist_version: Mapped[int] = mapped_column(nullable=False)
    template_set_key: Mapped[str | None] = mapped_column(String(255))
    template_set_version: Mapped[int | None] = mapped_column()
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    transcript_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    revision: Mapped[int] = mapped_column(nullable=False, default=1)
    thread_correlation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    round_index: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class FieldNoteFollowUpQuestion(Base):
    __tablename__ = "field_note_follow_up_questions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "review_id"],
            ["field_note_reviews.business_id", "field_note_reviews.id"],
            name="fk_field_note_questions_review",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "business_id", "id", name="uq_field_note_follow_up_questions_business_id_id"
        ),
        UniqueConstraint(
            "business_id",
            "review_id",
            "id",
            name="uq_field_note_follow_up_questions_business_review_id",
        ),
        UniqueConstraint(
            "review_id",
            "round_index",
            "item_key",
            name="uq_field_note_follow_up_questions_round_item",
        ),
        UniqueConstraint(
            "review_id",
            "correlation_id",
            name="uq_field_note_follow_up_questions_correlation",
        ),
        Index("ix_field_note_follow_up_questions_review_id", "review_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    review_id: Mapped[UUID] = mapped_column(nullable=False)
    item_key: Mapped[str] = mapped_column(String(255), nullable=False)
    round_index: Mapped[int] = mapped_column(nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    asked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class FieldNoteReviewAnswer(Base):
    __tablename__ = "field_note_review_answers"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "review_id"],
            ["field_note_reviews.business_id", "field_note_reviews.id"],
            name="fk_field_note_review_answers_business_id_field_note_reviews",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "review_id", "question_id"],
            [
                "field_note_follow_up_questions.business_id",
                "field_note_follow_up_questions.review_id",
                "field_note_follow_up_questions.id",
            ],
            name="fk_field_note_review_answers_question",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            name="fk_field_note_review_answers_business_id_inbound_messages",
            ondelete="CASCADE",
        ),
        UniqueConstraint("question_id", name="uq_field_note_review_answers_question_id"),
        UniqueConstraint(
            "review_id",
            "inbound_message_id",
            name="uq_field_note_review_answers_review_inbound_message",
        ),
        Index("ix_field_note_review_answers_review_id", "review_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    review_id: Mapped[UUID] = mapped_column(nullable=False)
    question_id: Mapped[UUID] = mapped_column(nullable=False)
    inbound_message_id: Mapped[UUID] = mapped_column(nullable=False)
    item_key: Mapped[str] = mapped_column(String(255), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
