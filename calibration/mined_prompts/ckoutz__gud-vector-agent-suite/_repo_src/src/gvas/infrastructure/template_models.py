from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from gvas.domain.identifiers import JsonValue
from gvas.infrastructure.db import json_type
from gvas.infrastructure.models import Base

_ACTIVE_STATUS_PREDICATE = text("status = 'active'")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class FieldNoteReportTemplate(Base):
    """Per-business versioned report structure; rows are immutable once written."""

    __tablename__ = "field_note_report_templates"
    __table_args__ = (
        UniqueConstraint(
            "business_id",
            "report_template_key",
            "version",
            name="uq_field_note_report_templates_key_version",
        ),
        Index("ix_field_note_report_templates_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    report_template_key: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    sections: Mapped[list[JsonValue]] = mapped_column(json_type, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class FieldNoteTemplateSet(Base):
    """Per-business versioned template set; rows are immutable apart from status."""

    __tablename__ = "field_note_template_sets"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "checklist_key", "checklist_version"],
            [
                "field_note_checklists.business_id",
                "field_note_checklists.checklist_key",
                "field_note_checklists.version",
            ],
            name="fk_field_note_template_sets_checklist",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["business_id", "report_template_key", "report_template_version"],
            [
                "field_note_report_templates.business_id",
                "field_note_report_templates.report_template_key",
                "field_note_report_templates.version",
            ],
            name="fk_field_note_template_sets_report_template",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "business_id",
            "template_set_key",
            "version",
            name="uq_field_note_template_sets_key_version",
        ),
        Index("ix_field_note_template_sets_business_id", "business_id"),
        Index(
            "uq_field_note_template_sets_active",
            "business_id",
            "template_set_key",
            unique=True,
            postgresql_where=_ACTIVE_STATUS_PREDICATE,
            sqlite_where=_ACTIVE_STATUS_PREDICATE,
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    template_set_key: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(nullable=False)
    industry_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    checklist_key: Mapped[str] = mapped_column(String(255), nullable=False)
    checklist_version: Mapped[int] = mapped_column(nullable=False)
    report_template_key: Mapped[str] = mapped_column(String(255), nullable=False)
    report_template_version: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class BusinessTemplateProfileRow(Base):
    """Which template-set key a business resolves, plus its seeded industry key."""

    __tablename__ = "business_template_profiles"

    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), primary_key=True
    )
    industry_key: Mapped[str] = mapped_column(String(255), nullable=False)
    template_set_key: Mapped[str] = mapped_column(String(255), nullable=False)
    default_template_set_key: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
