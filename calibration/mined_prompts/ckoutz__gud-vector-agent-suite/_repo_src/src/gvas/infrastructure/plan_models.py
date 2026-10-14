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


class SiteRow(Base):
    __tablename__ = "sites"
    __table_args__ = (
        UniqueConstraint("business_id", "id", name="uq_sites_business_id_id"),
        UniqueConstraint("business_id", "external_ref", name="uq_sites_business_id_external_ref"),
        Index("ix_sites_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SitePlanSetRow(Base):
    __tablename__ = "site_plan_sets"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "site_id"],
            ["sites.business_id", "sites.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id", name="uq_site_plan_sets_business_id_id"),
        UniqueConstraint(
            "business_id", "site_id", "plan_set_key", name="uq_site_plan_sets_site_key"
        ),
        Index("ix_site_plan_sets_business_id", "business_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    site_id: Mapped[UUID] = mapped_column(nullable=False)
    plan_set_key: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SitePlanSetVersionRow(Base):
    """Immutable: rows are inserted once and never updated in place."""

    __tablename__ = "site_plan_set_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "plan_set_id"],
            ["site_plan_sets.business_id", "site_plan_sets.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "site_id"],
            ["sites.business_id", "sites.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id", name="uq_site_plan_set_versions_business_id_id"),
        UniqueConstraint(
            "plan_set_id", "version", name="uq_site_plan_set_versions_plan_set_id_version"
        ),
        UniqueConstraint(
            "plan_set_id",
            "content_digest",
            name="uq_site_plan_set_versions_plan_set_id_content_digest",
        ),
        Index("ix_site_plan_set_versions_plan_set_id", "plan_set_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    site_id: Mapped[UUID] = mapped_column(nullable=False)
    plan_set_id: Mapped[UUID] = mapped_column(nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_id: Mapped[UUID] = mapped_column(nullable=False)
    media_kind: Mapped[str] = mapped_column(String(50), nullable=False)
    artifact_locator: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    filename: Mapped[str | None] = mapped_column(String(500))
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SitePlanSetUploadRow(Base):
    """Copy-into-custody state for one source-channel file, leased and fenced."""

    __tablename__ = "site_plan_set_uploads"
    __table_args__ = (
        ForeignKeyConstraint(
            ["business_id", "plan_set_id"],
            ["site_plan_sets.business_id", "site_plan_sets.id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["business_id", "site_id"],
            ["sites.business_id", "sites.id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("business_id", "id", name="uq_site_plan_set_uploads_business_id_id"),
        UniqueConstraint(
            "business_id",
            "plan_set_id",
            "source_attachment_id",
            name="uq_site_plan_set_uploads_plan_set_source",
        ),
        Index("ix_site_plan_set_uploads_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    business_id: Mapped[UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    site_id: Mapped[UUID] = mapped_column(nullable=False)
    plan_set_id: Mapped[UUID] = mapped_column(nullable=False)
    source_attachment_id: Mapped[UUID] = mapped_column(nullable=False)
    source_media_kind: Mapped[str] = mapped_column(String(50), nullable=False)
    source_locator: Mapped[str] = mapped_column(Text, nullable=False)
    source_mime_type: Mapped[str | None] = mapped_column(String(255))
    source_filename: Mapped[str | None] = mapped_column(String(500))
    source_byte_size: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version_id: Mapped[UUID | None] = mapped_column()
    leased_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID] = mapped_column(nullable=False, default=uuid4)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
