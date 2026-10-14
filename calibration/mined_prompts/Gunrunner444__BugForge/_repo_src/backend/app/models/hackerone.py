"""Persisted HackerOne program, scope, draft, and submission state.

API tokens, passwords, cookies, and session tokens are never stored here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class DBHackerOneProgram(Base):
    __tablename__ = "hackerone_programs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    handle: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    program_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    program_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sync_status: Mapped[str] = mapped_column(String(32), nullable=False, default="never")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="closed")
    offers_bounties: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    requires_severity: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    instructions: Mapped[str] = mapped_column(Text, nullable=False, default="")
    open_scope_policy: Mapped[str] = mapped_column(Text, nullable=False, default="")
    open_scope_acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    active_testing_approved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    scope_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scope_sync_complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    scope_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_scope_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    continuation_state: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    weaknesses_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scope_pages_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    structured_scopes: Mapped[list[DBHackerOneStructuredScope]] = relationship(
        "DBHackerOneStructuredScope",
        back_populates="program",
        cascade="all, delete-orphan",
    )
    exclusions: Mapped[list[DBHackerOneScopeExclusion]] = relationship(
        "DBHackerOneScopeExclusion",
        back_populates="program",
        cascade="all, delete-orphan",
    )
    weaknesses: Mapped[list[DBHackerOneWeakness]] = relationship(
        "DBHackerOneWeakness",
        back_populates="program",
        cascade="all, delete-orphan",
    )
    syncs: Mapped[list[DBHackerOneSync]] = relationship(
        "DBHackerOneSync",
        back_populates="program",
        cascade="all, delete-orphan",
    )


class DBHackerOneStructuredScope(Base):
    __tablename__ = "hackerone_structured_scopes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    program_pk: Mapped[UUID] = mapped_column(
        ForeignKey("hackerone_programs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hackerone_id: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_type_raw: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    asset_type: Mapped[str] = mapped_column(String(64), nullable=False, default="other")
    asset_identifier: Mapped[str] = mapped_column(Text, nullable=False)
    instruction: Mapped[str] = mapped_column(Text, nullable=False, default="")
    eligible_for_bounty: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    eligible_for_submission: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    original: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    snapshot_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    program: Mapped[DBHackerOneProgram] = relationship(
        "DBHackerOneProgram", back_populates="structured_scopes"
    )


class DBHackerOneScopeExclusion(Base):
    __tablename__ = "hackerone_scope_exclusions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    program_pk: Mapped[UUID] = mapped_column(
        ForeignKey("hackerone_programs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hackerone_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    category: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    details: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at_remote: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at_remote: Mapped[str | None] = mapped_column(String(64), nullable=True)
    original: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    program: Mapped[DBHackerOneProgram] = relationship(
        "DBHackerOneProgram", back_populates="exclusions"
    )


class DBHackerOneWeakness(Base):
    __tablename__ = "hackerone_weaknesses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    program_pk: Mapped[UUID] = mapped_column(
        ForeignKey("hackerone_programs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hackerone_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    external_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    original: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    program: Mapped[DBHackerOneProgram] = relationship(
        "DBHackerOneProgram", back_populates="weaknesses"
    )


class DBHackerOneSync(Base):
    __tablename__ = "hackerone_syncs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    program_pk: Mapped[UUID] = mapped_column(
        ForeignKey("hackerone_programs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_scope_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scope_sync_complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    weakness_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    program: Mapped[DBHackerOneProgram] = relationship("DBHackerOneProgram", back_populates="syncs")


class DBHackerOneReportDraft(Base):
    __tablename__ = "hackerone_report_drafts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    finding_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("security_findings.id", ondelete="SET NULL"), nullable=True, index=True
    )
    program_handle: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    vulnerability_information: Mapped[str] = mapped_column(Text, nullable=False, default="")
    impact: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    weakness_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weakness_candidates: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    structured_scope_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_references: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    reproduction: Mapped[str | None] = mapped_column(Text, nullable=True)
    target: Mapped[str | None] = mapped_column(Text, nullable=True)
    eligible_for_submission: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    eligible_for_bounty: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    finding_verification: Mapped[str | None] = mapped_column(String(64), nullable=True)
    human_review_state: Mapped[str] = mapped_column(
        String(64), nullable=False, default="unreviewed"
    )
    submission_state: Mapped[str] = mapped_column(String(64), nullable=False, default="local_draft")
    remote_state: Mapped[str] = mapped_column(String(64), nullable=False, default="not_fetched")
    hackerone_report_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    submission_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    remote_state_raw: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    claim_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    report_content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    scope_snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    approved_report_content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approved_evidence_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approved_scope_snapshot_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approved_payload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    approval_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    approved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    approval_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scope_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class DBHackerOneSubmission(Base):
    __tablename__ = "hackerone_submissions"
    __table_args__ = (
        UniqueConstraint(
            "finding_id", "program_handle", name="uq_hackerone_submission_finding_program"
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    draft_id: Mapped[str | None] = mapped_column(
        ForeignKey("hackerone_report_drafts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    finding_id: Mapped[str] = mapped_column(String(64), nullable=False)
    program_handle: Mapped[str] = mapped_column(String(255), nullable=False)
    hackerone_report_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    submission_state: Mapped[str] = mapped_column(String(64), nullable=False)
    remote_state: Mapped[str] = mapped_column(String(64), nullable=False, default="not_fetched")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    remote_state_raw: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    claim_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DBHackerOneApprovalEvent(Base):
    """Immutable approval / invalidation events. Rows are insert-only."""

    __tablename__ = "hackerone_approval_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    draft_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    approved_by: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    report_content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    scope_snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class DBHackerOneAuditEvent(Base):
    __tablename__ = "hackerone_audit_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    project_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    finding_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    draft_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    program_handle: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    operator_identity: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class DBHackerOneReportIntent(Base):
    __tablename__ = "hackerone_report_intents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    finding_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("security_findings.id", ondelete="SET NULL"), nullable=True, index=True
    )
    draft_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    program_handle: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    vulnerability_information: Mapped[str] = mapped_column(Text, nullable=False, default="")
    impact: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    local_status: Mapped[str] = mapped_column(String(64), nullable=False, default="local_draft")
    human_review_state: Mapped[str] = mapped_column(
        String(64), nullable=False, default="local_draft"
    )
    remote_intent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    remote_state: Mapped[str] = mapped_column(String(64), nullable=False, default="not_fetched")
    remote_state_raw: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    attachments: Mapped[list[DBHackerOneAttachment]] = relationship(
        "DBHackerOneAttachment",
        back_populates="intent",
        cascade="all, delete-orphan",
    )


class DBHackerOneAttachment(Base):
    __tablename__ = "hackerone_attachments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    intent_id: Mapped[str] = mapped_column(
        ForeignKey("hackerone_report_intents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str] = mapped_column(String(128), nullable=False, default="", index=True)
    finding_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    draft_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    detected_content_type: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    review_state: Mapped[str] = mapped_column(String(64), nullable=False, default="received")
    reviewed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    authorized_for_upload: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    uploaded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    remote_attachment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    upload_state: Mapped[str] = mapped_column(String(64), nullable=False, default="received")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    authorized_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    intent: Mapped[DBHackerOneReportIntent] = relationship(
        "DBHackerOneReportIntent", back_populates="attachments"
    )
