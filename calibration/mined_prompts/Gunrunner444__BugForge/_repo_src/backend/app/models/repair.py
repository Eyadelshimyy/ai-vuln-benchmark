"""Repair domain models — Phase 7 Automated Repair."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.verification import PatchVerification


class RepairSession(Base):
    """Orchestrates the automated repair of one confirmed/reproduced bug."""

    __tablename__ = "repair_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    debugging_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("debugging_sessions.id", ondelete="SET NULL"), nullable=True
    )
    hypothesis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("debugging_hypotheses.id", ondelete="SET NULL"), nullable=True
    )
    reproduction_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("bug_reproduction_sessions.id", ondelete="SET NULL"), nullable=True
    )
    # pending | running | completed | failed
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    total_candidates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_candidate_id: Mapped[UUID | None] = mapped_column(
        # Deferred FK — PatchCandidate is defined below; use string reference
        ForeignKey("patch_candidates.id", ondelete="SET NULL", use_alter=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship("Project", back_populates="repair_sessions")
    candidates: Mapped[list[PatchCandidate]] = relationship(
        "PatchCandidate",
        back_populates="session",
        primaryjoin="RepairSession.id == PatchCandidate.session_id",
        cascade="all, delete-orphan",
        order_by="PatchCandidate.rank",
    )


class PatchCandidate(Base):
    """One AI-generated patch candidate within a repair session."""

    __tablename__ = "patch_candidates"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("repair_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # 1-based rank within the session
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # AI generation metadata
    patch_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    patch_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Patch content
    patch_plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    patch_diff: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON array of repository-relative file paths changed by this patch
    changed_files_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lifecycle
    # pending | validating | applying | verifying | completed | rejected
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")

    # Validation
    # valid | rejected | skipped
    validation_status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    validation_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Pre-patch verification: does the original bug reproduce in the workspace?
    pre_patch_reproduced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pre_patch_stdout: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Post-patch verification: does the bug still reproduce after the patch?
    post_patch_reproduced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    post_patch_stdout: Mapped[str | None] = mapped_column(Text, nullable=True)
    bug_fixed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Existing test suite results after patch
    existing_tests_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    existing_tests_passed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    existing_tests_failed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # True if no previously-passing tests newly fail
    no_regressions: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    regression_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Static analysis delta: newly introduced findings after patch
    new_static_findings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Composite score 0.0–1.0 used for ranking
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # pending | accepted | rejected | best
    disposition: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[RepairSession] = relationship(
        "RepairSession",
        back_populates="candidates",
        foreign_keys=[session_id],
    )
    verification: Mapped[PatchVerification | None] = relationship(
        "PatchVerification",
        back_populates="candidate",
        uselist=False,
        cascade="all, delete-orphan",
    )
