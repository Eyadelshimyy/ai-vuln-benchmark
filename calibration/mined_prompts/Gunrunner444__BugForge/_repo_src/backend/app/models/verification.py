"""Patch Verification domain models — Phase 8 Patch Verification."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.repair import PatchCandidate, RepairSession


class PatchVerification(Base):
    """Full evidence record for a single patch candidate.

    Captures the complete before/after picture so any acceptance decision
    is reconstructable without asking the AI again.
    """

    __tablename__ = "patch_verifications"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("patch_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,  # one verification record per candidate
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("repair_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ---------- lifecycle ----------------------------------------------------
    # pending | running | baseline_failed | applying | testing |
    # analyzing | comparing | verified | rejected | inconclusive | environment_failed
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---------- execution metadata -------------------------------------------
    # docker | local | none
    executor_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    # ---------- baseline (pre-patch) -----------------------------------------
    # Reproduction
    baseline_reproduced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    baseline_reproduction_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Test suite
    baseline_tests_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    baseline_tests_passed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    baseline_tests_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    baseline_tests_error: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    baseline_tests_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # JSON array of passing/failing node IDs in the baseline
    baseline_passing_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    baseline_failing_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Static analysis
    baseline_static_findings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # success | timeout | environment_error | report_error | not_run
    baseline_test_execution_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="not_run"
    )
    # success | error | not_run
    baseline_static_analysis_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="not_run"
    )
    # JSON array of normalized finding identity strings (analyzer:category:file)
    baseline_finding_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    baseline_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ---------- patch application --------------------------------------------
    patch_applied: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    patch_apply_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---------- post-patch ---------------------------------------------------
    # Reproduction
    post_patch_reproduced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    post_patch_reproduction_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Test suite
    post_tests_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    post_tests_passed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    post_tests_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    post_tests_error: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    post_tests_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    post_passing_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    post_failing_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Static analysis
    post_static_findings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # success | timeout | environment_error | report_error | not_run
    post_test_execution_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="not_run"
    )
    # success | error | not_run
    post_static_analysis_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="not_run"
    )
    post_finding_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    post_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ---------- comparison results -------------------------------------------
    target_bug_fixed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # JSON arrays of node IDs
    newly_failing_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    recovered_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    regression_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    new_static_introduced: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    static_resolved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # JSON arrays of finding identity strings for identity-based comparison
    new_finding_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_finding_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---------- security checks ----------------------------------------------
    security_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # JSON array of security issue strings
    security_issues_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---------- decision & scoring -------------------------------------------
    verification_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # verified | rejected | inconclusive | environment_failed
    verification_decision: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # JSON array of human-readable rejection / acceptance reasons
    decision_reasons_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---------- evidence summary ---------------------------------------------
    evidence_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ---------- timestamps ---------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ---------- relationships ------------------------------------------------
    candidate: Mapped[PatchCandidate] = relationship(
        "PatchCandidate", back_populates="verification"
    )
    session: Mapped[RepairSession] = relationship("RepairSession")
    project: Mapped[Project] = relationship("Project")
