from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.project import Project


class BugReproductionSession(Base):
    __tablename__ = "bug_reproduction_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    debugging_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("debugging_sessions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    hypothesis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("debugging_hypotheses.id", ondelete="SET NULL"), nullable=True, index=True
    )
    generated_test_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("generated_tests.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
    )  # pending | running | completed | failed
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    reproducibility_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_classification: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # not_reproduced | inconclusive | intermittent | reproduced | consistently_reproduced
    strategy_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Fields from the reproduction plan used for evidence-based classification
    target_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_failure_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    observable_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship("Project", back_populates="bug_reproduction_sessions")
    attempts: Mapped[list[BugReproductionAttempt]] = relationship(
        "BugReproductionAttempt",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="BugReproductionAttempt.attempt_number",
    )


class BugReproductionAttempt(Base):
    __tablename__ = "bug_reproduction_attempts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("bug_reproduction_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    reproducer_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stdout: Mapped[str | None] = mapped_column(Text, nullable=True)
    stderr: Mapped[str | None] = mapped_column(Text, nullable=True)
    traceback: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    timed_out: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reproduced: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # reproduced | failed | error | timeout | environment_error | validator_rejected | no_evidence
    classification: Mapped[str] = mapped_column(String(50), nullable=False, default="failed")
    # Snippet from stdout/stderr that matched the expected failure pattern, if any
    evidence_matched: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[BugReproductionSession] = relationship(
        "BugReproductionSession", back_populates="attempts"
    )
