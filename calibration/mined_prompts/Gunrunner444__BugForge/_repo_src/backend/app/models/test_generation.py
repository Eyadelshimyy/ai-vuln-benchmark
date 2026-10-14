from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.project import Project


class TestGenerationSession(Base):
    __tablename__ = "test_generation_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True
    )
    test_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("test_runs.id", ondelete="SET NULL"), nullable=True
    )
    debugging_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("debugging_sessions.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
    )  # pending | running | completed | failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship("Project", back_populates="test_generation_sessions")
    generated_tests: Mapped[list[GeneratedTest]] = relationship(
        "GeneratedTest",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class GeneratedTest(Base):
    __tablename__ = "generated_tests"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("test_generation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Repository-relative paths only
    target_file: Mapped[str] = mapped_column(Text, nullable=False)
    target_symbol: Mapped[str] = mapped_column(Text, nullable=False)
    # edge_case | boundary | invalid_input | error_handling | regression | behavioral | security
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    generated_code: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # pending | valid | invalid
    validation_status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    validation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # pending | passed | failed | error | timeout | not_run
    execution_status: Mapped[str] = mapped_column(String(50), nullable=False, default="not_run")
    execution_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Optional reference back to the hypothesis that motivated this test
    hypothesis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("debugging_hypotheses.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    session: Mapped[TestGenerationSession] = relationship(
        "TestGenerationSession", back_populates="generated_tests"
    )
