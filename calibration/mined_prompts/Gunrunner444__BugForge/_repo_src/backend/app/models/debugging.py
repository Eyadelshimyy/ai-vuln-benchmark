from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.project import Project


class DebuggingSession(Base):
    __tablename__ = "debugging_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analysis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True, index=True
    )
    test_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("test_runs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending"
    )  # pending | running | completed | failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON summary of the evidence context sent to the AI
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship("Project", back_populates="debugging_sessions")
    hypotheses: Mapped[list[DebuggingHypothesis]] = relationship(
        "DebuggingHypothesis",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="DebuggingHypothesis.confidence.desc()",
    )
    ai_calls: Mapped[list[AIModelCall]] = relationship(
        "AIModelCall",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class DebuggingHypothesis(Base):
    __tablename__ = "debugging_hypotheses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("debugging_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_label: Mapped[str] = mapped_column(String(50), nullable=False)
    affected_files: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    affected_symbols: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    evidence_summary: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    contradictory_evidence: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    reproduction_strategy: Mapped[str] = mapped_column(Text, nullable=False, default="")
    recommended_tests: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    ai_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    ai_model: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    session: Mapped[DebuggingSession] = relationship(
        "DebuggingSession", back_populates="hypotheses"
    )


class AIModelCall(Base):
    __tablename__ = "ai_model_calls"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("debugging_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(nullable=False, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    session: Mapped[DebuggingSession] = relationship("DebuggingSession", back_populates="ai_calls")
