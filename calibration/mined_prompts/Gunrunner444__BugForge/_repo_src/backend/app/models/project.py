from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.debugging import DebuggingSession
    from app.models.github import GitHubRepository
    from app.models.repair import RepairSession
    from app.models.reproduction import BugReproductionSession
    from app.models.security_finding import DBSecurityFinding
    from app.models.test_generation import TestGenerationSession
    from app.models.test_run import TestRun


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    repository_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    analyses: Mapped[list[Analysis]] = relationship(
        "Analysis",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="Analysis.created_at.desc()",
    )

    test_runs: Mapped[list[TestRun]] = relationship(
        "TestRun",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="TestRun.created_at.desc()",
    )

    debugging_sessions: Mapped[list[DebuggingSession]] = relationship(
        "DebuggingSession",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="DebuggingSession.created_at.desc()",
    )

    test_generation_sessions: Mapped[list[TestGenerationSession]] = relationship(
        "TestGenerationSession",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="TestGenerationSession.created_at.desc()",
    )

    bug_reproduction_sessions: Mapped[list[BugReproductionSession]] = relationship(
        "BugReproductionSession",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="BugReproductionSession.created_at.desc()",
    )

    repair_sessions: Mapped[list[RepairSession]] = relationship(
        "RepairSession",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="RepairSession.created_at.desc()",
    )

    github_repository: Mapped[GitHubRepository | None] = relationship(
        "GitHubRepository",
        back_populates="project",
        cascade="all, delete-orphan",
        uselist=False,
    )

    security_findings: Mapped[list[DBSecurityFinding]] = relationship(
        "DBSecurityFinding",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="DBSecurityFinding.created_at.desc()",
    )
