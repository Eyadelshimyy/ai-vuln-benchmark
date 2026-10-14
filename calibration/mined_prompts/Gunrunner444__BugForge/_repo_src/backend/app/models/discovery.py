"""Domain models for autonomous repository discovery — v1.1.0.

Three new tables:
  repository_candidates  — a public GitHub repo under consideration.
  discovery_runs         — a single discovery sweep (records what was found).
  autonomous_analysis_runs — one analysis run against a candidate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# RepositoryCandidate
# ---------------------------------------------------------------------------


class RepositoryCandidate(Base):
    """A public GitHub repository that the discovery system has encountered.

    Eligibility and safety are tracked here, not inside the Project model,
    so that rejected/blocked repos never pollute the main project space.
    """

    __tablename__ = "repository_candidates"
    __table_args__ = (
        UniqueConstraint("github_repo_id", name="uq_candidates_github_repo_id"),
        UniqueConstraint("full_name", name="uq_candidates_full_name"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    # GitHub identity
    github_repo_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    html_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    clone_url: Mapped[str] = mapped_column(String(1024), nullable=False)

    # Metadata snapshot at discovery time
    stars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_fork: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    default_branch: Mapped[str] = mapped_column(String(255), nullable=False, default="main")
    primary_language: Mapped[str | None] = mapped_column(String(100), nullable=True)
    license_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_kb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    open_issues: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    topics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array string
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_pushed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Lifecycle
    # discovered → screening → eligible/rejected/blocked → queued → analyzing → completed/failed/paused
    eligibility_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="discovered", index=True
    )
    eligibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # safe_candidate | low_risk | needs_review | blocked
    safety_classification: Mapped[str | None] = mapped_column(String(30), nullable=True)
    safety_detail: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON policy results

    # Analysis tracking
    analysis_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    last_analyzed_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    last_analyzed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    autonomous_runs: Mapped[list[AutonomousAnalysisRun]] = relationship(
        "AutonomousAnalysisRun",
        back_populates="candidate",
        cascade="all, delete-orphan",
        order_by="AutonomousAnalysisRun.created_at.desc()",
    )


# ---------------------------------------------------------------------------
# DiscoveryRun
# ---------------------------------------------------------------------------


class DiscoveryRun(Base):
    """Records one GitHub repository-search sweep."""

    __tablename__ = "discovery_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    # running | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running", index=True)

    # Snapshot of the criteria used for this sweep (JSON)
    search_criteria: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Counters
    discovered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    eligible_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    github_api_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


# ---------------------------------------------------------------------------
# AutonomousAnalysisRun
# ---------------------------------------------------------------------------


class AutonomousAnalysisRun(Base):
    """One autonomous analysis of a RepositoryCandidate.

    An analysis run orchestrates the existing BugForge pipeline
    (Analysis → Debugging → TestGen → Reproduction → Repair → Verification)
    and records aggregate progress here. The detailed records still live in
    the existing tables.
    """

    __tablename__ = "autonomous_analysis_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("repository_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Nullable FK — set once the Project/Analysis has been created for this run
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # State machine
    # queued → screening → acquiring → static_analyzing → ai_analyzing →
    # finding_validation → completed | failed | cancelled | inconclusive
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="queued", index=True)
    current_stage: Mapped[str | None] = mapped_column(String(30), nullable=True)

    # Snapshot
    commit_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    ai_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ai_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_local_ai: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Aggregate counters (updated as pipeline progresses)
    static_findings_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ai_hypotheses_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validated_findings_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_findings_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tests_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tests_executed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repairs_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repairs_verified: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Error
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    candidate: Mapped[RepositoryCandidate] = relationship(
        "RepositoryCandidate",
        back_populates="autonomous_runs",
    )
