"""GitHub Integration domain models — Phase 9."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.project import Project


class GitHubRepository(Base):
    """Associates a BugForge project with a GitHub repository.

    One-to-one with Project.  GitHub credentials are NEVER stored here —
    they come from server-side environment configuration only.
    """

    __tablename__ = "github_repositories"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True,  # one GitHub repo per project
    )
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    repo: Mapped[str] = mapped_column(String(255), nullable=False)
    # Numerical GitHub repository ID (returned by GitHub API)
    github_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    default_branch: Mapped[str] = mapped_column(String(255), nullable=False, default="main")
    html_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    # Whether BugForge has successfully verified access to this repo
    connected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    project: Mapped[Project] = relationship("Project", back_populates="github_repository")
    deliveries: Mapped[list[GitHubDelivery]] = relationship(
        "GitHubDelivery",
        back_populates="github_repository",
        cascade="all, delete-orphan",
        order_by="GitHubDelivery.created_at.desc()",
    )


class GitHubDelivery(Base):
    """Persistent record of a verified patch delivery to GitHub.

    Delivery and verification are separate domain concepts.
    A delivery can only proceed from a VERIFIED PatchVerification.
    """

    __tablename__ = "github_deliveries"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    github_repository_id: Mapped[UUID] = mapped_column(
        ForeignKey("github_repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id: Mapped[UUID] = mapped_column(
        ForeignKey("patch_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    verification_id: Mapped[UUID] = mapped_column(
        ForeignKey("patch_verifications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Git delivery details
    owner: Mapped[str] = mapped_column(String(255), nullable=False)
    repo: Mapped[str] = mapped_column(String(255), nullable=False)
    base_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    delivery_branch: Mapped[str] = mapped_column(String(255), nullable=False)
    commit_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    pull_request_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pull_request_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Hash verification — ensures delivered patch == verified patch
    verified_patch_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    delivered_patch_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # pending | preparing | final_verification | branch_created | committing |
    # pushing | pr_created | completed | failed | aborted
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship("Project")
    github_repository: Mapped[GitHubRepository] = relationship(
        "GitHubRepository", back_populates="deliveries"
    )
