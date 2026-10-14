from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis
    from app.models.project import Project


class DBSecurityFinding(Base):
    """Persisted potential/corroborated security finding. Never auto-verified."""

    __tablename__ = "security_findings"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    analysis_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("analyses.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # analysis_id is the latest scan that observed this finding. Reconcile
    # reuses the row and points it at the newer analysis; older analyses are
    # not a historical finding store.
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="potential", index=True)
    vulnerability_class: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    evidence_tier: Mapped[str] = mapped_column(
        String(50), nullable=False, default="static_indicator"
    )
    confidence: Mapped[str] = mapped_column(String(50), nullable=False, default="low")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    impact: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    analyzer: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parser_backend: Mapped[str | None] = mapped_column(String(50), nullable=True)
    node_id: Mapped[str | None] = mapped_column(String(500), nullable=True)
    taint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    rule_ids: Mapped[str] = mapped_column(Text, nullable=False, default="")
    observation_refs: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    intelligence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    finding_key: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    asset: Mapped[str | None] = mapped_column(Text, nullable=True)
    target: Mapped[str | None] = mapped_column(Text, nullable=True)
    endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    reproduction: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_behavior: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "finding_key",
            name="uq_security_findings_project_finding_key",
        ),
    )

    project: Mapped[Project | None] = relationship("Project", back_populates="security_findings")
    analysis: Mapped[Analysis | None] = relationship("Analysis")
