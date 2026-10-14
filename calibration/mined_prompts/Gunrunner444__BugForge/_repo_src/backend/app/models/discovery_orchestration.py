"""Persistent Phase 49 research orchestration state.

The campaign row holds the canonical state document. Decision rows are
insert-only history. Neither table stores an approval, a credential, or raw
tool output.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


def _now() -> datetime:
    return datetime.now(UTC)


class DBOrchestrationCampaign(Base):
    __tablename__ = "discovery_orchestration_campaigns"

    campaign_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project: Mapped[str] = mapped_column(String(256), nullable=False, default="", index=True)
    target_digest: Mapped[str] = mapped_column(String(32), nullable=False, default="", index=True)
    source_snapshot: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    compiler_configuration: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    orchestrator_version: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    state: Mapped[str] = mapped_column(String(24), nullable=False)
    stop_reason: Mapped[str] = mapped_column(String(48), nullable=False, default="")
    round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class DBOrchestrationDecision(Base):
    __tablename__ = "discovery_orchestration_decisions"
    __table_args__ = (
        UniqueConstraint("campaign_id", "sequence", name="uq_orchestration_decision_sequence"),
    )

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("discovery_orchestration_campaigns.campaign_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    record_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class DBBountyCampaign(Base):
    """Operator inputs and campaign-level state for a Phase 51 bounty campaign.

    One row per orchestration campaign. It stores what is needed to rebuild the
    campaign after a restart (manifest, spec, operator, approvals) plus the
    campaign's control state, background-job progress, source selection, and
    VFCS feedback ledger. It holds no credential and no raw tool output. The
    orchestrator's own state stays in ``discovery_orchestration_campaigns``.
    """

    __tablename__ = "discovery_bounty_campaigns"

    campaign_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("discovery_orchestration_campaigns.campaign_id", ondelete="CASCADE"),
        primary_key=True,
    )
    operator_identity: Mapped[str] = mapped_column(
        String(128), nullable=False, default="", index=True
    )
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    program_context: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    manifest: Mapped[str] = mapped_column(Text, nullable=False)
    spec: Mapped[str] = mapped_column(Text, nullable=False)
    approvals: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    control: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    control_reason: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    job: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    artifacts: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
