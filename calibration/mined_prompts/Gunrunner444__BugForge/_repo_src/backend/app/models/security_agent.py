"""Persisted guided security-research agent sessions. Secrets are never stored."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class DBResearchSession(Base):
    __tablename__ = "security_research_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True
    )
    program_handle: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    target: Mapped[str] = mapped_column(Text, nullable=False, default="")
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="lab")
    state: Mapped[str] = mapped_column(String(64), nullable=False, default="created")
    model_provider: Mapped[str] = mapped_column(String(64), nullable=False, default="mock")
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    model_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    budget: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    paused_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    human_overrides: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    termination_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    privilege_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    disabled_tools: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    repo_root: Mapped[str] = mapped_column(Text, nullable=False, default=".")
    stopped: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    exchanges: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    operator_identity: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    research_project_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    hypotheses: Mapped[list[DBResearchHypothesis]] = relationship(
        "DBResearchHypothesis", back_populates="session", cascade="all, delete-orphan"
    )
    tool_calls: Mapped[list[DBResearchToolCall]] = relationship(
        "DBResearchToolCall", back_populates="session", cascade="all, delete-orphan"
    )
    timeline: Mapped[list[DBResearchTimelineEvent]] = relationship(
        "DBResearchTimelineEvent", back_populates="session", cascade="all, delete-orphan"
    )
    evidence_links: Mapped[list[DBResearchEvidenceLink]] = relationship(
        "DBResearchEvidenceLink", back_populates="session", cascade="all, delete-orphan"
    )
    evidence_nodes: Mapped[list[DBResearchEvidenceNode]] = relationship(
        "DBResearchEvidenceNode", back_populates="session", cascade="all, delete-orphan"
    )
    evidence_edges: Mapped[list[DBResearchEvidenceEdge]] = relationship(
        "DBResearchEvidenceEdge", back_populates="session", cascade="all, delete-orphan"
    )
    findings: Mapped[list[DBResearchFinding]] = relationship(
        "DBResearchFinding", back_populates="session", cascade="all, delete-orphan"
    )
    memories: Mapped[list[DBResearchMemory]] = relationship(
        "DBResearchMemory", back_populates="session", cascade="all, delete-orphan"
    )
    checkpoints: Mapped[list[DBResearchCheckpoint]] = relationship(
        "DBResearchCheckpoint", back_populates="session", cascade="all, delete-orphan"
    )
    identities: Mapped[list[DBResearchIdentity]] = relationship(
        "DBResearchIdentity", back_populates="session", cascade="all, delete-orphan"
    )
    exploratory_attempts: Mapped[list[DBResearchExploratoryAttempt]] = relationship(
        "DBResearchExploratoryAttempt", back_populates="session", cascade="all, delete-orphan"
    )
    research_project: Mapped[DBResearchProject | None] = relationship(
        "DBResearchProject", back_populates="session", uselist=False
    )


class DBResearchHypothesis(Base):
    __tablename__ = "security_research_hypotheses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    vulnerability_class: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    target: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence: Mapped[str] = mapped_column(String(32), nullable=False, default="low")
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    impact: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reproducibility: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    evidence_strength: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    supporting_evidence_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    contradicting_evidence_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    suggested_next_action: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="hypotheses"
    )


class DBResearchToolCall(Base):
    __tablename__ = "security_research_tool_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool: Mapped[str] = mapped_column(String(128), nullable=False)
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    authorization: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    authorization_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    result_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    execution_state: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    result_quality: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    evidence_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="tool_calls"
    )


class DBResearchTimelineEvent(Base):
    __tablename__ = "security_research_timeline"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    target: Mapped[str] = mapped_column(Text, nullable=False, default="")
    authorization: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    result: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    finding_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="timeline"
    )


class DBResearchEvidenceLink(Base):
    __tablename__ = "security_research_evidence_links"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    evidence_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    tool_call_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    finding_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    kind: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    provenance: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="evidence_links"
    )


class DBReproductionPlan(Base):
    __tablename__ = "security_reproduction_plans"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    finding_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    preconditions: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    setup: Mapped[str] = mapped_column(Text, nullable=False, default="")
    actions: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    expected_result: Mapped[str] = mapped_column(Text, nullable=False, default="")
    actual_result: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence_to_collect: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    cleanup: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="planned")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class DBResearchEvidenceNode(Base):
    __tablename__ = "research_evidence_node"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    provenance: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="evidence_nodes"
    )


class DBResearchEvidenceEdge(Base):
    __tablename__ = "research_evidence_edge"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_node: Mapped[str] = mapped_column(String(64), nullable=False)
    destination_node: Mapped[str] = mapped_column(String(64), nullable=False)
    relation: Mapped[str] = mapped_column(String(64), nullable=False, default="supports")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="evidence_edges"
    )


class DBResearchFinding(Base):
    __tablename__ = "research_findings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="potential")
    vulnerability_class: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    target: Mapped[str] = mapped_column(Text, nullable=False, default="")
    verification_state: Mapped[str] = mapped_column(String(64), nullable=False, default="potential")
    evidence_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    reproduction_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[str] = mapped_column(String(32), nullable=False, default="low")
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    impact: Mapped[str] = mapped_column(Text, nullable=False, default="")
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="findings"
    )


class DBResearchMemory(Base):
    __tablename__ = "research_memory"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    project_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False, default="observation")
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession | None] = relationship(
        "DBResearchSession", back_populates="memories"
    )


class DBResearchCheckpoint(Base):
    __tablename__ = "research_checkpoints"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="checkpoints"
    )


class DBResearchIdentity(Base):
    __tablename__ = "research_identities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(32), nullable=False, default="A")
    credential_ref: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    browser_context_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    http_session_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    storage_namespace: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    authentication_state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unauthenticated"
    )
    credential_provenance: Mapped[str] = mapped_column(String(64), nullable=False, default="none")
    header_names: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    header_secret_refs: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)
    cookie_names: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    cookie_secret_ref: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    storage_keys: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    storage_secret_ref: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="identities"
    )


class DBResearchProject(Base):
    __tablename__ = "security_research_projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: uuid4().hex)
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    project_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    session_id: Mapped[str | None] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    program_handle: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    target: Mapped[str] = mapped_column(Text, nullable=False, default="")
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="lab")
    strategy: Mapped[str] = mapped_column(String(64), nullable=False, default="passive_recon")
    state: Mapped[str] = mapped_column(String(64), nullable=False, default="create")
    operator_identity: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession | None] = relationship(
        "DBResearchSession", back_populates="research_project"
    )


class DBResearchExploratoryAttempt(Base):
    """One exploratory sandbox attempt. Host secrets are not stored."""

    __tablename__ = "research_exploratory_attempts"

    attempt_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("security_research_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    hypothesis_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    parent_attempt_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    candidate_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    test_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    target_file: Mapped[str] = mapped_column(Text, nullable=False, default="")
    target_symbol: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    language: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    framework: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    test_code: Mapped[str] = mapped_column(Text, nullable=False, default="")
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    expected_behavior: Mapped[str] = mapped_column(Text, nullable=False, default="")
    oracle: Mapped[str] = mapped_column(String(40), nullable=False, default="")
    confidence: Mapped[str] = mapped_column(String(16), nullable=False, default="low")
    repository_snapshot: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    repository_commit: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    execution_profile: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    command_identity: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    classification: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    stdout_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    stderr_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    artifacts: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    duration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    timeout: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    executed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    meaningful: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    follow_up_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    repeat_classification: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    follow_up: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    replayable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    session: Mapped[DBResearchSession] = relationship(
        "DBResearchSession", back_populates="exploratory_attempts"
    )


class DBResearchLead(Base):
    """Persistent research lead. Status is not a verification state."""

    __tablename__ = "research_leads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    target: Mapped[str] = mapped_column(Text, nullable=False, default="")
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="NEW")
    priority: Mapped[str] = mapped_column(String(32), nullable=False, default="medium")
    next_action: Mapped[str] = mapped_column(Text, nullable=False, default="")
    kill_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    observation_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    related_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    chain_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    hypothesis_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    finding_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    semantic_node_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
