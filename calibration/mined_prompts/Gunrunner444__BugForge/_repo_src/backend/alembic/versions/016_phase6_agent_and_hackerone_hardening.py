"""Phase 5 hardening columns, persisted intents/attachments, and research agent.

Revision ID: 016
Revises: 015
Create Date: 2026-09-20 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "hackerone_programs", sa.Column("scope_content_hash", sa.String(length=64), nullable=False, server_default="")
    )
    op.add_column("hackerone_programs", sa.Column("weaknesses_synced_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "hackerone_programs", sa.Column("scope_pages_fetched", sa.Integer(), nullable=False, server_default="0")
    )

    op.add_column("hackerone_report_drafts", sa.Column("submission_result", sa.JSON(), nullable=True))
    op.add_column(
        "hackerone_report_drafts",
        sa.Column("remote_state_raw", sa.String(length=128), nullable=False, server_default=""),
    )
    op.add_column("hackerone_report_drafts", sa.Column("claim_token", sa.String(length=64), nullable=True))
    op.add_column("hackerone_report_drafts", sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True))

    op.add_column(
        "hackerone_submissions",
        sa.Column("remote_state_raw", sa.String(length=128), nullable=False, server_default=""),
    )
    op.add_column("hackerone_submissions", sa.Column("claim_token", sa.String(length=64), nullable=True))
    op.add_column("hackerone_submissions", sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("hackerone_submissions", sa.Column("http_status", sa.Integer(), nullable=True))

    op.add_column(
        "hackerone_report_intents", sa.Column("title", sa.String(length=512), nullable=False, server_default="")
    )
    op.add_column(
        "hackerone_report_intents",
        sa.Column("vulnerability_information", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column("hackerone_report_intents", sa.Column("impact", sa.Text(), nullable=False, server_default=""))
    op.add_column("hackerone_report_intents", sa.Column("severity", sa.String(length=32), nullable=True))
    op.add_column(
        "hackerone_report_intents",
        sa.Column("local_status", sa.String(length=64), nullable=False, server_default="local_draft"),
    )
    op.add_column("hackerone_report_intents", sa.Column("remote_intent_id", sa.String(length=64), nullable=True))
    op.add_column(
        "hackerone_report_intents",
        sa.Column("remote_state", sa.String(length=64), nullable=False, server_default="not_fetched"),
    )
    op.add_column(
        "hackerone_report_intents",
        sa.Column("remote_state_raw", sa.String(length=128), nullable=False, server_default=""),
    )
    op.add_column("hackerone_report_intents", sa.Column("error", sa.Text(), nullable=True))
    op.add_column("hackerone_report_intents", sa.Column("last_payload", sa.JSON(), nullable=True))
    op.add_column(
        "hackerone_report_intents",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_hackerone_report_intents_draft_id", "hackerone_report_intents", ["draft_id"])

    op.add_column(
        "hackerone_attachments", sa.Column("project_id", sa.String(length=128), nullable=False, server_default="")
    )
    op.add_column(
        "hackerone_attachments", sa.Column("finding_id", sa.String(length=64), nullable=False, server_default="")
    )
    op.add_column(
        "hackerone_attachments", sa.Column("draft_id", sa.String(length=64), nullable=False, server_default="")
    )
    op.add_column("hackerone_attachments", sa.Column("stored_path", sa.Text(), nullable=False, server_default=""))
    op.add_column(
        "hackerone_attachments",
        sa.Column("detected_content_type", sa.String(length=128), nullable=False, server_default=""),
    )
    op.add_column(
        "hackerone_attachments",
        sa.Column("review_state", sa.String(length=64), nullable=False, server_default="received"),
    )
    op.add_column("hackerone_attachments", sa.Column("remote_attachment_id", sa.String(length=64), nullable=True))
    op.add_column(
        "hackerone_attachments",
        sa.Column("upload_state", sa.String(length=64), nullable=False, server_default="received"),
    )
    op.add_column("hackerone_attachments", sa.Column("error", sa.Text(), nullable=True))
    op.add_column("hackerone_attachments", sa.Column("reviewed_by", sa.String(length=255), nullable=True))
    op.add_column("hackerone_attachments", sa.Column("authorized_by", sa.String(length=255), nullable=True))
    op.add_column("hackerone_attachments", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_hackerone_attachments_project_id", "hackerone_attachments", ["project_id"])

    op.create_table(
        "security_research_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("program_handle", sa.String(length=255), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("model_provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_config", sa.JSON(), nullable=True),
        sa.Column("budget", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("paused_reason", sa.Text(), nullable=True),
        sa.Column("human_overrides", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_security_research_sessions_project_id", "security_research_sessions", ["project_id"])

    op.create_table(
        "security_research_hypotheses",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("vulnerability_class", sa.String(length=128), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(length=32), nullable=False),
        sa.Column("supporting_evidence_ids", sa.JSON(), nullable=True),
        sa.Column("contradicting_evidence_ids", sa.JSON(), nullable=True),
        sa.Column("suggested_next_action", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_security_research_hypotheses_session_id", "security_research_hypotheses", ["session_id"])

    op.create_table(
        "security_research_tool_calls",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("tool", sa.String(length=128), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("authorization", sa.String(length=32), nullable=False),
        sa.Column("authorization_reason", sa.Text(), nullable=False),
        sa.Column("result_summary", sa.Text(), nullable=False),
        sa.Column("evidence_ids", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_security_research_tool_calls_session_id", "security_research_tool_calls", ["session_id"])

    op.create_table(
        "security_research_timeline",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("tool", sa.String(length=128), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("authorization", sa.String(length=32), nullable=False),
        sa.Column("result", sa.Text(), nullable=False),
        sa.Column("evidence_id", sa.String(length=64), nullable=False),
        sa.Column("finding_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_security_research_timeline_session_id", "security_research_timeline", ["session_id"])

    op.create_table(
        "security_research_evidence_links",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("hypothesis_id", sa.String(length=64), nullable=False),
        sa.Column("evidence_id", sa.String(length=64), nullable=False),
        sa.Column("tool_call_id", sa.String(length=64), nullable=False),
        sa.Column("finding_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("provenance", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["security_research_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_security_research_evidence_links_session_id", "security_research_evidence_links", ["session_id"]
    )

    op.create_table(
        "security_reproduction_plans",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("hypothesis_id", sa.String(length=64), nullable=False),
        sa.Column("finding_id", sa.String(length=64), nullable=False),
        sa.Column("preconditions", sa.JSON(), nullable=True),
        sa.Column("setup", sa.Text(), nullable=False),
        sa.Column("actions", sa.JSON(), nullable=True),
        sa.Column("expected_result", sa.Text(), nullable=False),
        sa.Column("actual_result", sa.Text(), nullable=False),
        sa.Column("evidence_to_collect", sa.JSON(), nullable=True),
        sa.Column("cleanup", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_security_reproduction_plans_session_id", "security_reproduction_plans", ["session_id"])


def downgrade() -> None:
    op.drop_table("security_reproduction_plans")
    op.drop_table("security_research_evidence_links")
    op.drop_table("security_research_timeline")
    op.drop_table("security_research_tool_calls")
    op.drop_table("security_research_hypotheses")
    op.drop_table("security_research_sessions")
    op.drop_index("ix_hackerone_attachments_project_id", table_name="hackerone_attachments")
    for column in (
        "updated_at",
        "authorized_by",
        "reviewed_by",
        "error",
        "upload_state",
        "remote_attachment_id",
        "review_state",
        "detected_content_type",
        "stored_path",
        "draft_id",
        "finding_id",
        "project_id",
    ):
        op.drop_column("hackerone_attachments", column)
    op.drop_index("ix_hackerone_report_intents_draft_id", table_name="hackerone_report_intents")
    for column in (
        "updated_at",
        "last_payload",
        "error",
        "remote_state_raw",
        "remote_state",
        "remote_intent_id",
        "local_status",
        "severity",
        "impact",
        "vulnerability_information",
        "title",
    ):
        op.drop_column("hackerone_report_intents", column)
    op.drop_column("hackerone_submissions", "http_status")
    op.drop_column("hackerone_submissions", "claim_expires_at")
    op.drop_column("hackerone_submissions", "claim_token")
    op.drop_column("hackerone_submissions", "remote_state_raw")
    op.drop_column("hackerone_report_drafts", "claim_expires_at")
    op.drop_column("hackerone_report_drafts", "claim_token")
    op.drop_column("hackerone_report_drafts", "remote_state_raw")
    op.drop_column("hackerone_report_drafts", "submission_result")
    op.drop_column("hackerone_programs", "scope_pages_fetched")
    op.drop_column("hackerone_programs", "weaknesses_synced_at")
    op.drop_column("hackerone_programs", "scope_content_hash")
