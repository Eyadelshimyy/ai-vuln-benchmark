"""HackerOne production-readiness tables and finding target/reproduction fields.

Revision ID: 015
Revises: 014
Create Date: 2026-09-20 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("security_findings", sa.Column("target", sa.Text(), nullable=True))
    op.add_column("security_findings", sa.Column("endpoint", sa.Text(), nullable=True))
    op.add_column("security_findings", sa.Column("reproduction", sa.Text(), nullable=True))
    op.add_column("security_findings", sa.Column("observed_behavior", sa.Text(), nullable=True))
    op.add_column("security_findings", sa.Column("expected_behavior", sa.Text(), nullable=True))

    op.create_table(
        "hackerone_programs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("handle", sa.String(length=255), nullable=False),
        sa.Column("program_id", sa.String(length=64), nullable=True),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("program_url", sa.Text(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sync_status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("scope_mode", sa.String(length=32), nullable=False),
        sa.Column("offers_bounties", sa.Boolean(), nullable=True),
        sa.Column("requires_severity", sa.Boolean(), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=False),
        sa.Column("open_scope_policy", sa.Text(), nullable=False),
        sa.Column("open_scope_acknowledged", sa.Boolean(), nullable=False),
        sa.Column("active_testing_approved", sa.Boolean(), nullable=False),
        sa.Column("scope_version", sa.Integer(), nullable=False),
        sa.Column("scope_sync_complete", sa.Boolean(), nullable=False),
        sa.Column("scope_count", sa.Integer(), nullable=False),
        sa.Column("last_scope_id", sa.String(length=64), nullable=True),
        sa.Column("continuation_state", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("handle"),
    )
    op.create_index("ix_hackerone_programs_handle", "hackerone_programs", ["handle"])

    op.create_table(
        "hackerone_structured_scopes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("program_pk", sa.Uuid(), nullable=False),
        sa.Column("hackerone_id", sa.String(length=64), nullable=False),
        sa.Column("asset_type_raw", sa.String(length=64), nullable=False),
        sa.Column("asset_type", sa.String(length=64), nullable=False),
        sa.Column("asset_identifier", sa.Text(), nullable=False),
        sa.Column("instruction", sa.Text(), nullable=False),
        sa.Column("eligible_for_bounty", sa.Boolean(), nullable=False),
        sa.Column("eligible_for_submission", sa.Boolean(), nullable=False),
        sa.Column("reference", sa.Text(), nullable=True),
        sa.Column("original", sa.JSON(), nullable=True),
        sa.Column("snapshot_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["program_pk"], ["hackerone_programs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_hackerone_structured_scopes_program_pk",
        "hackerone_structured_scopes",
        ["program_pk"],
    )

    op.create_table(
        "hackerone_scope_exclusions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("program_pk", sa.Uuid(), nullable=False),
        sa.Column("hackerone_id", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=255), nullable=False),
        sa.Column("details", sa.Text(), nullable=False),
        sa.Column("created_at_remote", sa.String(length=64), nullable=True),
        sa.Column("updated_at_remote", sa.String(length=64), nullable=True),
        sa.Column("original", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["program_pk"], ["hackerone_programs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_hackerone_scope_exclusions_program_pk",
        "hackerone_scope_exclusions",
        ["program_pk"],
    )

    op.create_table(
        "hackerone_weaknesses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("program_pk", sa.Uuid(), nullable=False),
        sa.Column("hackerone_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("external_id", sa.String(length=64), nullable=False),
        sa.Column("original", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["program_pk"], ["hackerone_programs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hackerone_weaknesses_program_pk", "hackerone_weaknesses", ["program_pk"])

    op.create_table(
        "hackerone_syncs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("program_pk", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("scope_count", sa.Integer(), nullable=False),
        sa.Column("last_scope_id", sa.String(length=64), nullable=True),
        sa.Column("scope_sync_complete", sa.Boolean(), nullable=False),
        sa.Column("weakness_count", sa.Integer(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["program_pk"], ["hackerone_programs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hackerone_syncs_program_pk", "hackerone_syncs", ["program_pk"])

    op.create_table(
        "hackerone_report_drafts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("finding_id", sa.Uuid(), nullable=True),
        sa.Column("program_handle", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("vulnerability_information", sa.Text(), nullable=False),
        sa.Column("impact", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=True),
        sa.Column("weakness_id", sa.Integer(), nullable=True),
        sa.Column("weakness_candidates", sa.JSON(), nullable=True),
        sa.Column("structured_scope_id", sa.Integer(), nullable=True),
        sa.Column("evidence_references", sa.JSON(), nullable=True),
        sa.Column("reproduction", sa.Text(), nullable=True),
        sa.Column("target", sa.Text(), nullable=True),
        sa.Column("eligible_for_submission", sa.Boolean(), nullable=False),
        sa.Column("eligible_for_bounty", sa.Boolean(), nullable=False),
        sa.Column("finding_verification", sa.String(length=64), nullable=True),
        sa.Column("human_review_state", sa.String(length=64), nullable=False),
        sa.Column("submission_state", sa.String(length=64), nullable=False),
        sa.Column("remote_state", sa.String(length=64), nullable=False),
        sa.Column("hackerone_report_id", sa.String(length=64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("last_payload", sa.JSON(), nullable=True),
        sa.Column("report_content_hash", sa.String(length=64), nullable=False),
        sa.Column("evidence_hash", sa.String(length=64), nullable=False),
        sa.Column("scope_snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("approved_report_content_hash", sa.String(length=64), nullable=True),
        sa.Column("approved_evidence_hash", sa.String(length=64), nullable=True),
        sa.Column("approved_scope_snapshot_hash", sa.String(length=64), nullable=True),
        sa.Column("approved_payload_hash", sa.String(length=64), nullable=True),
        sa.Column("approval_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_by", sa.String(length=255), nullable=True),
        sa.Column("approval_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scope_snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["finding_id"], ["security_findings.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hackerone_report_drafts_project_id", "hackerone_report_drafts", ["project_id"])
    op.create_index("ix_hackerone_report_drafts_finding_id", "hackerone_report_drafts", ["finding_id"])
    op.create_index(
        "ix_hackerone_report_drafts_program_handle",
        "hackerone_report_drafts",
        ["program_handle"],
    )

    op.create_table(
        "hackerone_submissions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("draft_id", sa.String(length=64), nullable=True),
        sa.Column("finding_id", sa.String(length=64), nullable=False),
        sa.Column("program_handle", sa.String(length=255), nullable=False),
        sa.Column("hackerone_report_id", sa.String(length=64), nullable=True),
        sa.Column("submission_state", sa.String(length=64), nullable=False),
        sa.Column("remote_state", sa.String(length=64), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["draft_id"], ["hackerone_report_drafts.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "finding_id", "program_handle", name="uq_hackerone_submission_finding_program"
        ),
    )
    op.create_index("ix_hackerone_submissions_draft_id", "hackerone_submissions", ["draft_id"])

    op.create_table(
        "hackerone_approval_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("draft_id", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("approved_by", sa.String(length=255), nullable=False),
        sa.Column("report_content_hash", sa.String(length=64), nullable=False),
        sa.Column("evidence_hash", sa.String(length=64), nullable=False),
        sa.Column("scope_snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_hackerone_approval_events_draft_id", "hackerone_approval_events", ["draft_id"]
    )

    op.create_table(
        "hackerone_audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("project_id", sa.String(length=128), nullable=False),
        sa.Column("finding_id", sa.String(length=64), nullable=False),
        sa.Column("draft_id", sa.String(length=64), nullable=False),
        sa.Column("program_handle", sa.String(length=255), nullable=False),
        sa.Column("operator_identity", sa.String(length=255), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hackerone_audit_events_action", "hackerone_audit_events", ["action"])

    op.create_table(
        "hackerone_report_intents",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("finding_id", sa.Uuid(), nullable=True),
        sa.Column("draft_id", sa.String(length=64), nullable=True),
        sa.Column("program_handle", sa.String(length=255), nullable=False),
        sa.Column("human_review_state", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["finding_id"], ["security_findings.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_hackerone_report_intents_project_id", "hackerone_report_intents", ["project_id"]
    )
    op.create_index(
        "ix_hackerone_report_intents_finding_id", "hackerone_report_intents", ["finding_id"]
    )

    op.create_table(
        "hackerone_attachments",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("intent_id", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("reviewed", sa.Boolean(), nullable=False),
        sa.Column("authorized_for_upload", sa.Boolean(), nullable=False),
        sa.Column("uploaded", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["intent_id"], ["hackerone_report_intents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hackerone_attachments_intent_id", "hackerone_attachments", ["intent_id"])


def downgrade() -> None:
    op.drop_table("hackerone_attachments")
    op.drop_table("hackerone_report_intents")
    op.drop_table("hackerone_audit_events")
    op.drop_table("hackerone_approval_events")
    op.drop_table("hackerone_submissions")
    op.drop_table("hackerone_report_drafts")
    op.drop_table("hackerone_syncs")
    op.drop_table("hackerone_weaknesses")
    op.drop_table("hackerone_scope_exclusions")
    op.drop_table("hackerone_structured_scopes")
    op.drop_table("hackerone_programs")
    op.drop_column("security_findings", "expected_behavior")
    op.drop_column("security_findings", "observed_behavior")
    op.drop_column("security_findings", "reproduction")
    op.drop_column("security_findings", "endpoint")
    op.drop_column("security_findings", "target")
