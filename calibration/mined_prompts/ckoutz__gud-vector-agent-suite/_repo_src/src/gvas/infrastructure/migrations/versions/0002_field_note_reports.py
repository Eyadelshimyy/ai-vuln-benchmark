"""field-note report records and versions

Revision ID: 0002_field_note_reports
Revises: 0001_initial_shared_records
"""
# ruff: noqa: E501

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# fmt: off
revision = "0002_field_note_reports"
down_revision = "0001_initial_shared_records"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "field_note_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("leased_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_token", sa.Uuid(), nullable=False),
        sa.Column("source_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_field_note_reports_business_id_businesses"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_reports")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_field_note_reports_business_id_id")),
        sa.UniqueConstraint("business_id", "case_id", name=op.f("uq_field_note_reports_business_id_case_id")),
    )
    op.create_index(op.f("ix_field_note_reports_business_id"), "field_note_reports", ["business_id"])
    op.create_table(
        "field_note_report_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("report_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.String(length=100), nullable=False),
        sa.Column("source_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("document", json_type, nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["business_id", "report_id"], ["field_note_reports.business_id", "field_note_reports.id"], name=op.f("fk_field_note_report_versions_business_id_report_id_field_note_reports"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_report_versions")),
        sa.UniqueConstraint("report_id", "version", name=op.f("uq_field_note_report_versions_report_id_version")),
        sa.UniqueConstraint("report_id", "source_fingerprint", name=op.f("uq_field_note_report_versions_report_id_source_fingerprint")),
    )
    op.create_index(op.f("ix_field_note_report_versions_report_id"), "field_note_report_versions", ["report_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_field_note_report_versions_report_id"), table_name="field_note_report_versions")
    op.drop_table("field_note_report_versions")
    op.drop_index(op.f("ix_field_note_reports_business_id"), table_name="field_note_reports")
    op.drop_table("field_note_reports")
