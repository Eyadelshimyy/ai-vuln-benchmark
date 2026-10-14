"""sites, plan sets, immutable plan-set versions and upload custody

Revision ID: 0007_sites_and_plan_set_custody
Revises: 0006_field_note_review_revisions
"""
# ruff: noqa: E501

import sqlalchemy as sa
from alembic import op

# fmt: off
revision = "0007_sites_and_plan_set_custody"
down_revision = "0006_field_note_review_revisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sites",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("external_ref", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"],
            name=op.f("fk_sites_business_id_businesses"), ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sites")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_sites_business_id_id")),
        sa.UniqueConstraint(
            "business_id", "external_ref", name=op.f("uq_sites_business_id_external_ref"),
        ),
    )
    op.create_index(op.f("ix_sites_business_id"), "sites", ["business_id"])
    op.create_table(
        "site_plan_sets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("site_id", sa.Uuid(), nullable=False),
        sa.Column("plan_set_key", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"],
            name=op.f("fk_site_plan_sets_business_id_businesses"), ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "site_id"], ["sites.business_id", "sites.id"],
            name=op.f("fk_site_plan_sets_business_id_site_id_sites"), ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_site_plan_sets")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_site_plan_sets_business_id_id")),
        sa.UniqueConstraint(
            "business_id", "site_id", "plan_set_key", name=op.f("uq_site_plan_sets_site_key"),
        ),
    )
    op.create_index(op.f("ix_site_plan_sets_business_id"), "site_plan_sets", ["business_id"])
    op.create_table(
        "site_plan_set_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("site_id", sa.Uuid(), nullable=False),
        sa.Column("plan_set_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("media_kind", sa.String(length=50), nullable=False),
        sa.Column("artifact_locator", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("filename", sa.String(length=500), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("content_digest", sa.String(length=64), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"],
            name=op.f("fk_site_plan_set_versions_business_id_businesses"), ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "plan_set_id"], ["site_plan_sets.business_id", "site_plan_sets.id"],
            name=op.f("fk_site_plan_set_versions_business_id_plan_set_id_site_plan_sets"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "site_id"], ["sites.business_id", "sites.id"],
            name=op.f("fk_site_plan_set_versions_business_id_site_id_sites"), ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_site_plan_set_versions")),
        sa.UniqueConstraint(
            "business_id", "id", name=op.f("uq_site_plan_set_versions_business_id_id"),
        ),
        sa.UniqueConstraint(
            "plan_set_id", "version", name=op.f("uq_site_plan_set_versions_plan_set_id_version"),
        ),
        sa.UniqueConstraint(
            "plan_set_id", "content_digest",
            name=op.f("uq_site_plan_set_versions_plan_set_id_content_digest"),
        ),
    )
    op.create_index(
        op.f("ix_site_plan_set_versions_plan_set_id"), "site_plan_set_versions", ["plan_set_id"],
    )
    op.create_table(
        "site_plan_set_uploads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("site_id", sa.Uuid(), nullable=False),
        sa.Column("plan_set_id", sa.Uuid(), nullable=False),
        sa.Column("source_attachment_id", sa.Uuid(), nullable=False),
        sa.Column("source_media_kind", sa.String(length=50), nullable=False),
        sa.Column("source_locator", sa.Text(), nullable=False),
        sa.Column("source_mime_type", sa.String(length=255), nullable=True),
        sa.Column("source_filename", sa.String(length=500), nullable=True),
        sa.Column("source_byte_size", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=True),
        sa.Column("leased_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_token", sa.Uuid(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"],
            name=op.f("fk_site_plan_set_uploads_business_id_businesses"), ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "plan_set_id"], ["site_plan_sets.business_id", "site_plan_sets.id"],
            name=op.f("fk_site_plan_set_uploads_business_id_plan_set_id_site_plan_sets"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "site_id"], ["sites.business_id", "sites.id"],
            name=op.f("fk_site_plan_set_uploads_business_id_site_id_sites"), ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_site_plan_set_uploads")),
        sa.UniqueConstraint(
            "business_id", "id", name=op.f("uq_site_plan_set_uploads_business_id_id"),
        ),
        sa.UniqueConstraint(
            "business_id", "plan_set_id", "source_attachment_id",
            name=op.f("uq_site_plan_set_uploads_plan_set_source"),
        ),
    )
    op.create_index(
        op.f("ix_site_plan_set_uploads_status"), "site_plan_set_uploads", ["status"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_site_plan_set_uploads_status"), table_name="site_plan_set_uploads")
    op.drop_table("site_plan_set_uploads")
    op.drop_index(
        op.f("ix_site_plan_set_versions_plan_set_id"), table_name="site_plan_set_versions",
    )
    op.drop_table("site_plan_set_versions")
    op.drop_index(op.f("ix_site_plan_sets_business_id"), table_name="site_plan_sets")
    op.drop_table("site_plan_sets")
    op.drop_index(op.f("ix_sites_business_id"), table_name="sites")
    op.drop_table("sites")
