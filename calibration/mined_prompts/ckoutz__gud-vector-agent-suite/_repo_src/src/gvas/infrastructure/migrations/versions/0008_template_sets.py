"""per-business template sets and the case-level template pin

Revision ID: 0008_template_sets
Revises: 0007_sites_and_plan_set_custody
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008_template_sets"
down_revision = "0007_sites_and_plan_set_custody"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "field_note_report_templates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("report_template_key", sa.String(length=255), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("sections", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_field_note_report_templates"),
        sa.ForeignKeyConstraint(
            ["business_id"],
            ["businesses.id"],
            name="fk_field_note_report_templates_business_id_businesses",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "business_id",
            "report_template_key",
            "version",
            name="uq_field_note_report_templates_key_version",
        ),
    )
    op.create_index(
        "ix_field_note_report_templates_business_id",
        "field_note_report_templates",
        ["business_id"],
    )
    op.create_table(
        "field_note_template_sets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("template_set_key", sa.String(length=255), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("industry_key", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("checklist_key", sa.String(length=255), nullable=False),
        sa.Column("checklist_version", sa.Integer(), nullable=False),
        sa.Column("report_template_key", sa.String(length=255), nullable=False),
        sa.Column("report_template_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_field_note_template_sets"),
        sa.ForeignKeyConstraint(
            ["business_id"],
            ["businesses.id"],
            name="fk_field_note_template_sets_business_id_businesses",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "checklist_key", "checklist_version"],
            [
                "field_note_checklists.business_id",
                "field_note_checklists.checklist_key",
                "field_note_checklists.version",
            ],
            name="fk_field_note_template_sets_checklist",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "report_template_key", "report_template_version"],
            [
                "field_note_report_templates.business_id",
                "field_note_report_templates.report_template_key",
                "field_note_report_templates.version",
            ],
            name="fk_field_note_template_sets_report_template",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "business_id",
            "template_set_key",
            "version",
            name="uq_field_note_template_sets_key_version",
        ),
    )
    op.create_index(
        "ix_field_note_template_sets_business_id",
        "field_note_template_sets",
        ["business_id"],
    )
    op.create_index(
        "uq_field_note_template_sets_active",
        "field_note_template_sets",
        ["business_id", "template_set_key"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
        sqlite_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "business_template_profiles",
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("industry_key", sa.String(length=255), nullable=False),
        sa.Column("template_set_key", sa.String(length=255), nullable=False),
        sa.Column("default_template_set_key", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("business_id", name="pk_business_template_profiles"),
        sa.ForeignKeyConstraint(
            ["business_id"],
            ["businesses.id"],
            name="fk_business_template_profiles_business_id_businesses",
            ondelete="CASCADE",
        ),
    )
    op.add_column(
        "field_note_reviews",
        sa.Column("template_set_key", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "field_note_reviews",
        sa.Column("template_set_version", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_field_note_reviews_template_set",
        "field_note_reviews",
        "field_note_template_sets",
        ["business_id", "template_set_key", "template_set_version"],
        ["business_id", "template_set_key", "version"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_field_note_reviews_template_set", "field_note_reviews", type_="foreignkey"
    )
    op.drop_column("field_note_reviews", "template_set_version")
    op.drop_column("field_note_reviews", "template_set_key")
    op.drop_table("business_template_profiles")
    op.drop_index("uq_field_note_template_sets_active", table_name="field_note_template_sets")
    op.drop_index("ix_field_note_template_sets_business_id", table_name="field_note_template_sets")
    op.drop_table("field_note_template_sets")
    op.drop_index(
        "ix_field_note_report_templates_business_id",
        table_name="field_note_report_templates",
    )
    op.drop_table("field_note_report_templates")
