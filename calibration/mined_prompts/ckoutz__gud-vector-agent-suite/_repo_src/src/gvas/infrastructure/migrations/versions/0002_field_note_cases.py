"""field note cases and transcription lifecycle

Revision ID: 0002_field_note_cases
Revises: 0001_initial_shared_records
"""
# ruff: noqa: E501

import sqlalchemy as sa
from alembic import op

# fmt: off
revision = "0002_field_note_cases"
down_revision = "0001_initial_shared_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "field_note_cases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("origin_inbound_message_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"],
            name=op.f("fk_field_note_cases_business_id_businesses"), ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            name=op.f("fk_field_note_cases_business_id_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "origin_inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            name=op.f("fk_field_note_cases_business_id_origin_inbound_message_id_inbound_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_cases")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_field_note_cases_business_id_id")),
        sa.UniqueConstraint(
            "origin_inbound_message_id",
            name=op.f("uq_field_note_cases_origin_inbound_message_id"),
        ),
    )
    op.create_index(
        op.f("ix_field_note_cases_conversation_id"),
        "field_note_cases",
        ["conversation_id"],
    )
    op.create_table(
        "field_note_parts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("source_inbound_message_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=50), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("attachment_id", sa.Uuid(), nullable=True),
        sa.Column("media_kind", sa.String(length=50), nullable=True),
        sa.Column("attachment_locator", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("filename", sa.String(length=500), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=True),
        sa.Column("transcription_status", sa.String(length=50), nullable=False),
        sa.Column("transcript_text", sa.Text(), nullable=True),
        sa.Column("transcript_language", sa.String(length=50), nullable=True),
        sa.Column("transcript_confidence", sa.Float(), nullable=True),
        sa.Column("transcript_duration_seconds", sa.Float(), nullable=True),
        sa.Column("transcript_provider_ref", sa.String(length=255), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("leased_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_token", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"],
            name=op.f("fk_field_note_parts_business_id_businesses"), ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "case_id"],
            ["field_note_cases.business_id", "field_note_cases.id"],
            name=op.f("fk_field_note_parts_business_id_case_id_field_note_cases"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "source_inbound_message_id"],
            ["inbound_messages.business_id", "inbound_messages.id"],
            name=op.f("fk_field_note_parts_business_id_source_inbound_message_id_inbound_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_parts")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_field_note_parts_business_id_id")),
        sa.UniqueConstraint(
            "case_id", "sequence", name=op.f("uq_field_note_parts_case_id_sequence")
        ),
        sa.UniqueConstraint(
            "case_id", "source_inbound_message_id", "sequence",
            name=op.f("uq_field_note_parts_case_source_sequence"),
        ),
    )
    op.create_index(
        op.f("ix_field_note_parts_case_sequence"),
        "field_note_parts",
        ["case_id", "sequence"],
    )
    op.create_index(
        op.f("ix_field_note_parts_transcription_status"),
        "field_note_parts",
        ["transcription_status"],
    )
    op.create_table(
        "field_note_conversation_states",
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("active_case_id", sa.Uuid(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"],
            name=op.f("fk_field_note_conversation_states_business_id_businesses"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            name=op.f("fk_field_note_conversation_states_business_id_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("conversation_id", name=op.f("pk_field_note_conversation_states")),
    )
    op.create_index(
        op.f("ix_field_note_conversation_states_business_id"),
        "field_note_conversation_states",
        ["business_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_field_note_conversation_states_business_id"),
        table_name="field_note_conversation_states",
    )
    op.drop_table("field_note_conversation_states")
    op.drop_index(
        op.f("ix_field_note_parts_transcription_status"), table_name="field_note_parts"
    )
    op.drop_index(op.f("ix_field_note_parts_case_sequence"), table_name="field_note_parts")
    op.drop_table("field_note_parts")
    op.drop_index(op.f("ix_field_note_cases_conversation_id"), table_name="field_note_cases")
    op.drop_table("field_note_cases")
