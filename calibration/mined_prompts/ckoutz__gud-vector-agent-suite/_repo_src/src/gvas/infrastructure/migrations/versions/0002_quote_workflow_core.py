"""quote workflow core

Revision ID: 0002_quote_workflow_core
Revises: 0001_initial_shared_records
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_quote_workflow_core"
down_revision = "0001_initial_shared_records"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "quotes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("active_conversation_id", sa.Uuid(), nullable=True),
        sa.Column("external_conversation_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_message_key", sa.String(length=255), nullable=False),
        sa.Column("last_message_key", sa.String(length=255), nullable=False),
        sa.Column("pending_request_text", sa.Text(), nullable=False),
        sa.Column("draft", json_type, nullable=True),
        sa.Column("approval_correlation_id", sa.String(length=255), nullable=True),
        sa.Column("delivery_receipt", json_type, nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"],
            ["businesses.id"],
            name=op.f("fk_quotes_business_id_businesses"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["conversations.business_id", "conversations.id"],
            name=op.f("fk_quotes_business_id_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_quotes")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_quotes_business_id_id")),
        sa.UniqueConstraint(
            "business_id",
            "active_conversation_id",
            name=op.f("uq_quotes_business_id_active_conversation_id"),
        ),
        sa.UniqueConstraint(
            "conversation_id",
            "last_message_key",
            name=op.f("uq_quotes_conversation_id_last_message_key"),
        ),
    )
    op.create_index(op.f("ix_quotes_business_id"), "quotes", ["business_id"])
    op.create_index(op.f("ix_quotes_conversation_id"), "quotes", ["conversation_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_quotes_conversation_id"), table_name="quotes")
    op.drop_index(op.f("ix_quotes_business_id"), table_name="quotes")
    op.drop_table("quotes")
