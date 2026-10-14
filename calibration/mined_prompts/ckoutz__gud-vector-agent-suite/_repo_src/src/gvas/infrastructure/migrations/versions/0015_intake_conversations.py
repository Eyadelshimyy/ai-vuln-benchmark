"""intake conversations: website booking agent chats and transcripts

Revision ID: 0015_intake_conversations
Revises: 0014_customer_portal
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0015_intake_conversations"
down_revision = "0014_customer_portal"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _tenant_business_column() -> sa.Column[sa.Uuid]:
    return sa.Column(
        "business_id",
        sa.Uuid(),
        sa.ForeignKey("businesses.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "intake_conversations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant_business_column(),
        sa.Column("customer_id", sa.Uuid(), nullable=True),
        sa.Column("reference", sa.String(length=16), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("channel", sa.String(length=20), nullable=False),
        sa.Column("state", sa.String(length=30), nullable=False),
        sa.Column("collected", json_type, nullable=False),
        sa.Column("proposed_slots", json_type, nullable=True),
        sa.Column("requested_slot_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("requested_slot_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("booking_kind", sa.String(length=20), nullable=True),
        sa.Column("booking_link", sa.String(length=2048), nullable=True),
        sa.Column("booking_attempted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("booked_event_uri", sa.String(length=2048), nullable=True),
        sa.Column("decision_reason", sa.String(length=500), nullable=True),
        sa.Column("decision_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("owner_notified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("escalation_notified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["customer_id"], ["customers.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("business_id", "id", name="uq_intake_conversations_business_id_id"),
        sa.UniqueConstraint(
            "business_id", "reference", name="uq_intake_conversations_business_id_reference"
        ),
    )
    op.create_index("ix_intake_conversations_business_id", "intake_conversations", ["business_id"])
    op.create_index(
        "ix_intake_conversations_created",
        "intake_conversations",
        ["business_id", "created_at"],
    )

    op.create_table(
        "intake_messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _tenant_business_column(),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=10), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["intake_conversations.business_id", "intake_conversations.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_intake_messages_conversation_id", "intake_messages", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_intake_messages_conversation_id", table_name="intake_messages")
    op.drop_table("intake_messages")
    op.drop_index("ix_intake_conversations_created", table_name="intake_conversations")
    op.drop_index("ix_intake_conversations_business_id", table_name="intake_conversations")
    op.drop_table("intake_conversations")
