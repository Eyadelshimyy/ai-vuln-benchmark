"""initial shared records

Revision ID: 0001_initial_shared_records
Revises:
"""
# ruff: noqa: E501

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# fmt: off
revision = "0001_initial_shared_records"
down_revision = None
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "businesses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_businesses")),
        sa.UniqueConstraint("slug", name=op.f("uq_businesses_slug")),
    )
    op.create_table(
        "owner_channel_endpoints",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("source_namespace", sa.String(length=100), nullable=False),
        sa.Column("external_endpoint_id", sa.String(length=255), nullable=False),
        sa.Column("owner_external_id", sa.String(length=255), nullable=True),
        sa.Column("routing", json_type, nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id"], ["businesses.id"], name=op.f("fk_owner_channel_endpoints_business_id_businesses"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_owner_channel_endpoints")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_owner_channel_endpoints_business_id_id")),
        sa.UniqueConstraint("business_id", "source_namespace", "external_endpoint_id", name=op.f("uq_owner_channel_endpoints_source_endpoint")),
    )
    op.create_index(op.f("ix_owner_channel_endpoints_business_id"), "owner_channel_endpoints", ["business_id"])
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("endpoint_id", sa.Uuid(), nullable=False),
        sa.Column("external_conversation_id", sa.String(length=255), nullable=False),
        sa.Column("routing", json_type, nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_conversations_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "endpoint_id"], ["owner_channel_endpoints.business_id", "owner_channel_endpoints.id"], name=op.f("fk_conversations_business_id_endpoint_id_owner_channel_endpoints"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_conversations_business_id_id")),
        sa.UniqueConstraint("endpoint_id", "external_conversation_id", name=op.f("uq_conversations_endpoint_id")),
    )
    op.create_index(op.f("ix_conversations_endpoint_id"), "conversations", ["endpoint_id"])
    op.create_table(
        "inbound_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("endpoint_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("message_key", sa.String(length=255), nullable=False),
        sa.Column("sender_external_id", sa.String(length=255), nullable=False),
        sa.Column("sender_role", sa.String(length=50), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("parts", json_type, nullable=False),
        sa.Column("reply_to", json_type, nullable=True),
        sa.Column("routing", json_type, nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_inbound_messages_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "endpoint_id"], ["owner_channel_endpoints.business_id", "owner_channel_endpoints.id"], name=op.f("fk_inbound_messages_business_id_endpoint_id_owner_channel_endpoints"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "conversation_id"], ["conversations.business_id", "conversations.id"], name=op.f("fk_inbound_messages_business_id_conversation_id_conversations"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inbound_messages")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_inbound_messages_business_id_id")),
        sa.UniqueConstraint("endpoint_id", "message_key", name=op.f("uq_inbound_messages_endpoint_id")),
    )
    op.create_index(op.f("ix_inbound_messages_conversation_id"), "inbound_messages", ["conversation_id"])
    op.create_table(
        "outbound_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("inbound_message_id", sa.Uuid(), nullable=True),
        sa.Column("parts", json_type, nullable=False),
        sa.Column("reply_to", json_type, nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("provider_message_id", sa.String(length=255), nullable=True),
        sa.Column("correlation_id", sa.String(length=255), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_detail", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_outbound_messages_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "conversation_id"], ["conversations.business_id", "conversations.id"], name=op.f("fk_outbound_messages_business_id_conversation_id_conversations"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "inbound_message_id"], ["inbound_messages.business_id", "inbound_messages.id"], name=op.f("fk_outbound_messages_business_id_inbound_message_id_inbound_messages"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outbound_messages")),
        sa.UniqueConstraint("business_id", "id", name=op.f("uq_outbound_messages_business_id_id")),
        sa.UniqueConstraint("inbound_message_id", "correlation_id", name=op.f("uq_outbound_messages_inbound_message_id")),
    )
    op.create_index(op.f("ix_outbound_messages_conversation_id"), "outbound_messages", ["conversation_id"])
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("inbound_message_id", sa.Uuid(), nullable=False),
        sa.Column("intent", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("leased_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_token", sa.Uuid(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_workflow_runs_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "inbound_message_id"], ["inbound_messages.business_id", "inbound_messages.id"], name=op.f("fk_workflow_runs_business_id_inbound_message_id_inbound_messages"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_workflow_runs")),
        sa.UniqueConstraint("inbound_message_id", name=op.f("uq_workflow_runs_inbound_message_id")),
    )
    op.create_index(op.f("ix_workflow_runs_business_id"), "workflow_runs", ["business_id"])
    op.create_table(
        "outbox_messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("outbound_message_id", sa.Uuid(), nullable=True),
        sa.Column("inbound_message_id", sa.Uuid(), nullable=True),
        sa.Column("command_type", sa.String(length=255), nullable=False),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("dedup_key", sa.String(length=255), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_outbox_messages_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "inbound_message_id"], ["inbound_messages.business_id", "inbound_messages.id"], name=op.f("fk_outbox_messages_business_id_inbound_message_id_inbound_messages"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "outbound_message_id"], ["outbound_messages.business_id", "outbound_messages.id"], name=op.f("fk_outbox_messages_business_id_outbound_message_id_outbound_messages"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outbox_messages")),
        sa.UniqueConstraint("business_id", "dedup_key", name=op.f("uq_outbox_messages_business_id_dedup_key")),
        sa.UniqueConstraint("inbound_message_id", name=op.f("uq_outbox_messages_inbound_message_id")),
        sa.UniqueConstraint("outbound_message_id", name=op.f("uq_outbox_messages_outbound_message_id")),
    )
    op.create_index(op.f("ix_outbox_messages_claim"), "outbox_messages", ["status", "available_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_outbox_messages_claim"), table_name="outbox_messages")
    op.drop_table("outbox_messages")
    op.drop_index(op.f("ix_workflow_runs_business_id"), table_name="workflow_runs")
    op.drop_table("workflow_runs")
    op.drop_index(op.f("ix_outbound_messages_conversation_id"), table_name="outbound_messages")
    op.drop_table("outbound_messages")
    op.drop_index(op.f("ix_inbound_messages_conversation_id"), table_name="inbound_messages")
    op.drop_table("inbound_messages")
    op.drop_index(op.f("ix_conversations_endpoint_id"), table_name="conversations")
    op.drop_table("conversations")
    op.drop_index(op.f("ix_owner_channel_endpoints_business_id"), table_name="owner_channel_endpoints")
    op.drop_table("owner_channel_endpoints")
    op.drop_table("businesses")
