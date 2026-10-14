"""field note completeness state

Revision ID: 0003_field_note_completeness
Revises: 0002_field_note_cases
"""
# ruff: noqa: E501

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# fmt: off
revision = "0003_field_note_completeness"
down_revision = "0002_field_note_cases"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "field_note_checklists",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("checklist_key", sa.String(length=255), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("items", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_field_note_checklists_business_id_businesses"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_checklists")),
        sa.UniqueConstraint("business_id", "checklist_key", "version", name="uq_field_note_checklists_key_version"),
    )
    op.create_index("ix_field_note_checklists_business_id", "field_note_checklists", ["business_id"], unique=False)

    op.create_table(
        "field_note_reviews",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("active_conversation_id", sa.Uuid(), nullable=True),
        sa.Column("external_conversation_id", sa.String(length=255), nullable=False),
        sa.Column("inbound_message_id", sa.Uuid(), nullable=False),
        sa.Column("checklist_key", sa.String(length=255), nullable=False),
        sa.Column("checklist_version", sa.Integer(), nullable=False),
        sa.Column("transcript_text", sa.Text(), nullable=False),
        sa.Column("thread_correlation_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("round_index", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_field_note_reviews_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "conversation_id"], ["conversations.business_id", "conversations.id"], name=op.f("fk_field_note_reviews_business_id_conversations"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "inbound_message_id"], ["inbound_messages.business_id", "inbound_messages.id"], name=op.f("fk_field_note_reviews_business_id_inbound_messages"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "active_conversation_id"], ["conversations.business_id", "conversations.id"], name="fk_field_note_reviews_active_conversation", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "checklist_key", "checklist_version"], ["field_note_checklists.business_id", "field_note_checklists.checklist_key", "field_note_checklists.version"], name="fk_field_note_reviews_checklist", ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_reviews")),
        sa.UniqueConstraint("business_id", "id", name="uq_field_note_reviews_business_id_id"),
        sa.UniqueConstraint("business_id", "inbound_message_id", name="uq_field_note_reviews_business_id_inbound_message_id"),
        sa.UniqueConstraint("business_id", "active_conversation_id", name="uq_field_note_reviews_active_conversation"),
    )
    op.create_index("ix_field_note_reviews_conversation_id", "field_note_reviews", ["conversation_id"], unique=False)

    op.create_table(
        "field_note_follow_up_questions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("review_id", sa.Uuid(), nullable=False),
        sa.Column("item_key", sa.String(length=255), nullable=False),
        sa.Column("round_index", sa.Integer(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("correlation_id", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_field_note_follow_up_questions_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "review_id"], ["field_note_reviews.business_id", "field_note_reviews.id"], name="fk_field_note_questions_review", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_follow_up_questions")),
        sa.UniqueConstraint("business_id", "id", name="uq_field_note_follow_up_questions_business_id_id"),
        sa.UniqueConstraint("business_id", "review_id", "id", name="uq_field_note_follow_up_questions_business_review_id"),
        sa.UniqueConstraint("review_id", "round_index", "item_key", name="uq_field_note_follow_up_questions_round_item"),
        sa.UniqueConstraint("review_id", "correlation_id", name="uq_field_note_follow_up_questions_correlation"),
    )
    op.create_index("ix_field_note_follow_up_questions_review_id", "field_note_follow_up_questions", ["review_id"], unique=False)

    op.create_table(
        "field_note_review_answers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("business_id", sa.Uuid(), nullable=False),
        sa.Column("review_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.Uuid(), nullable=False),
        sa.Column("inbound_message_id", sa.Uuid(), nullable=False),
        sa.Column("item_key", sa.String(length=255), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["business_id"], ["businesses.id"], name=op.f("fk_field_note_review_answers_business_id_businesses"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "review_id"], ["field_note_reviews.business_id", "field_note_reviews.id"], name=op.f("fk_field_note_review_answers_business_id_field_note_reviews"), ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "review_id", "question_id"], ["field_note_follow_up_questions.business_id", "field_note_follow_up_questions.review_id", "field_note_follow_up_questions.id"], name="fk_field_note_review_answers_question", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["business_id", "inbound_message_id"], ["inbound_messages.business_id", "inbound_messages.id"], name=op.f("fk_field_note_review_answers_business_id_inbound_messages"), ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_field_note_review_answers")),
        sa.UniqueConstraint("question_id", name="uq_field_note_review_answers_question_id"),
        sa.UniqueConstraint("review_id", "inbound_message_id", name="uq_field_note_review_answers_review_inbound_message"),
    )
    op.create_index("ix_field_note_review_answers_review_id", "field_note_review_answers", ["review_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_field_note_review_answers_review_id", table_name="field_note_review_answers")
    op.drop_table("field_note_review_answers")
    op.drop_index("ix_field_note_follow_up_questions_review_id", table_name="field_note_follow_up_questions")
    op.drop_table("field_note_follow_up_questions")
    op.drop_index("ix_field_note_reviews_conversation_id", table_name="field_note_reviews")
    op.drop_table("field_note_reviews")
    op.drop_index("ix_field_note_checklists_business_id", table_name="field_note_checklists")
    op.drop_table("field_note_checklists")
# fmt: on
