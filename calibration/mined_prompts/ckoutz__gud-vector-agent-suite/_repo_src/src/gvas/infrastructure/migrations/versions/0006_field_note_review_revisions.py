"""field note review revisions

Revision ID: 0006_field_note_review_revisions
Revises: 0005_field_note_case_closure
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_field_note_review_revisions"
down_revision = "0005_field_note_case_closure"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "field_note_reviews",
        sa.Column("transcript_fingerprint", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "field_note_reviews",
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.execute(
        "UPDATE field_note_reviews "
        "SET transcript_fingerprint = encode(sha256(transcript_text::bytea), 'hex')"
    )
    op.alter_column("field_note_reviews", "transcript_fingerprint", nullable=False)
    op.alter_column("field_note_reviews", "revision", server_default=None)
    op.drop_constraint(
        "uq_field_note_reviews_business_id_inbound_message_id",
        "field_note_reviews",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_field_note_reviews_business_id_inbound_message_transcript",
        "field_note_reviews",
        ["business_id", "inbound_message_id", "transcript_fingerprint"],
    )
    op.create_unique_constraint(
        "uq_field_note_reviews_business_id_inbound_message_revision",
        "field_note_reviews",
        ["business_id", "inbound_message_id", "revision"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_field_note_reviews_business_id_inbound_message_revision",
        "field_note_reviews",
        type_="unique",
    )
    op.drop_constraint(
        "uq_field_note_reviews_business_id_inbound_message_transcript",
        "field_note_reviews",
        type_="unique",
    )
    op.execute(
        "DELETE FROM field_note_reviews WHERE revision > 1",
    )
    op.create_unique_constraint(
        "uq_field_note_reviews_business_id_inbound_message_id",
        "field_note_reviews",
        ["business_id", "inbound_message_id"],
    )
    op.drop_column("field_note_reviews", "revision")
    op.drop_column("field_note_reviews", "transcript_fingerprint")
