"""explicit field note case closure timestamp

Revision ID: 0005_field_note_case_closure
Revises: 0004_merge_completeness_head
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_field_note_case_closure"
down_revision = "0004_merge_completeness_head"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "field_note_cases",
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("field_note_cases", "closed_at")
