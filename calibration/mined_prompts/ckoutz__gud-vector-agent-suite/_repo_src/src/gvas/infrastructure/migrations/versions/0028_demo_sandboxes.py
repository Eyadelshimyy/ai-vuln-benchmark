"""demo sandboxes: a visitor's own copy of the demo business

Revision ID: 0028_demo_sandboxes
Revises: 0027_manual_plans
"""

import sqlalchemy as sa
from alembic import op

revision = "0028_demo_sandboxes"
down_revision = "0027_manual_plans"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "demo_sandboxes",
        sa.Column(
            "business_id",
            sa.Uuid(),
            sa.ForeignKey("businesses.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_demo_sandboxes_last_active_at", "demo_sandboxes", ["last_active_at"])


def downgrade() -> None:
    op.drop_index("ix_demo_sandboxes_last_active_at", table_name="demo_sandboxes")
    op.drop_table("demo_sandboxes")
