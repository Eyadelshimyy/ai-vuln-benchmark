"""per-business offer line and visit question for the booking agent

Revision ID: 0029_intake_offer_visit
Revises: 0028_demo_sandboxes
"""

import sqlalchemy as sa
from alembic import op

revision = "0029_intake_offer_visit"
down_revision = "0028_demo_sandboxes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("businesses", sa.Column("intake_offer_line", sa.Text(), nullable=True))
    op.add_column("businesses", sa.Column("intake_visit_question", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("businesses", "intake_visit_question")
    op.drop_column("businesses", "intake_offer_line")
