"""merge Round 2 migration heads

Revision ID: 0003_merge_round2_heads
Revises: 0002_field_note_cases, 0002_field_note_reports, 0002_quote_workflow_core
"""

revision = "0003_merge_round2_heads"
down_revision = (
    "0002_field_note_cases",
    "0002_field_note_reports",
    "0002_quote_workflow_core",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
