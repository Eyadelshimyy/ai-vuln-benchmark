"""Add evidence-classification fields to reproduction tables (Phase 6 fix)

Adds target_behavior, expected_failure_pattern, observable_evidence to
bug_reproduction_sessions, and evidence_matched to bug_reproduction_attempts.

Revision ID: 007
Revises: 006
Create Date: 2024-01-07 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "bug_reproduction_sessions",
        sa.Column("target_behavior", sa.Text(), nullable=True),
    )
    op.add_column(
        "bug_reproduction_sessions",
        sa.Column("expected_failure_pattern", sa.Text(), nullable=True),
    )
    op.add_column(
        "bug_reproduction_sessions",
        sa.Column("observable_evidence", sa.Text(), nullable=True),
    )
    op.add_column(
        "bug_reproduction_attempts",
        sa.Column("evidence_matched", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("bug_reproduction_attempts", "evidence_matched")
    op.drop_column("bug_reproduction_sessions", "observable_evidence")
    op.drop_column("bug_reproduction_sessions", "expected_failure_pattern")
    op.drop_column("bug_reproduction_sessions", "target_behavior")
