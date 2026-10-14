"""Persist hash-chained security-testing audit events.

Revision ID: 014
Revises: 013
Create Date: 2026-09-20 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "security_audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.String(length=128), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("scope_decision", sa.Text(), nullable=False),
        sa.Column("tool", sa.String(length=100), nullable=False),
        sa.Column("action", sa.String(length=200), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("rate_limit_decision", sa.Text(), nullable=False, server_default=""),
        sa.Column("result", sa.String(length=100), nullable=False, server_default=""),
        sa.Column("evidence_id", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("human_approval", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("entry_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("entry_hash"),
    )
    op.create_index("ix_security_audit_events_project_id", "security_audit_events", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_security_audit_events_project_id", table_name="security_audit_events")
    op.drop_table("security_audit_events")
