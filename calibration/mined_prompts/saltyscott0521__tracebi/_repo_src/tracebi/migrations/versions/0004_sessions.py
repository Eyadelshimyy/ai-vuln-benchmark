"""Browser sign-in sessions for the web app.

Only the SHA-256 of the session id is stored, never the id itself.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-08
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    have = set(sa.inspect(op.get_bind()).get_table_names())
    if "tracebi_sessions" not in have:
        op.create_table(
            "tracebi_sessions",
            sa.Column("session_hash", sa.String(64), primary_key=True),
            sa.Column("sub", sa.Text(), nullable=False),
            sa.Column("actor", sa.Text(), nullable=False),
            sa.Column("role", sa.String(10), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("expires_at", sa.BigInteger(), nullable=False),
            sa.Column("absolute_expires_at", sa.BigInteger(), nullable=False),
            sa.Column("revoked_at", sa.BigInteger(), nullable=True),
        )


def downgrade() -> None:
    if "tracebi_sessions" in set(sa.inspect(op.get_bind()).get_table_names()):
        op.drop_table("tracebi_sessions")
