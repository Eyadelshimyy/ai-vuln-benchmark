"""Per-person sign-in for the MCP gateway: clients, grants, codes, tokens.

Codes and tokens are stored as SHA-256 hashes, never the raw values.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-08
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_TABLES = (
    "tracebi_oauth_tokens",
    "tracebi_oauth_codes",
    "tracebi_oauth_pending",
    "tracebi_oauth_clients",
)


def upgrade() -> None:
    have = set(sa.inspect(op.get_bind()).get_table_names())
    if "tracebi_oauth_clients" not in have:
        op.create_table(
            "tracebi_oauth_clients",
            sa.Column("client_id", sa.String(255), primary_key=True),
            sa.Column("info", sa.Text(), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
        )
    if "tracebi_oauth_pending" not in have:
        op.create_table(
            "tracebi_oauth_pending",
            sa.Column("state_hash", sa.String(64), primary_key=True),
            sa.Column("client_id", sa.String(2048), nullable=False),
            sa.Column("params", sa.Text(), nullable=False),
            sa.Column("binder_hash", sa.String(64), nullable=False),
            sa.Column("nonce", sa.Text(), nullable=False),
            sa.Column("idp_verifier", sa.Text(), nullable=False),
            sa.Column("expires_at", sa.BigInteger(), nullable=False),
        )
    if "tracebi_oauth_codes" not in have:
        op.create_table(
            "tracebi_oauth_codes",
            sa.Column("code_hash", sa.String(64), primary_key=True),
            sa.Column("client_id", sa.String(2048), nullable=False),
            sa.Column("params", sa.Text(), nullable=False),
            sa.Column("identity", sa.Text(), nullable=False),
            sa.Column("family_id", sa.String(64), nullable=False),
            sa.Column("expires_at", sa.BigInteger(), nullable=False),
            sa.Column("used_at", sa.BigInteger(), nullable=True),
        )
    if "tracebi_oauth_tokens" not in have:
        op.create_table(
            "tracebi_oauth_tokens",
            sa.Column("token_hash", sa.String(64), primary_key=True),
            sa.Column("kind", sa.String(10), nullable=False),
            sa.Column("family_id", sa.String(64), nullable=False, index=True),
            sa.Column("client_id", sa.String(2048), nullable=False),
            sa.Column("identity", sa.Text(), nullable=False),
            sa.Column("scopes", sa.Text(), nullable=False),
            sa.Column("resource", sa.Text(), nullable=True),
            sa.Column("expires_at", sa.BigInteger(), nullable=False),
            sa.Column("session_expires_at", sa.BigInteger(), nullable=False),
            sa.Column("revoked_at", sa.BigInteger(), nullable=True),
            sa.Column("rotated_at", sa.BigInteger(), nullable=True),
        )


def downgrade() -> None:
    have = set(sa.inspect(op.get_bind()).get_table_names())
    for name in _TABLES:
        if name in have:
            op.drop_table(name)
