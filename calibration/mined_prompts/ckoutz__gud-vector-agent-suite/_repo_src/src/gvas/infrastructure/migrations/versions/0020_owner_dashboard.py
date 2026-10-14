"""owner dashboard: owner e-mail, calendar feed, owner sign-in

Revision ID: 0020_owner_dashboard
Revises: 0019_business_notification_email
"""

import sqlalchemy as sa
from alembic import op

revision = "0020_owner_dashboard"
down_revision = "0019_business_notification_email"
branch_labels = None
depends_on = None


def _credential_table(name: str, expiry_column: str) -> None:
    op.create_table(
        name,
        sa.Column("token_hash", sa.String(length=64), primary_key=True),
        sa.Column(
            "business_id",
            sa.Uuid(),
            sa.ForeignKey("businesses.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(expiry_column, sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(f"ix_{name}_business_id", name, ["business_id"])


def upgrade() -> None:
    op.add_column("businesses", sa.Column("owner_email", sa.String(length=320), nullable=True))
    op.add_column("businesses", sa.Column("calendar_feed_url", sa.Text(), nullable=True))
    _credential_table("owner_login_tokens", "used_at")
    _credential_table("owner_sessions", "revoked_at")


def downgrade() -> None:
    op.drop_index("ix_owner_sessions_business_id", table_name="owner_sessions")
    op.drop_table("owner_sessions")
    op.drop_index("ix_owner_login_tokens_business_id", table_name="owner_login_tokens")
    op.drop_table("owner_login_tokens")
    op.drop_column("businesses", "calendar_feed_url")
    op.drop_column("businesses", "owner_email")
