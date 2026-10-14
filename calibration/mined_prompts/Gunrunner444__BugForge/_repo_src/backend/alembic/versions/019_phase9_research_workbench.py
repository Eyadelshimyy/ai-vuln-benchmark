"""Phase 9 research projects, identity secret references, and operator ownership.

Revision ID: 019
Revises: 018
Create Date: 2026-09-20 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "019"
down_revision: str | None = "018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "security_research_sessions",
        sa.Column("operator_identity", sa.String(length=128), nullable=False, server_default=""),
    )
    op.add_column(
        "security_research_sessions",
        sa.Column("research_project_id", sa.String(length=64), nullable=False, server_default=""),
    )
    with op.batch_alter_table("research_identities") as batch:
        batch.drop_column("cookies")
        batch.drop_column("storage")
        batch.add_column(
            sa.Column("credential_ref", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("browser_context_id", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("http_session_id", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("storage_namespace", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column(
                "authentication_state",
                sa.String(length=32),
                nullable=False,
                server_default="unauthenticated",
            )
        )
        batch.add_column(
            sa.Column(
                "credential_provenance", sa.String(length=64), nullable=False, server_default="none"
            )
        )
        batch.add_column(sa.Column("header_names", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("header_secret_refs", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("cookie_names", sa.JSON(), nullable=True))
        batch.add_column(
            sa.Column("cookie_secret_ref", sa.String(length=64), nullable=False, server_default="")
        )
        batch.add_column(sa.Column("storage_keys", sa.JSON(), nullable=True))
        batch.add_column(
            sa.Column("storage_secret_ref", sa.String(length=64), nullable=False, server_default="")
        )
    op.create_table(
        "security_research_projects",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("program_handle", sa.String(length=255), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=64), nullable=False),
        sa.Column("operator_identity", sa.String(length=128), nullable=False),
        sa.Column("extra", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"], ["security_research_sessions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_security_research_projects_session_id", "security_research_projects", ["session_id"]
    )


def downgrade() -> None:
    op.drop_index(
        "ix_security_research_projects_session_id", table_name="security_research_projects"
    )
    op.drop_table("security_research_projects")
    with op.batch_alter_table("research_identities") as batch:
        batch.drop_column("storage_secret_ref")
        batch.drop_column("storage_keys")
        batch.drop_column("cookie_secret_ref")
        batch.drop_column("cookie_names")
        batch.drop_column("header_secret_refs")
        batch.drop_column("header_names")
        batch.drop_column("credential_provenance")
        batch.drop_column("authentication_state")
        batch.drop_column("storage_namespace")
        batch.drop_column("http_session_id")
        batch.drop_column("browser_context_id")
        batch.drop_column("credential_ref")
        batch.add_column(sa.Column("cookies", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("storage", sa.JSON(), nullable=True))
    op.drop_column("security_research_sessions", "research_project_id")
    op.drop_column("security_research_sessions", "operator_identity")
