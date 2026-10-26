"""Backfill word_count for documents imported before the counter existed.

Revision ID: 003
Revises: 002
Create Date: 2025-01-21 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE documents
        SET word_count = COALESCE(
            array_length(regexp_split_to_array(btrim(content), E'\\\\s+'), 1), 0
        )
        WHERE word_count = 0 AND btrim(content) <> ''
        """
    )


def downgrade() -> None:
    pass
