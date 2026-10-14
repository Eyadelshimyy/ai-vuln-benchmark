"""One run shape: kind, target, finished, output path, verdict, detail.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

_COLUMNS = (
    "kind",
    "target",
    "finished",
    "output_path",
    "verdict",
    "detail",
)


def upgrade() -> None:
    have = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("tracebi_runs")}
    for name in _COLUMNS:
        if name not in have:
            op.add_column("tracebi_runs", sa.Column(name, sa.Text(), nullable=True))


def downgrade() -> None:
    have = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("tracebi_runs")}
    for name in reversed(_COLUMNS):
        if name in have:
            op.drop_column("tracebi_runs", name)
