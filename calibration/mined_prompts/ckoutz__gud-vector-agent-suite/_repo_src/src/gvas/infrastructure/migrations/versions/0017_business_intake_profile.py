"""business intake profile; intake collected ``problem`` becomes ``details``

Adds the per-business booking-agent copy (brief, questions, opening) and
rewrites stored intake snapshots from the inspection-specific ``problem`` key
to the generic ``details`` key (plus an empty ``notes``). Downgrade folds
``notes`` back into ``problem`` so nothing collected is lost either way.

Revision ID: 0017_business_intake_profile
Revises: 0016_intake_booking_event_type
"""

from collections.abc import Callable

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0017_business_intake_profile"
down_revision = "0016_intake_booking_event_type"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
TEXT_MAX_CHARS = 2000
SEPARATOR = " — "

intake_conversations = sa.table(
    "intake_conversations",
    sa.column("id", sa.Uuid()),
    sa.column("collected", json_type),
)


def _problem_to_details(collected: dict[str, object]) -> dict[str, object]:
    values = dict(collected)
    problem = values.pop("problem", None)
    if values.get("details") is None:
        values["details"] = problem
    values.setdefault("notes", None)
    return values


def _details_to_problem(collected: dict[str, object]) -> dict[str, object]:
    values = dict(collected)
    details = values.pop("details", None)
    notes = values.pop("notes", None)
    if details and notes:
        # Both survive a combined value over the limit: notes keep at least
        # half of the room, details fill the rest.
        room = TEXT_MAX_CHARS - len(SEPARATOR)
        kept_notes = str(notes)[: max(room // 2, room - len(str(details)))]
        values["problem"] = f"{str(details)[: room - len(kept_notes)]}{SEPARATOR}{kept_notes}"
    else:
        values["problem"] = str(details or notes)[:TEXT_MAX_CHARS] if details or notes else None
    return values


def _rewrite_collected(transform: Callable[[dict[str, object]], dict[str, object]]) -> None:
    connection = op.get_bind()
    rows = connection.execute(
        sa.select(intake_conversations.c.id, intake_conversations.c.collected)
    ).all()
    for row_id, collected in rows:
        if not isinstance(collected, dict):
            continue
        rewritten = transform(collected)
        if rewritten != collected:
            connection.execute(
                intake_conversations.update()
                .where(intake_conversations.c.id == row_id)
                .values(collected=rewritten)
            )


def upgrade() -> None:
    op.add_column("businesses", sa.Column("intake_brief", sa.Text(), nullable=True))
    op.add_column("businesses", sa.Column("intake_questions", sa.Text(), nullable=True))
    op.add_column("businesses", sa.Column("intake_opening", sa.Text(), nullable=True))
    _rewrite_collected(_problem_to_details)


def downgrade() -> None:
    _rewrite_collected(_details_to_problem)
    op.drop_column("businesses", "intake_opening")
    op.drop_column("businesses", "intake_questions")
    op.drop_column("businesses", "intake_brief")
