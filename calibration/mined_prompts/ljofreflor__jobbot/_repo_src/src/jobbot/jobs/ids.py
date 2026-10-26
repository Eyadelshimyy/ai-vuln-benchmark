"""Allocate readable internal IDs: J0001, A0001, …"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from jobbot.db.models import ApplicationRow, JobRow, OpsFailureRow, OpsSymptomRow


def next_job_id(session: Session) -> str:
    rows = session.scalars(select(JobRow.id)).all()
    return _next_from_values(list(rows), prefix="J", width=4)


def next_application_id(session: Session) -> str:
    rows = session.scalars(select(ApplicationRow.id)).all()
    return _next_from_values(list(rows), prefix="A", width=4)


def next_failure_id(session: Session) -> str:
    rows = session.scalars(select(OpsFailureRow.id)).all()
    return _next_from_values(list(rows), prefix="F", width=4)


def next_symptom_id(session: Session) -> str:
    rows = session.scalars(select(OpsSymptomRow.id)).all()
    return _next_from_values(list(rows), prefix="S", width=4)


def _next_from_values(values: list[str], *, prefix: str, width: int) -> str:
    max_n = 0
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for value in values:
        match = pattern.match(value)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return f"{prefix}{max_n + 1:0{width}d}"
