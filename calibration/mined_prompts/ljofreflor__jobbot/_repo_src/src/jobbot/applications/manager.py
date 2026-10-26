"""Application package preparation and tracking."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from jobbot.db.models import ApplicationEventRow, ApplicationRow
from jobbot.jobs.ids import next_application_id
from jobbot.models.application import Application, ApplicationEvent, ApplicationStatus
from jobbot.models.job import JobPosting

ANSWERS_TEMPLATE = """\
# Fill only what you know. Leave blank rather than inventing.
salary_expectation:
notice_period:
work_authorization:
english_level:
relocation:
remote_preference:
"""


class ApplicationRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_for_job(
        self,
        job_id: str,
        *,
        status: ApplicationStatus,
        package_dir: str | None = None,
    ) -> Application:
        existing = self._session.scalars(
            select(ApplicationRow).where(ApplicationRow.job_id == job_id)
        ).first()
        now = datetime.now(UTC)
        if existing is None:
            app_id = next_application_id(self._session)
            row = ApplicationRow(
                id=app_id,
                job_id=job_id,
                status=status.value,
                created_at=now,
                updated_at=now,
                package_dir=package_dir,
            )
            self._session.add(row)
            self._add_event(app_id, "created", status.value)
        else:
            existing.status = status.value
            existing.updated_at = now
            if package_dir:
                existing.package_dir = package_dir
            row = existing
            self._add_event(row.id, "status", status.value)
        self._session.commit()
        return Application(
            id=row.id,
            job_id=row.job_id,
            status=ApplicationStatus(row.status),
            created_at=row.created_at,
            updated_at=row.updated_at,
            package_dir=row.package_dir,
        )

    def get_for_job(self, job_id: str) -> Application | None:
        row = self._session.scalars(
            select(ApplicationRow).where(ApplicationRow.job_id == job_id)
        ).first()
        if row is None:
            return None
        return _application_from_row(row)

    def list_all(self) -> list[Application]:
        rows = self._session.scalars(select(ApplicationRow).order_by(ApplicationRow.id)).all()
        return [_application_from_row(r) for r in rows]

    def events_for(self, application_id: str) -> list[ApplicationEvent]:
        rows = self._session.scalars(
            select(ApplicationEventRow)
            .where(ApplicationEventRow.application_id == application_id)
            .order_by(ApplicationEventRow.id)
        ).all()
        return [_event_from_row(row) for row in rows]

    def add_event(
        self,
        application_id: str,
        event_type: str,
        detail: str | None = None,
    ) -> ApplicationEvent:
        event = self._add_event(application_id, event_type, detail)
        self._session.commit()
        return event

    def _add_event(
        self,
        application_id: str,
        event_type: str,
        detail: str | None,
    ) -> ApplicationEvent:
        row = ApplicationEventRow(
            application_id=application_id,
            event_type=event_type,
            detail=detail,
            created_at=datetime.now(UTC),
        )
        self._session.add(row)
        self._session.flush()
        return _event_from_row(row)


def _application_from_row(row: ApplicationRow) -> Application:
    return Application(
        id=row.id,
        job_id=row.job_id,
        status=ApplicationStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        package_dir=row.package_dir,
    )


def _event_from_row(row: ApplicationEventRow) -> ApplicationEvent:
    return ApplicationEvent(
        id=str(row.id),
        application_id=row.application_id,
        event_type=row.event_type,
        detail=row.detail,
        created_at=row.created_at,
    )


class FilesystemApplicationPackage:
    """Minimal ApplicationPackage over a prepared job application directory."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def cv_pdf(self) -> Path | None:
        pdf = self._directory / "cv.pdf"
        if pdf.is_file():
            return pdf
        parent_pdf = self._directory.parent / "cv.pdf"
        return parent_pdf if parent_pdf.is_file() else None


def prepare_application_package(
    job: JobPosting,
    output_dir: Path,
    *,
    job_dir: Path | None = None,
    candidate: object | None = None,
) -> Path:
    """Create output/jobs/<id>/application/ package. Does not invent answers."""
    base = job_dir or (output_dir / "jobs" / job.id)
    app_dir = base / "application"
    app_dir.mkdir(parents=True, exist_ok=True)

    (app_dir / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")

    for name in ("cv.pdf", "cv_ats.txt", "match.json", "selection.json"):
        src = base / name
        if src.is_file():
            shutil.copy2(src, app_dir / name)

    answers = app_dir / "answers.yaml"
    if not answers.is_file():
        answers.write_text(ANSWERS_TEMPLATE, encoding="utf-8")

    notes = app_dir / "notes.md"
    if not notes.is_file():
        notes.write_text(
            f"# Notes — {job.id} {job.company} / {job.title}\n\n",
            encoding="utf-8",
        )

    if candidate is not None and _is_getonboard(job):
        from jobbot.adapters.getonboard.draft import (
            draft_getonboard_fields,
            load_permanent_profile,
            render_getonboard_markdown,
        )
        from jobbot.models.candidate import Candidate

        if isinstance(candidate, Candidate):
            permanent = load_permanent_profile(output_dir)
            fields = draft_getonboard_fields(
                candidate, job, permanent=permanent
            )
            (app_dir / "getonboard_es.md").write_text(
                render_getonboard_markdown(fields, job=job),
                encoding="utf-8",
            )

    manifest = {
        "job_id": job.id,
        "prepared_at": datetime.now(UTC).isoformat(),
        "files": sorted(p.name for p in app_dir.iterdir() if p.is_file()),
    }
    (app_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return app_dir


def _is_getonboard(job: JobPosting) -> bool:
    kind = (job.ats_kind or "").casefold()
    if kind == "getonboard":
        return True
    return any(url and "getonbrd.com" in url.casefold() for url in (job.ats_url, job.url))


def load_answers(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}
