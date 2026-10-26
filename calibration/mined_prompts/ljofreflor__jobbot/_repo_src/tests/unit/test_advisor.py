"""Advisor status, report, consent and deletion stay inside one sandbox."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from jobbot.advisor import (
    Consent,
    WorkspaceDeleteRefused,
    delete_workspace,
    status_lines,
    write_consent,
    write_report,
)
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.db.models import ApplicationRow, JobRow
from jobbot.workspace import owner_fingerprint, write_stamp


def _stamp(root: Path, name: str) -> None:
    write_stamp(
        root / "data" / ".jobbot-owner.json",
        label=name,
        fingerprint=owner_fingerprint(name),
    )


def _seed(root: Path, *, job_id: str, title: str, company: str, status: str, note: str) -> None:
    when = datetime(2026, 10, 1, tzinfo=UTC)
    engine = make_engine(root / "data" / "jobbot.sqlite")
    session = make_session_factory(engine)()
    session.add(
        JobRow(
            id=job_id,
            title=title,
            company=company,
            description="",
            note=note,
            discovered_at=when,
        )
    )
    session.add(
        ApplicationRow(
            id=f"A-{job_id}",
            job_id=job_id,
            status=status,
            created_at=when,
            updated_at=when,
            package_dir=None,
        )
    )
    session.commit()
    session.close()


def test_status_lists_each_sandbox_without_profile_pii(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOBBOT_SANDBOXES", str(tmp_path))
    ana = tmp_path / "ana"
    bea = tmp_path / "bea"
    (ana / "data").mkdir(parents=True)
    (bea / "data").mkdir(parents=True)
    _stamp(ana, "Ana Ejemplo")
    _stamp(bea, "Bea Ejemplo")
    (ana / "data" / "profile.yaml").write_text(
        "personal:\n  name: Ana Ejemplo\n  email: ana@example.com\n  phone: '+56 9 1111 2222'\n",
        encoding="utf-8",
    )
    _seed(ana, job_id="J0001", title="DS", company="Acme", status="prepared", note="")
    (ana / "output" / "cv").mkdir(parents=True)
    (ana / "output" / "cv" / "improvement_proposal.yaml").write_text("ok\n", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()

    text = "\n".join(
        status_lines(
            [("ana", ana), ("bea", bea), ("empty", empty)],
            today=datetime(2026, 10, 2, tzinfo=UTC),
        )
    )

    assert "owner=" + owner_fingerprint("Ana Ejemplo") in text
    assert "jobs=1" in text
    assert "prepared=1" in text
    assert "proposal=yes" in text
    assert "jobs=0" in text
    assert "unstamped" in text
    assert "Ana Ejemplo" not in text
    assert "ana@example.com" not in text
    assert "1111" not in text


def test_report_stays_inside_one_sandbox_and_redacts(tmp_path: Path) -> None:
    ana = tmp_path / "ana"
    bea = tmp_path / "bea"
    _seed(
        ana,
        job_id="J0001",
        title="DS",
        company="Acme",
        status="prepared",
        note="write ana.real@example.com",
    )
    _seed(bea, job_id="J0009", title="Nurse", company="Clinica Sur", status="prepared", note="")
    since = datetime(2026, 9, 1, tzinfo=UTC)
    path = write_report(ana, since=since, now=datetime(2026, 10, 2, tzinfo=UTC))
    text = path.read_text(encoding="utf-8")
    assert "Acme" in text
    assert "Clinica Sur" not in text
    assert "J0009" not in text
    assert "ana.real@example.com" not in text
    assert "[email]" in text


def test_empty_period_is_an_explicit_report(tmp_path: Path) -> None:
    root = tmp_path / "ana"
    path = write_report(
        root,
        since=datetime(2026, 10, 2, tzinfo=UTC),
        now=datetime(2026, 10, 2, tzinfo=UTC),
    )
    assert "No activity in this period." in path.read_text(encoding="utf-8")


def test_delete_requires_confirmation_and_removes_only_that_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JOBBOT_SANDBOXES", str(tmp_path))
    ana = tmp_path / "ana"
    bea = tmp_path / "bea"
    ana.mkdir()
    bea.mkdir()
    (ana / "data").mkdir()
    (bea / "keep.txt").write_text("stay", encoding="utf-8")
    with pytest.raises(WorkspaceDeleteRefused):
        delete_workspace("ana", confirmed=False)
    assert ana.is_dir()
    delete_workspace("ana", confirmed=True)
    assert not ana.exists()
    assert (bea / "keep.txt").is_file()


def test_retention_due_shows_on_the_status_line(tmp_path: Path) -> None:
    root = tmp_path / "ana"
    (root / "data").mkdir(parents=True)
    write_consent(
        root,
        Consent(consented_at="2026-01-01", scope="prepare", delete_after="2026-02-01"),
    )
    line = status_lines([("ana", root)], today=datetime(2026, 10, 2, tzinfo=UTC))[0]
    assert "retention=due" in line
