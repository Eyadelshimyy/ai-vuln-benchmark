"""jobs add --stdin tells you it is waiting; Ctrl-C is not an ops failure."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import typer

from jobbot.exit_codes import SUCCESS, USER_CANCEL, VALIDATION_FAILURE


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_stdin_tty_says_it_is_waiting(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        typer,
        "get_text_stream",
        lambda _name: io.StringIO("Title: Enfermera\nCompany: Clínica\n\nTurnos de noche.\n"),
    )
    code = run_cli(["jobs", "add", "--stdin"], standalone_mode=False)
    assert code == SUCCESS
    assert "Ctrl-D" in capsys.readouterr().err


def test_filled_text_is_not_stored(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    body = "Title: Un cargo\nCompany: Empresa\n\nEste cargo ya fue cubierto.\n"
    path = tmp_path / "closed.txt"
    path.write_text(body, encoding="utf-8")
    code = run_cli(["jobs", "add", "--file", str(path)], standalone_mode=False)
    assert code == VALIDATION_FAILURE
    err = capsys.readouterr().err
    assert "Not stored" in err
    assert "Recorded failure" not in err
    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    assert JobRepository(session).list_all() == []


def test_filled_page_behind_url_is_not_stored(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    monkeypatch.setattr(
        "jobbot.jobs.closure.fetch_posting_text",
        lambda _url, **_k: "<p>This position has been filled.</p>",
    )
    path = tmp_path / "open.txt"
    path.write_text(
        "Title: Un cargo\nCompany: Empresa\n\nDescripción del trabajo.\n",
        encoding="utf-8",
    )
    code = run_cli(
        ["jobs", "add", "--file", str(path), "--url", "https://careers.example.com/job/abc/titulo"],
        standalone_mode=False,
    )
    assert code == VALIDATION_FAILURE
    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    assert JobRepository(session).list_all() == []


def test_ctrl_c_prints_cancelled_and_records_nothing(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.ops.failures import list_failures

    def _cancel(*_args: object, **_kwargs: object) -> None:
        raise typer.Exit(USER_CANCEL)

    monkeypatch.setattr("jobbot.cli.app", _cancel)
    code = run_cli(["jobs", "add", "--stdin"], standalone_mode=False)
    assert code == USER_CANCEL
    captured = capsys.readouterr()
    assert "Cancelled." in captured.err
    assert "Recorded failure" not in captured.err
    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    assert list_failures(session, status=None) == []
