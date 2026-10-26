"""`jobbot get URL` — portal knowledge → JD → CV package (HITL apply)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS, VALIDATION_FAILURE

FIXTURE = Path("tests/fixtures/jobs/getonboard_applied_scientist.html")
URL = (
    "https://www.getonbrd.com/empleos/data-science-analytics/"
    "applied-scientist-neuralworks-santiago-e3c8"
)


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_unknown_domain_never_fetches(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    fetched: list[str] = []

    def _boom(url: str, **_k: object) -> str:
        fetched.append(url)
        raise AssertionError("must not fetch unknown portals")

    monkeypatch.setattr(
        "jobbot.adapters.getonboard.jobs.fetch_job_html",
        _boom,
    )

    code = run_cli(
        ["get", "https://careers.totally-unknown-corp.example/jobs/1"],
        standalone_mode=False,
    )
    assert code == VALIDATION_FAILURE
    assert fetched == []
    captured = capsys.readouterr()
    assert "recibiendo información del mundo" in captured.out + captured.err
    assert "Unknown employment domain" in captured.out + captured.err


def test_get_fixture_stores_job_and_prepares_package(
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
    from jobbot.portals.registry import default_portals_path, load_registry

    fixture = project_root / FIXTURE
    assert run_cli(
        ["get", URL, "--fixture", str(fixture)],
        standalone_mode=False,
    ) == SUCCESS

    out = capsys.readouterr().out
    assert "getonbrd.com" in out
    assert "Stored" in out
    assert "Prepared" in out
    assert "knowledge=" in out

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    jobs = JobRepository(session).list_all()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.company == "NeuralWorks"
    assert job.title == "Applied Scientist"
    assert job.match_score is not None
    assert (config.output_dir / "jobs" / job.id / "application").is_dir()
    assert (
        (config.output_dir / "jobs" / job.id / "cv_ats.txt").is_file()
        or (config.output_dir / "jobs" / job.id / "cv.pdf").is_file()
    )

    portals = load_registry(default_portals_path(tmp_path))
    assert portals.find("getonbrd.com") is not None


def test_known_but_unsupported_ats_explains_the_gap(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        ["get", "https://boards.greenhouse.io/acme/jobs/12345"],
        standalone_mode=False,
    )
    assert code == GENERIC_FAILURE
    captured = capsys.readouterr()
    err = captured.out + captured.err
    assert "No hard-link fetcher" in err or "greenhouse" in err.casefold()


def test_help_does_not_promise_greenhouse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jobbot.cli import run_cli
    from tests.conftest import plain_cli_text

    code = run_cli(["get", "--help"], standalone_mode=False)
    assert code == SUCCESS
    text = plain_cli_text(capsys.readouterr().out)
    assert "Get on Board" in text
    assert "Indeed" in text
    assert "Greenhouse" not in text
    assert "--fixture" in text


def test_career_fixture_on_unknown_host_stores_job(
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

    fixture = project_root / "tests/fixtures/jobs/career_phenom_open.html"
    url = (
        "https://careers.example-corp.test/global/en/job/abc123/"
        "Applied-Research-Analyst"
    )
    assert run_cli(
        ["get", url, "--fixture", str(fixture)],
        standalone_mode=False,
    ) == SUCCESS

    out = capsys.readouterr().out
    assert "Stored" in out
    assert "Prepared" in out

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    jobs = JobRepository(session).list_all()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.company == "Northwind Labs"
    assert job.title == "Applied Research Analyst"
    assert "SQL" in job.description or "spreadsheets" in job.description.casefold()
    assert (config.output_dir / "jobs" / job.id / "application").is_dir()


def test_filled_career_fixture_is_not_stored(
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

    fixture = project_root / "tests/fixtures/jobs/career_phenom_filled.html"
    url = "https://careers.example-corp.test/global/en/job/abc123/role"
    code = run_cli(
        ["get", url, "--fixture", str(fixture)],
        standalone_mode=False,
    )
    assert code == VALIDATION_FAILURE
    captured = capsys.readouterr()
    assert "filled" in (captured.out + captured.err).casefold()
    assert "Recorded failure" not in captured.out + captured.err

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    assert JobRepository(session).list_all() == []


def test_live_getonboard_download_shows_progress(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    html = (project_root / FIXTURE).read_text(encoding="utf-8")
    seen: dict[str, object] = {}

    def fake(url: str, **kwargs: object) -> str:
        seen["url"] = url
        seen["on_chunk"] = kwargs.get("on_chunk")
        on_chunk = kwargs.get("on_chunk")
        if callable(on_chunk):
            on_chunk(4, 4)
        return html

    monkeypatch.setattr("jobbot.adapters.getonboard.jobs.fetch_job_html", fake)
    assert run_cli(["get", URL], standalone_mode=False) == SUCCESS
    assert callable(seen["on_chunk"])
    assert "descargando la oferta" in capsys.readouterr().out
