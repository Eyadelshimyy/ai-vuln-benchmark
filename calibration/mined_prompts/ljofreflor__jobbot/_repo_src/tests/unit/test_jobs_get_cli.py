"""`jobbot get` Indeed hard links: store → package; `--apply` opens ATS (HITL)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS, VALIDATION_FAILURE


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


def _write_fixture(tmp_path: Path, extra: str = "") -> Path:
    html = f"""<!DOCTYPE html><html><body>
    <h1 data-testid="jobsearch-JobInfoHeader-title">Senior Data Scientist</h1>
    <div data-testid="inlineHeader-companyName">Sodimac</div>
    <div data-testid="jobsearch-JobInfoHeader-locationText">Santiago, Chile</div>
    <div id="jobDescriptionText"><p>Python and SQL.</p></div>
    {extra}
    </body></html>"""
    path = tmp_path / "viewjob.html"
    path.write_text(html, encoding="utf-8")
    return path


def _patch_browser(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )
    return opened


def test_get_stores_and_prepares_without_opening_browser(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    fixture = _write_fixture(tmp_path)
    opened = _patch_browser(monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        [
            "get",
            "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=email",
            "--fixture",
            str(fixture),
        ],
        standalone_mode=False,
    )
    out = capsys.readouterr().out

    assert code == SUCCESS
    assert "Stored" in out
    assert "Senior Data Scientist" in out or "Sodimac" in out
    assert "Prepared" in out
    assert (tmp_path / "output" / "jobs" / "J0001" / "job.json").is_file()
    assert opened == []


def test_get_stores_a_canonical_job(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    fixture = _write_fixture(tmp_path)
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    code = run_cli(
        [
            "get",
            "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97&from=email&tk=xyz",
            "--fixture",
            str(fixture),
        ],
        standalone_mode=False,
    )
    out = capsys.readouterr().out

    assert code == SUCCESS
    assert "J0001" in out

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    job = JobRepository(session).get("J0001")
    assert job is not None
    assert job.source == "indeed"
    assert job.source_job_id == "8a5fab1a7c476a97"
    assert job.url == "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97"
    assert job.ats_url == "https://cl.indeed.com/applystart?jk=8a5fab1a7c476a97"


def test_a_url_without_jk_is_rejected(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        ["get", "https://cl.indeed.com/viewjob?foo=bar"],
        standalone_mode=False,
    )

    assert code == GENERIC_FAILURE
    assert not (tmp_path / "output" / "jobs").exists()


def test_known_linkedin_without_fetcher_explains_the_gap(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        ["get", "https://www.linkedin.com/jobs/view/123"],
        standalone_mode=False,
    )

    assert code == GENERIC_FAILURE
    assert not (tmp_path / "output" / "jobs").exists()


def test_an_external_ats_is_what_gets_stored(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    fixture = _write_fixture(
        tmp_path,
        extra=(
            '<a href="https://boards.greenhouse.io/acme/jobs/4001">'
            "Apply on company site</a>"
        ),
    )
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    assert (
        run_cli(
            [
                "get",
                "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97",
                "--fixture",
                str(fixture),
            ],
            standalone_mode=False,
        )
        == SUCCESS
    )
    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    job = JobRepository(session).get("J0001")
    assert job is not None
    assert job.ats_url == "https://boards.greenhouse.io/acme/jobs/4001"
    assert "indeed.com" not in (job.ats_url or "")


def test_an_expired_posting_is_not_stored(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    fixture = _write_fixture(tmp_path, extra="<p>This job has expired on Indeed.</p>")
    from jobbot.cli import run_cli

    code = run_cli(
        [
            "get",
            "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97",
            "--fixture",
            str(fixture),
        ],
        standalone_mode=False,
    )
    captured = capsys.readouterr()
    text = captured.out + captured.err

    assert code == VALIDATION_FAILURE
    assert "expir" in text.casefold() or "filled" in text.casefold()
    assert not (tmp_path / "output" / "jobs").exists()


def test_application_apply_opens_indeed_apply_and_does_not_submit(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    fixture = _write_fixture(tmp_path)
    opened = _patch_browser(monkeypatch)
    from jobbot.cli import run_cli

    assert (
        run_cli(
            [
                "get",
                "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97",
                "--fixture",
                str(fixture),
            ],
            standalone_mode=False,
        )
        == SUCCESS
    )
    assert opened == []
    code = run_cli(
        ["application", "apply", "J0001", "--apply", "--yes"],
        standalone_mode=False,
    )

    assert code == SUCCESS
    assert opened == ["https://cl.indeed.com/applystart?jk=8a5fab1a7c476a97"]


def test_get_apply_opens_the_external_ats(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    fixture = _write_fixture(
        tmp_path,
        extra=(
            '<a href="https://boards.greenhouse.io/acme/jobs/4001">'
            "Apply on company site</a>"
        ),
    )
    opened = _patch_browser(monkeypatch)
    from jobbot.cli import run_cli

    code = run_cli(
        [
            "get",
            "https://cl.indeed.com/viewjob?jk=8a5fab1a7c476a97",
            "--fixture",
            str(fixture),
            "--apply",
            "--yes",
        ],
        standalone_mode=False,
    )

    assert code == SUCCESS
    assert opened == ["https://boards.greenhouse.io/acme/jobs/4001"]


def test_get_url_gap_records_ops_failure_for_autopoiesis(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Novel hard-link shapes must land in ops_failures with a real fingerprint.

    Autopoietic lane: exception → observability → ops failure work → issue → PR.
    A bare typer.Exit with empty message collapses every get gap into one useless
    fingerprint and breaks that loop.
    """
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.ops.failure_work import work_script
    from jobbot.ops.failures import list_failures

    code = run_cli(
        ["get", "https://www.linkedin.com/jobs/view/123"],
        standalone_mode=False,
    )
    err = capsys.readouterr().err

    assert code == GENERIC_FAILURE
    assert "Recorded failure" in err

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    try:
        rows = list_failures(session, status="new")
        assert len(rows) == 1
        rec = rows[0]
        assert rec.error_class == "UnsupportedPortalFetchError"
        assert rec.message
        assert "fetcher" in rec.message.casefold()
        script = work_script(rec)
        assert f"ops failure issue {rec.id}" in script
        assert "gh pr create" in script
        assert "jobbot get" in script
    finally:
        session.close()


def test_indeed_url_shape_failures_share_fingerprint_across_tracking(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracking params must not split one URL-shape gap into many fingerprints."""
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.ops.failures import list_failures

    urls = [
        "https://cl.indeed.com/viewjob?foo=bar&tk=aaa",
        "https://cl.indeed.com/viewjob?from=email&tk=bbb&xpse=1",
    ]
    for url in urls:
        assert run_cli(["get", url], standalone_mode=False) == GENERIC_FAILURE

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    try:
        rows = list_failures(session, status="new")
        assert len(rows) == 2
        assert rows[0].error_class == "UnsupportedPortalFetchError"
        assert rows[0].fingerprint == rows[1].fingerprint
        assert "jk" in rows[0].message.casefold()
        # Command keeps a stable host+path (no tracking noise) for the re-run lane.
        assert "tk=" not in rows[0].command
        assert "tk=" not in rows[1].command
    finally:
        session.close()
