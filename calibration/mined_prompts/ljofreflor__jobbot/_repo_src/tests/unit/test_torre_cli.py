"""`jobbot torre search` stores jobs and feeds the company registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jobbot.exit_codes import SUCCESS


@pytest.fixture
def workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _install_fixture_fetcher(
    monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> list[dict[str, Any]]:
    """Serve the recorded payload; the CLI must never reach the network."""
    import jobbot.adapters.torre.jobs as torre

    payload = json.loads(
        (project_root / "tests" / "fixtures" / "torre_search.json").read_text(encoding="utf-8")
    )
    bodies: list[dict[str, Any]] = []

    class FixtureFetcher:
        def post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
            bodies.append(body)
            return payload

    monkeypatch.setattr(torre, "UrllibFetcher", FixtureFetcher)
    return bodies


def test_search_stores_jobs_and_learns_the_portal(
    workspace: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from jobbot.cli import run_cli

    bodies = _install_fixture_fetcher(monkeypatch, project_root)

    args = ["torre", "search", "analista de datos", "--limit", "3"]
    assert run_cli(args, standalone_mode=False) == SUCCESS

    out = capsys.readouterr().out
    assert "Torre results" in out
    assert "Estudio Neutro" in out
    assert "Stored 5 jobs" in out
    assert bodies[0]["skill/role"]["text"] == "analista de datos"
    assert (workspace / "data" / "portals.yaml").is_file()


def test_remote_flag_does_not_store_office_roles(
    workspace: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--remote`` must drop physical_location / on_site even if the API leaks them."""
    from jobbot.cli import run_cli
    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository

    bodies = _install_fixture_fetcher(monkeypatch, project_root)

    assert (
        run_cli(
            ["torre", "search", "cientifico", "--remote", "--limit", "20"],
            standalone_mode=False,
        )
        == SUCCESS
    )
    out = capsys.readouterr().out
    assert bodies[0]["remote"] == {"term": True}
    assert "Oficina Asia" not in out
    assert "Singapore" not in out
    assert "Clínica Neutra" not in out

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    try:
        jobs = JobRepository(session).list_all()
    finally:
        session.close()
    assert jobs
    assert all(j.remote_type not in {"physical_location", "on_site"} for j in jobs)


def test_an_opportunity_on_an_external_ats_feeds_the_company_registry(
    workspace: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from jobbot.cli import run_cli

    _install_fixture_fetcher(monkeypatch, project_root)

    args = ["torre", "search", "enfermera", "--limit", "3"]
    assert run_cli(args, standalone_mode=False) == SUCCESS

    registry_path = workspace / "data" / "companies.yaml"
    assert registry_path.is_file(), "the Greenhouse opportunity taught us nothing"
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    hosts = [
        site.get("domain")
        for company in raw.get("companies", [])
        for site in company.get("career_sites", [])
    ]
    assert any(host and "greenhouse.io" in host for host in hosts)
    assert not any(host and "torre.ai" in host for host in hosts), (
        "Torre is a board: it must not be registered as one company's career site"
    )


def test_limit_out_of_range_is_refused(
    workspace: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jobbot.cli import run_cli

    _install_fixture_fetcher(monkeypatch, project_root)

    args = ["torre", "search", "analista", "--limit", "99"]
    assert run_cli(args, standalone_mode=False) != SUCCESS
