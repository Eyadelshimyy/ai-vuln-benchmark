"""`profile queries` and the searches that run without typing a query."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from jobbot.exit_codes import SUCCESS
from jobbot.jobs.sources import JobSearchQuery
from jobbot.models.job import JobPosting
from tests.fixtures.profile import public_health_profile_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jobs"


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: dict[str, Any]) -> Path:
    (tmp_path / "data").mkdir(exist_ok=True)
    profile = tmp_path / "data" / "profile.yaml"
    profile.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return profile


def _run(args: list[str]) -> int:
    from jobbot.cli import run_cli

    return run_cli(args, standalone_mode=False)


def test_queries_dry_run_prints_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = _workspace(tmp_path, monkeypatch, public_health_profile_dict())
    before = profile.read_text(encoding="utf-8")

    assert _run(["profile", "queries"]) == SUCCESS

    out = capsys.readouterr().out
    assert "Vigilancia epidemiológica" in out
    assert "--apply" in out
    assert profile.read_text(encoding="utf-8") == before


def test_queries_apply_saves_an_editable_list_with_a_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _workspace(tmp_path, monkeypatch, public_health_profile_dict())

    assert _run(["profile", "queries", "--apply", "--yes", "--limit", "4"]) == SUCCESS

    saved = yaml.safe_load(profile.read_text(encoding="utf-8"))["search_queries"]
    assert len(saved) == 4
    assert "Vigilancia epidemiológica" in saved
    assert profile.with_suffix(".yaml.bak").is_file()


def test_queries_apply_can_be_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _workspace(tmp_path, monkeypatch, public_health_profile_dict())
    monkeypatch.setattr("typer.confirm", lambda *_a, **_k: False)

    assert _run(["profile", "queries", "--apply"]) == SUCCESS

    assert "search_queries" not in yaml.safe_load(profile.read_text(encoding="utf-8"))


def test_saved_queries_are_shown_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = public_health_profile_dict()
    raw["search_queries"] = ["epidemiólogo remoto"]
    _workspace(tmp_path, monkeypatch, raw)

    assert _run(["profile", "queries"]) == SUCCESS

    out = capsys.readouterr().out
    assert "epidemiólogo remoto" in out
    assert out.index("epidemiólogo remoto") < out.index("Vigilancia epidemiológica")


def _record_getonboard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from jobbot.adapters.getonboard.jobs import GetOnBoardJobSource

    asked: list[str] = []

    def fake_search(self: object, query: JobSearchQuery) -> list[JobPosting]:
        asked.append(query.query)
        return []

    monkeypatch.setattr(GetOnBoardJobSource, "search_jobs", fake_search)
    return asked


def test_getonboard_without_query_runs_the_saved_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = public_health_profile_dict()
    raw["search_queries"] = ["vigilancia epidemiológica", "salud pública", "zoonosis"]
    _workspace(tmp_path, monkeypatch, raw)
    asked = _record_getonboard(monkeypatch)

    assert _run(["getonboard", "search"]) == SUCCESS

    assert asked == ["vigilancia epidemiológica", "salud pública", "zoonosis"]
    assert "data scientist" not in capsys.readouterr().out.casefold()


def test_getonboard_without_saved_queries_derives_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _workspace(tmp_path, monkeypatch, public_health_profile_dict())
    asked = _record_getonboard(monkeypatch)

    assert _run(["getonboard", "search", "--max-queries", "2"]) == SUCCESS

    assert len(asked) == 2
    assert all("doctora" not in query.casefold() for query in asked)
    assert "jobbot profile queries --apply" in capsys.readouterr().out


def test_an_explicit_query_is_the_only_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = public_health_profile_dict()
    raw["search_queries"] = ["vigilancia epidemiológica", "salud pública"]
    _workspace(tmp_path, monkeypatch, raw)
    asked = _record_getonboard(monkeypatch)

    assert _run(["getonboard", "search", "epidemiología"]) == SUCCESS

    assert asked == ["epidemiología"]


def test_getonboard_merges_results_across_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from jobbot.adapters.getonboard.jobs import GetOnBoardJobSource

    raw = public_health_profile_dict()
    raw["search_queries"] = ["vigilancia epidemiológica", "salud pública"]
    _workspace(tmp_path, monkeypatch, raw)
    shared = JobPosting(
        id="gob-1",
        source="getonboard",
        source_job_id="epi-1",
        url="https://www.getonbrd.com/jobs/epi-1",
        title="Epidemiólogo/a",
        company="Servicio Neutro",
        description="Vigilancia epidemiológica e investigación de brotes.",
    )

    def fake_search(self: object, query: JobSearchQuery) -> list[JobPosting]:
        return [shared]

    monkeypatch.setattr(GetOnBoardJobSource, "search_jobs", fake_search)

    assert _run(["getonboard", "search"]) == SUCCESS

    assert "Stored 1 jobs" in capsys.readouterr().out


def test_torre_without_query_runs_the_saved_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jobbot.adapters.torre.jobs as torre

    raw = public_health_profile_dict()
    raw["search_queries"] = ["vigilancia epidemiológica", "zoonosis"]
    _workspace(tmp_path, monkeypatch, raw)
    bodies: list[dict[str, Any]] = []

    class FixtureFetcher:
        def post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
            bodies.append(body)
            return {"results": []}

    monkeypatch.setattr(torre, "UrllibFetcher", FixtureFetcher)

    assert _run(["torre", "search"]) == SUCCESS

    assert [b["skill/role"]["text"] for b in bodies] == ["vigilancia epidemiológica", "zoonosis"]


def test_sweep_without_query_runs_the_saved_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_root: Path
) -> None:
    from jobbot.adapters.linkedin.posts_source import LinkedInPostJobSource

    raw = public_health_profile_dict()
    raw["search_queries"] = ["vigilancia epidemiológica", "zoonosis"]
    _workspace(tmp_path, monkeypatch, raw)
    asked: list[str] = []

    def fake_fixture(self: object, path: Path, *, query: str, **_: object) -> list[JobPosting]:
        asked.append(query)
        return []

    monkeypatch.setattr(LinkedInPostJobSource, "search_from_fixture", fake_fixture)

    fixture = project_root / "tests" / "fixtures" / "linkedin_posts.txt"
    assert _run(["linkedin", "sweep", "--fixture", str(fixture)]) == SUCCESS

    assert asked == ["hiring vigilancia epidemiológica", "hiring zoonosis"]


def test_shortlist_warns_when_every_job_scores_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _workspace(tmp_path, monkeypatch, public_health_profile_dict())
    sales = FIXTURES / "field_sales_es.txt"
    assert _run(["jobs", "add", "--file", str(sales)]) == SUCCESS
    assert _run(["jobs", "add", "--file", str(sales)]) == SUCCESS
    capsys.readouterr()

    assert _run(["jobs", "shortlist"]) == SUCCESS

    out = capsys.readouterr().out
    assert "Matcher alert" in out


def test_shortlist_is_quiet_when_something_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _workspace(tmp_path, monkeypatch, public_health_profile_dict())
    assert _run(["jobs", "add", "--file", str(FIXTURES / "field_sales_es.txt")]) == SUCCESS
    epi = FIXTURES / "public_health_epidemiology_es.txt"
    assert _run(["jobs", "add", "--file", str(epi)]) == SUCCESS
    capsys.readouterr()

    assert _run(["jobs", "shortlist"]) == SUCCESS

    assert "Matcher alert" not in capsys.readouterr().out
