"""Unknown keys in profile.yaml must be reported, not silently dropped (#142)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE
from jobbot.models.candidate import Candidate
from jobbot.profile.market import apply_confirmed_skills
from jobbot.profile.validator import unknown_profile_keys, validate_candidate
from tests.fixtures.profile import sample_profile_dict

_UNKNOWN_MSG = "clave desconocida, se ignora"


def _health_profile_with_unknown_keys() -> dict[str, Any]:
    """Fictional public-health profile: extras the schema does not know yet."""
    return {
        "personal": {"name": "Ana Ejemplo", "headline": "Epidemióloga"},
        "languages": [{"name": "Inglés", "level": "B2"}],
        "experience": [
            {
                "id": "e1",
                "company": "Servicio Ficticio",
                "title": "Analista",
                "start_date": "2020-01",
                "current": True,
                "achivements": [{"id": "a1", "text": "Coordiné vigilancia de brotes."}],
            }
        ],
        "education": [
            {
                "id": "d1",
                "institution": "Universidad Ficticia",
                "degree": "MSc",
                "tags": ["salud_publica"],
            }
        ],
    }


def test_unknown_keys_are_listed_with_id_paths() -> None:
    issues = unknown_profile_keys(_health_profile_with_unknown_keys())
    paths = {i.path: i.message for i in issues}
    assert paths["languages"] == _UNKNOWN_MSG
    assert paths["experience.e1.achivements"] == _UNKNOWN_MSG
    assert paths["education.d1.tags"] == _UNKNOWN_MSG
    assert Candidate.model_validate(_health_profile_with_unknown_keys())


def test_a_schema_profile_has_no_unknown_keys() -> None:
    assert unknown_profile_keys(sample_profile_dict()) == []
    assert validate_candidate(Candidate.model_validate(sample_profile_dict())).ok


def test_skill_group_extras_are_allowed() -> None:
    data = sample_profile_dict()
    data["skills"]["epidemiology"] = ["Vigilancia de brotes"]
    assert unknown_profile_keys(data) == []


def test_apply_confirmed_skills_on_raw_keeps_unknown_keys() -> None:
    raw = _health_profile_with_unknown_keys()
    raw["skills"] = {"engineering": ["Vigilancia epidemiológica"]}
    out = apply_confirmed_skills(raw, ["Epidemiología de campo"])
    assert out["languages"][0]["name"] == "Inglés"
    assert out["experience"][0]["achivements"]
    assert out["education"][0]["tags"] == ["salud_publica"]
    assert "Epidemiología de campo" in out["skills"]["other"]
    assert "Vigilancia epidemiológica" in out["skills"]["engineering"]


def _workspace(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch, raw: dict[str, Any]
) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def _run_validate(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> tuple[int, str, str]:
    import jobbot.cli as cli

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, color_system=None))
    monkeypatch.setattr(cli, "err_console", Console(file=err, color_system=None))
    code = cli.run_cli(argv, standalone_mode=False)
    return code, out.getvalue(), err.getvalue()


def test_profile_validate_warns_and_stays_valid(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch, _health_profile_with_unknown_keys())
    code, out, err = _run_validate(monkeypatch, ["profile", "validate"])
    assert code == SUCCESS
    assert "PROFILE VALID" in out
    blob = out + err
    assert "languages: clave desconocida, se ignora" in blob
    assert "experience.e1.achivements: clave desconocida, se ignora" in blob
    assert "education.d1.tags: clave desconocida, se ignora" in blob


def test_profile_validate_strict_exits_validation(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch, _health_profile_with_unknown_keys())
    code, out, err = _run_validate(monkeypatch, ["profile", "validate", "--strict"])
    assert code == VALIDATION_FAILURE
    assert "PROFILE INVALID" in err or "PROFILE INVALID" in out
    assert "languages" in (out + err)


def test_profile_validate_without_extras_matches_the_valid_report(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = yaml.safe_load(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8")
    )
    _workspace(tmp_path, project_root, monkeypatch, raw)
    code, out, err = _run_validate(monkeypatch, ["profile", "validate"])
    assert code == SUCCESS
    assert err == ""
    assert "PROFILE VALID" in out
    assert "clave desconocida" not in out
    assert "Experiences:" in out
    assert "Achievements:" in out
    assert "Skills:" in out
    assert "Publications:" in out


def test_cv_build_summarises_ignored_keys(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch, _health_profile_with_unknown_keys())
    import jobbot.cli as cli

    monkeypatch.setattr(cli, "build_cv", lambda **_k: [tmp_path / "cv.pdf"])
    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, color_system=None))
    monkeypatch.setattr(cli, "err_console", Console(file=err, color_system=None))
    assert cli.run_cli(["cv", "build"], standalone_mode=False) == SUCCESS
    blob = out.getvalue() + err.getvalue()
    assert "3 claves desconocidas" in blob
    assert "profile validate" in blob


def test_suggest_promote_keeps_languages(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _health_profile_with_unknown_keys()
    raw["skills"] = {"engineering": ["Vigilancia epidemiológica"]}
    _workspace(tmp_path, project_root, monkeypatch, raw)

    from jobbot.config import load_config
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    config = load_config()
    session = make_session_factory(make_engine(config.database_path))()
    repo = JobRepository(session)
    for i in range(2):
        repo.upsert_external(
            JobPosting(
                id="PENDING",
                title="Analista",
                company=f"Servicio {i}",
                description=(
                    "Looking for Tableau experience and clear written reports. "
                    "Manejo de Tableau is required."
                ),
                skills=["Tableau"],
            )
        )

    monkeypatch.setattr("typer.confirm", lambda *_a, **_k: True)
    import jobbot.cli as cli

    out, err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, color_system=None))
    monkeypatch.setattr(cli, "err_console", Console(file=err, color_system=None))
    code = cli.run_cli(
        ["profile", "suggest-from-market", "--ask", "--promote", "--yes"],
        standalone_mode=False,
    )
    assert code == SUCCESS
    saved = yaml.safe_load((tmp_path / "data" / "profile.yaml").read_text(encoding="utf-8"))
    assert saved["languages"][0]["name"] == "Inglés"
    assert saved["experience"][0]["achivements"]
    assert saved["education"][0]["tags"] == ["salud_publica"]
    other = saved.get("skills", {}).get("other") or []
    assert any("tableau" in str(s).casefold() for s in other)
