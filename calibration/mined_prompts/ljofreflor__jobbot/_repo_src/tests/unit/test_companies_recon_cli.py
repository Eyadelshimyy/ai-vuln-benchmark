"""CLI `companies recon` dry-run vs --apply (issue #45)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE
from jobbot.portals.detect import AtsKind
from jobbot.portals.form_learn import default_form_knowledge_path, load_form_knowledge

FIXTURE = Path("tests/fixtures/forms/career_inside_greenhouse.html")


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    payload = {
        "version": 1,
        "companies": [
            {
                "id": "acme",
                "name": "Acme",
                "career_sites": [
                    {
                        "url": "https://careers.acme-demo.cl/join",
                        "domain": "careers.acme-demo.cl",
                        "site_type": "company_career_portal",
                        "ats": "unknown",
                        "status": "active",
                    }
                ],
            }
        ],
    }
    (tmp_path / "data" / "companies.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_recon_dry_run_with_fixture_does_not_write(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.companies.registry import default_companies_path, load_companies

    before = load_companies(default_companies_path(tmp_path)).find_company("acme")
    assert before is not None and before.career_sites[0].ats == AtsKind.UNKNOWN

    assert (
        run_cli(
            ["companies", "recon", "Acme", "--fixture", str(project_root / FIXTURE)],
            standalone_mode=False,
        )
        == SUCCESS
    )
    out = capsys.readouterr().out
    assert "greenhouse" in out.casefold()
    assert "Dry-run" in out
    after = load_companies(default_companies_path(tmp_path)).find_company("acme")
    assert after is not None and after.career_sites[0].ats == AtsKind.UNKNOWN
    assert not default_form_knowledge_path(tmp_path).is_file()


def test_recon_apply_stores_ats_and_form(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.companies.registry import default_companies_path, load_companies

    assert (
        run_cli(
            [
                "companies",
                "recon",
                "Acme",
                "--fixture",
                str(project_root / FIXTURE),
                "--apply",
            ],
            standalone_mode=False,
        )
        == SUCCESS
    )
    site = load_companies(default_companies_path(tmp_path)).find_company("acme")
    assert site is not None
    assert site.career_sites[0].ats == AtsKind.GREENHOUSE
    forms = load_form_knowledge(default_form_knowledge_path(tmp_path))
    assert forms and "Resume/CV" in [f.label for f in forms[0].fields]


def test_recon_apply_without_html_fails(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert run_cli(["companies", "recon", "Acme", "--apply"], standalone_mode=False) == (
        VALIDATION_FAILURE
    )
