"""The recruiters lifecycle from the CLI: discover, promote, then it advises."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import SUCCESS
from jobbot.recruiters.sources import default_recruiters_path, load_sources

FIXTURE = Path("tests/fixtures/recruiters/hiring_notes.html")
URL = "https://hiring.example.org/how-we-screen"


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def _discover(project_root: Path) -> int:
    from jobbot.cli import run_cli

    return run_cli(
        ["recruiters", "discover", URL, "--fixture", str(project_root / FIXTURE)],
        standalone_mode=False,
    )


def test_discovered_knowledge_starts_as_a_candidate(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)

    assert _discover(project_root) == SUCCESS

    sources = load_sources(default_recruiters_path(tmp_path))
    assert len(sources) == 1
    assert sources[0].status.value == "candidate"
    assert sources[0].practices


def test_a_candidate_does_not_reach_cv_advise_until_promoted(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.recruiters.playbook import advisor_notes

    _discover(project_root)
    assert advisor_notes(tmp_path) == []

    assert run_cli(["recruiters", "promote", URL, "--yes"], standalone_mode=False) == SUCCESS

    assert advisor_notes(tmp_path)


def test_export_shares_practices_and_no_people(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    _discover(project_root)
    run_cli(["recruiters", "promote", URL, "--yes"], standalone_mode=False)
    out = tmp_path / "shared.yaml"

    assert run_cli(["recruiters", "export", "--out", str(out)], standalone_mode=False) == SUCCESS

    shared = out.read_text(encoding="utf-8")
    assert "practices" in shared
    assert "Jane Doe" not in shared
    assert "jane.doe@somecorp.example" not in shared


def test_a_rejected_source_stays_out(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli
    from jobbot.recruiters.playbook import advisor_notes

    _discover(project_root)
    assert run_cli(["recruiters", "reject", URL], standalone_mode=False) == SUCCESS

    assert load_sources(default_recruiters_path(tmp_path))[0].status.value == "rejected"
    assert advisor_notes(tmp_path) == []


def test_discover_without_a_url_is_refused(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suite must never reach the network by accident."""
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert run_cli(["recruiters", "discover"], standalone_mode=False) != SUCCESS
    assert not default_recruiters_path(tmp_path).exists()


def test_cv_advise_uses_promoted_practice(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The loop's last link: what hiring says, reaching your CV as a suggestion."""
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    _discover(project_root)
    run_cli(["recruiters", "promote", URL, "--yes"], standalone_mode=False)
    capsys.readouterr()

    assert run_cli(["cv", "advise", "--limit", "8"], standalone_mode=False) == SUCCESS

    out = " ".join(capsys.readouterr().out.split())
    assert "recruiter knowledge you promoted" in out, "the promoted practice must be cited"
    assert "Quantify the outcome" in out or "seven seconds" in out
