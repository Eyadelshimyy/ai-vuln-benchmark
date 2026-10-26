"""`jobbot jobs conditions`: review a stored posting's conditions before applying."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobbot.exit_codes import GENERIC_FAILURE, SUCCESS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jobs"


@pytest.fixture
def workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _add(name: str) -> None:
    from jobbot.cli import run_cli

    assert run_cli(["jobs", "add", "--file", str(FIXTURES / name)], standalone_mode=False) in (
        SUCCESS,
        None,
    )


def test_dealbreaker_prints_a_banner_and_exits_non_zero(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from jobbot.cli import run_cli

    _add("conditions_residency_en.txt")
    capsys.readouterr()

    code = run_cli(["jobs", "conditions", "J0001"], standalone_mode=False)

    captured = capsys.readouterr()
    out = captured.out
    assert code == GENERIC_FAILURE
    # A dealbreaker is an answer, not a defect: it must not land in ops failures.
    assert "Recorded failure" not in captured.err
    assert "DEALBREAKER" in out
    assert "must reside in" in out.casefold()
    assert "application_answers.yaml" in out


def test_no_dealbreaker_exits_zero_and_reads_the_answers_file(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from jobbot.cli import run_cli

    _add("conditions_residency_es.txt")
    (workspace / "data" / "application_answers.yaml").write_text(
        "languages:\n  english: C1\naccepts_contractor: true\n", encoding="utf-8"
    )
    capsys.readouterr()

    code = run_cli(["jobs", "conditions", "J0001"], standalone_mode=False)

    out = capsys.readouterr().out
    assert code == SUCCESS
    assert "DEALBREAKER" not in out
    assert "✅" in out


def test_json_output_lists_verdicts(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from jobbot.cli import run_cli

    _add("conditions_residency_es.txt")
    capsys.readouterr()

    run_cli(["jobs", "conditions", "J0001", "--json"], standalone_mode=False)

    data = json.loads(capsys.readouterr().out)
    assert data[0]["job_id"] == "J0001"
    kinds = {v["kind"] for v in data[0]["verdicts"]}
    assert {"residency", "language", "contract"} <= kinds


def test_unknown_job_is_an_error(workspace: Path) -> None:
    from jobbot.cli import run_cli

    assert run_cli(["jobs", "conditions", "J0999"], standalone_mode=False) == GENERIC_FAILURE
