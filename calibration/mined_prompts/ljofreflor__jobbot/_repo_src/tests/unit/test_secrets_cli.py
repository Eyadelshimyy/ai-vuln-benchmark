"""`jobbot secrets` stores passwords the human types; it never prints them."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE

PASSWORD = "cli-only-secret-value"


class _Stdin(StringIO):
    def isatty(self) -> bool:
        return False


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def test_list_without_a_vault_is_quiet(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert run_cli(["secrets", "list"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert "No vault" in out
    assert "secrets init" in out


def test_init_set_list_never_echo_the_password(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert run_cli(["secrets", "init"], standalone_mode=False) == SUCCESS
    monkeypatch.setattr("sys.stdin", _Stdin(PASSWORD))
    assert (
        run_cli(
            ["secrets", "set", "indeed", "--password-stdin"],
            standalone_mode=False,
        )
        == SUCCESS
    )
    capsys.readouterr()
    assert run_cli(["secrets", "list"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert PASSWORD not in out
    assert "indeed" in out
    assert "password=***" in out
    assert "fill_login=off" in out


def test_empty_stdin_password_is_refused(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert run_cli(["secrets", "init"], standalone_mode=False) == SUCCESS
    monkeypatch.setattr("sys.stdin", _Stdin(""))
    assert (
        run_cli(
            ["secrets", "set", "indeed", "--password-stdin"],
            standalone_mode=False,
        )
        == VALIDATION_FAILURE
    )


def test_allow_fill_with_yes_is_consent_only(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    from jobbot.cli import run_cli

    assert run_cli(["secrets", "init"], standalone_mode=False) == SUCCESS
    assert run_cli(["secrets", "allow-fill", "--yes"], standalone_mode=False) == SUCCESS
    out = capsys.readouterr().out
    assert "fill_login=on" in out
    assert "not shipped" in out
    assert run_cli(["secrets", "deny-fill"], standalone_mode=False) == SUCCESS
    capsys.readouterr()
    assert run_cli(["secrets", "list"], standalone_mode=False) == SUCCESS
    assert "fill_login=off" in capsys.readouterr().out
