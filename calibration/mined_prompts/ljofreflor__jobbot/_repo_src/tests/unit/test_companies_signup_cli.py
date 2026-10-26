"""`companies signup` prepares the registration; the human performs it."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.exit_codes import SUCCESS


def _workspace(tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)


def _learn_company(tmp_path: Path) -> None:
    from jobbot.cli import run_cli

    run_cli(
        [
            "companies",
            "learn",
            "https://acme.wd3.myworkdayjobs.com/careers",
            "--company",
            "Acme",
            "--country",
            "CL",
        ],
        standalone_mode=False,
    )


def test_it_lists_what_registering_will_ask_without_opening_anything(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    _learn_company(tmp_path)
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )

    code = run_cli(["companies", "signup", "Acme", "--no-open"], standalone_mode=False)
    out = capsys.readouterr().out

    assert code == SUCCESS
    assert opened == [], "--no-open must not open a browser"
    assert "Email" in out
    assert "you type the password" in out
    assert "fill fields the profile already answers" in out or "signup --apply" in out


def test_opening_is_the_only_side_effect(
    tmp_path: Path, project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path, project_root, monkeypatch)
    _learn_company(tmp_path)
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )

    assert run_cli(["companies", "signup", "Acme"], standalone_mode=False) == SUCCESS

    assert opened == ["https://acme.wd3.myworkdayjobs.com/careers"]


def test_apply_without_confirm_skips_browser_and_does_not_submit(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--apply` without confirm must not open a browser or submit."""
    _workspace(tmp_path, project_root, monkeypatch)
    _learn_company(tmp_path)
    from jobbot import cli
    from jobbot.cli import run_cli

    opened: list[str] = []
    monkeypatch.setattr(
        "jobbot.browser.background.open_url", lambda url, **_k: opened.append(url) or url
    )
    monkeypatch.setattr(cli.typer, "confirm", lambda *_a, **_k: False)

    sessions: list[object] = []

    class BoomSession:
        def __init__(self, *args: object, **kwargs: object) -> None:
            sessions.append((args, kwargs))
            raise AssertionError("BrowserSession must not start without confirm")

        def __enter__(self) -> object:
            raise AssertionError("unreachable")

        def __exit__(self, *_a: object) -> None:
            return None

    monkeypatch.setattr("jobbot.browser.session.BrowserSession", BoomSession)

    code = run_cli(
        ["companies", "signup", "Acme", "--apply", "--no-open"],
        standalone_mode=False,
    )
    out = " ".join(capsys.readouterr().out.split())

    assert code == SUCCESS
    assert sessions == []
    assert opened == []
    assert "fill plan" in out.casefold() or "will fill" in out.casefold()
    assert "skipped" in out.casefold()
