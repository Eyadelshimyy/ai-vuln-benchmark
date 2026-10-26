"""Indeed HITL login helpers (magic / continue links)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jobbot.adapters.indeed.client import normalize_indeed_continue_url
from jobbot.exit_codes import SUCCESS, VALIDATION_FAILURE


def test_normalize_accepts_secure_indeed_auth_link() -> None:
    url = "https://secure.indeed.com/auth?continue=https%3A%2F%2Fcl.indeed.com%2F"
    assert normalize_indeed_continue_url(url) == url


def test_normalize_accepts_smartredirect() -> None:
    url = "https://smartredirect.indeed.com/redirect?tok=abc"
    assert normalize_indeed_continue_url(url) == url


def test_normalize_rejects_non_indeed() -> None:
    with pytest.raises(ValueError, match="Indeed link"):
        normalize_indeed_continue_url("https://evil.example/login")


def test_normalize_splits_double_paste() -> None:
    a = "https://secure.indeed.com/auth?hl=es"
    glued = a + a
    assert normalize_indeed_continue_url(glued) == a


def test_cli_continue_url_rejects_foreign_host(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    from jobbot.cli import run_cli

    assert (
        run_cli(
            ["indeed", "login", "--continue-url", "https://example.com/x"],
            standalone_mode=False,
        )
        == VALIDATION_FAILURE
    )


def test_cli_login_help_mentions_continue_url(capsys: pytest.CaptureFixture[str]) -> None:
    from jobbot.cli import run_cli
    from tests.conftest import plain_cli_text

    assert run_cli(["indeed", "login", "--help"], standalone_mode=False) == SUCCESS
    text = plain_cli_text(capsys.readouterr().out)
    assert "--continue-url" in text
    assert "magic" in text.casefold() or "email" in text.casefold()
