"""Opening a page must leave the focus on the terminal (the user's request)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from jobbot.browser import background
from jobbot.exit_codes import SUCCESS


def test_macos_opens_with_open_g_and_never_raises_the_browser() -> None:
    spawned: list[list[str]] = []

    def fallback(*_a: object, **_k: object) -> bool:
        raise AssertionError("macOS must not go through webbrowser")

    background.open_url(
        "https://boards.example.com/jobs/1",
        platform="darwin",
        spawn=spawned.append,
        fallback=fallback,
    )

    assert spawned == [["open", "-g", "https://boards.example.com/jobs/1"]]


def test_focus_opt_in_drops_the_background_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setenv(background.FOCUS_ENV, "1")

    background.open_url("https://example.com/", platform="darwin", spawn=spawned.append)

    assert spawned == [["open", "https://example.com/"]]


def test_other_platforms_use_webbrowser_without_autoraise() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fallback(url: str, **kwargs: object) -> bool:
        calls.append((url, kwargs))
        return True

    background.open_url("https://example.com/a", platform="linux", fallback=fallback)

    assert calls == [("https://example.com/a", {"new": 2, "autoraise": False})]


def test_missing_open_binary_falls_back_without_autoraise() -> None:
    calls: list[dict[str, object]] = []

    def spawn(_argv: list[str]) -> None:
        raise FileNotFoundError("open")

    def fallback(_url: str, **kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    background.open_url("https://example.com/", platform="darwin", spawn=spawn, fallback=fallback)

    assert calls == [{"new": 2, "autoraise": False}]


def test_notice_tells_where_the_tab_is_instead_of_focusing() -> None:
    line = background.background_notice(
        "https://example.com/apply", where="Chrome", title="Apply now"
    )

    assert "https://example.com/apply" in line
    assert "segundo plano" in line
    assert "Chrome" in line
    assert "Apply now" in line


def test_chrome_launch_on_macos_goes_through_open_g_new_instance() -> None:
    argv = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--remote-debugging-port=9230",
        "--user-data-dir=/tmp/jobbot-profile",
        "https://example.com/",
    ]

    launched = background.background_launch_argv(argv, platform="darwin")

    assert launched == [
        "open",
        "-g",
        "-n",
        "-a",
        "/Applications/Google Chrome.app",
        "--args",
        "--remote-debugging-port=9230",
        "--user-data-dir=/tmp/jobbot-profile",
        "https://example.com/",
    ]


def test_chrome_launch_outside_macos_or_with_focus_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = ["/usr/bin/google-chrome", "--remote-debugging-port=9230", "https://example.com/"]
    assert background.background_launch_argv(argv, platform="linux") == argv

    bundle = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "x"]
    monkeypatch.setenv(background.FOCUS_ENV, "true")
    assert background.background_launch_argv(bundle, platform="darwin") == bundle


class _Page:
    def __init__(self) -> None:
        self.url = "about:blank"

    def bring_to_front(self) -> None:
        raise AssertionError("JobBot never brings a tab to the front")


class _PageInfo:
    def __init__(self) -> None:
        self.value: _Page | None = None


class _Context:
    def __init__(self) -> None:
        self.new_pages = 0
        self.created: _Page | None = None

    def new_page(self) -> _Page:
        self.new_pages += 1
        return _Page()

    @contextmanager
    def expect_page(self, **_kwargs: object) -> Iterator[_PageInfo]:
        info = _PageInfo()
        yield info
        info.value = self.created


class _CdpSession:
    def __init__(self, context: _Context) -> None:
        self._context = context
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.detached = False

    def send(self, method: str, params: dict[str, Any]) -> dict[str, str]:
        self.sent.append((method, params))
        self._context.created = _Page()
        return {"targetId": "T1"}

    def detach(self) -> None:
        self.detached = True


class _Browser:
    def __init__(self, context: _Context) -> None:
        self.contexts = [context]
        self.session = _CdpSession(context)

    def new_browser_cdp_session(self) -> _CdpSession:
        return self.session


def test_cdp_tab_is_created_in_the_background_and_never_raised() -> None:
    context = _Context()
    browser = _Browser(context)

    page = background.new_background_page(context, browser=browser)

    assert browser.session.sent == [
        ("Target.createTarget", {"url": "about:blank", "background": True})
    ]
    assert page is context.created
    assert context.new_pages == 0
    assert browser.session.detached is True


def test_cdp_tab_with_focus_opt_in_uses_a_normal_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _Context()
    browser = _Browser(context)
    monkeypatch.setenv(background.FOCUS_ENV, "yes")

    background.new_background_page(context, browser=browser)

    assert browser.session.sent == []
    assert context.new_pages == 1


def test_application_open_goes_through_the_background_opener(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "data" / "profile.yaml").write_text(
        (project_root / "data" / "profile.example.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / ".jobbot.toml").write_text(
        f'[paths]\ntemplates = "{project_root / "templates"}"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    import jobbot.cli as cli
    from jobbot.cli import run_cli
    from jobbot.db.engine import make_engine, make_session_factory
    from jobbot.jobs.repository import JobRepository
    from jobbot.models.job import JobPosting

    monkeypatch.setattr(
        "jobbot.jobs.closure.fetch_posting_text",
        lambda _url, **_k: "<html>still open</html>",
    )
    opened: list[str] = []

    def fake_open(url: str, **_kwargs: object) -> str:
        opened.append(url)
        return background.background_notice(url)

    monkeypatch.setattr("jobbot.browser.background.open_url", fake_open)
    config = cli.load_config()
    session = make_session_factory(make_engine(config.database_path))()
    url = "https://boards.greenhouse.io/acme/jobs/1"
    stored = JobRepository(session).upsert_external(
        JobPosting.model_validate(
            {
                "id": "PENDING",
                "source": "manual",
                "title": "Un cargo",
                "company": "Acme",
                "url": url,
                "ats_url": url,
                "ats_kind": "greenhouse",
                "description": "Una descripción.",
            }
        )
    )

    code = run_cli(["application", "open", stored.id], standalone_mode=False)

    assert code == SUCCESS
    assert opened == [url]
    assert "segundo plano" in capsys.readouterr().out
