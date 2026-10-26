"""Redirect resolution tests (httpx mock transport, no network)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from jobbot.portals import redirect as redirect_module
from jobbot.portals.redirect import (
    expand_urls,
    follow_redirect_url,
    read_interstitial_destination,
)

GREENHOUSE = "https://boards.greenhouse.io/acme/jobs/1"


def _client_factory(handler: httpx.MockTransport) -> object:
    def factory(**_kwargs: object) -> httpx.Client:
        return httpx.Client(transport=handler, follow_redirects=True)

    return factory


def _install(monkeypatch: pytest.MonkeyPatch, handler: httpx.MockTransport) -> None:
    monkeypatch.setattr(redirect_module, "_client", _client_factory(handler))


def test_follow_redirect_url_returns_original_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network")

    _install(monkeypatch, httpx.MockTransport(boom))
    assert follow_redirect_url("https://lnkd.in/abc") == "https://lnkd.in/abc"


def test_follow_redirect_url_follows_location(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "lnkd.in":
            return httpx.Response(302, headers={"Location": GREENHOUSE})
        return httpx.Response(200)

    _install(monkeypatch, httpx.MockTransport(handler))
    assert follow_redirect_url("https://lnkd.in/xyz") == GREENHOUSE


def test_follow_redirect_url_follows_relative_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/xyz":
            return httpx.Response(301, headers={"Location": "/acme/jobs/9"})
        return httpx.Response(200)

    _install(monkeypatch, httpx.MockTransport(handler))
    assert follow_redirect_url("https://lnkd.in/xyz") == "https://lnkd.in/acme/jobs/9"


def test_follow_redirect_url_retries_with_get_when_head_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some ATS hosts answer 405 to HEAD; the link is still resolvable with GET."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(405)
        if request.url.host == "lnkd.in":
            return httpx.Response(302, headers={"Location": GREENHOUSE})
        return httpx.Response(200)

    _install(monkeypatch, httpx.MockTransport(handler))
    assert follow_redirect_url("https://lnkd.in/xyz") == GREENHOUSE
    assert seen[0] == "HEAD"
    assert "GET" in seen


def test_follow_redirect_url_adds_scheme_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.scheme == "https"
        return httpx.Response(200)

    _install(monkeypatch, httpx.MockTransport(handler))
    assert follow_redirect_url("lnkd.in/abc") == "https://lnkd.in/abc"


def test_follow_redirect_url_ignores_blank_input() -> None:
    assert follow_redirect_url("") == ""
    assert follow_redirect_url("   ") == "   "


def test_expand_urls_keeps_order_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        redirect_module,
        "follow_redirect_url",
        lambda u: u.replace("lnkd.in/abc", "boards.greenhouse.io/acme/jobs/9"),
    )
    out = expand_urls(
        [
            "https://lnkd.in/abc",
            "https://www.linkedin.com/feed/update/1",
        ]
    )
    assert "https://boards.greenhouse.io/acme/jobs/9" in out
    assert out[0] == "https://lnkd.in/abc"


def test_expand_urls_leaves_non_short_links_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(_url: str) -> str:
        raise AssertionError("must not resolve a direct ATS link")

    monkeypatch.setattr(redirect_module, "follow_redirect_url", explode)
    assert expand_urls([GREENHOUSE]) == [GREENHOUSE]


def test_read_interstitial_destination(project_root: Path) -> None:
    html = (project_root / "tests/fixtures/lnkd_in_interstitial.html").read_text(encoding="utf-8")
    assert read_interstitial_destination(html) == (
        "https://career.example.com/en-us/vacancies/lead-data-scientist-89203"
    )


def test_read_interstitial_destination_ignores_other_linkedin_links(project_root: Path) -> None:
    html = (project_root / "tests/fixtures/lnkd_in_interstitial.html").read_text(encoding="utf-8")
    assert "answer/a1341680" not in (read_interstitial_destination(html) or "")
    assert read_interstitial_destination("<html><a href='https://x.test'>x</a></html>") is None


def test_follow_redirect_url_reads_interstitial_when_lnkd_in_answers_200(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
) -> None:
    """A lnkd.in short link that answers 200 with the leaving-site page still resolves."""
    html = (project_root / "tests/fixtures/lnkd_in_interstitial.html").read_text(encoding="utf-8")
    dest = "https://career.example.com/en-us/vacancies/lead-data-scientist-89203"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "lnkd.in":
            return httpx.Response(200, text=html)
        return httpx.Response(200, text="ok")

    _install(monkeypatch, httpx.MockTransport(handler))
    assert follow_redirect_url("https://lnkd.in/abc") == dest

