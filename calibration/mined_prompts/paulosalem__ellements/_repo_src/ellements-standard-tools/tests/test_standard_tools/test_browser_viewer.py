"""Tests for browser display helpers."""

from __future__ import annotations

import urllib.request

from ellements.standard_tools.web.browser_viewer import BrowserViewer, is_remote_url


def test_is_remote_url_only_accepts_http_urls() -> None:
    assert is_remote_url("https://example.com")
    assert is_remote_url("http://example.com/report.html")
    assert not is_remote_url("file:///tmp/report.html")
    assert not is_remote_url("/tmp/report.html")


def test_open_remote_can_skip_system_browser(monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    result = BrowserViewer().open_remote("https://example.com", open_browser=False)

    assert result.display_url == "https://example.com"
    assert result.source_type == "remote-url"
    assert result.opened is False
    assert opened == []


def test_serve_local_file_exposes_localhost_url(tmp_path) -> None:
    report = tmp_path / "report.html"
    report.write_text("<!doctype html><h1>ok</h1>", encoding="utf-8")

    server, result = BrowserViewer().serve_local(report, open_browser=False)
    try:
        assert result.source_type == "local-file"
        assert result.display_url.startswith("http://127.0.0.1:")
        body = urllib.request.urlopen(result.display_url, timeout=5).read().decode()
        assert "<h1>ok</h1>" in body
    finally:
        server.shutdown()
