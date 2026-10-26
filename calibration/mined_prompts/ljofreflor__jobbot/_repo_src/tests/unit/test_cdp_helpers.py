"""CDP helper unit tests."""

from pathlib import Path

from jobbot.browser.cdp import (
    cdp_http_url,
    chrome_debug_argv,
    find_chrome_executable,
    resolve_cdp_url,
)


def test_resolve_cdp_url_prefers_explicit() -> None:
    assert resolve_cdp_url("http://127.0.0.1:9333", env={"JOBBOT_CDP_URL": "x"}) == (
        "http://127.0.0.1:9333"
    )


def test_resolve_cdp_url_from_env() -> None:
    assert resolve_cdp_url(None, env={"JOBBOT_CDP_URL": "http://127.0.0.1:9222"}) == (
        "http://127.0.0.1:9222"
    )


def test_resolve_cdp_url_empty_disables() -> None:
    assert resolve_cdp_url("", env={"JOBBOT_CDP_URL": "http://x"}) is None
    assert resolve_cdp_url(None, env={"JOBBOT_CDP_URL": ""}) is None


def test_chrome_debug_argv_includes_port_and_profile() -> None:
    chrome = find_chrome_executable()
    if chrome is None:
        return
    argv = chrome_debug_argv(profile_dir=Path("/tmp/jobbot-cdp"), port=9222, chrome=chrome)
    assert "--remote-debugging-port=9222" in argv
    assert "--user-data-dir=/tmp/jobbot-cdp" in argv
    assert cdp_http_url(9222) == "http://127.0.0.1:9222"
