"""
Clients learning about, and getting, a newer TraceBi (tracebi/_updates.py).

GitHub is never contacted: a local HTTP server stands in for the
releases/latest endpoint via TRACEBI_UPDATE_URL.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tracebi import _updates


@pytest.fixture
def feed(monkeypatch, tmp_path):
    """A fake releases/latest endpoint; set .release to change its answer."""
    state = {"release": {"tag_name": "v9.9.0", "html_url": "https://example.test/r/v9.9.0",
                         "published_at": "2026-09-25T00:00:00Z", "body": "- Faster\n- Better",
                         "assets": [{"name": "tracebi-9.9.0-py3-none-any.whl",
                                     "browser_download_url": "https://example.test/tracebi-9.9.0-py3-none-any.whl"}]},
             "hits": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"] += 1
            body = json.dumps(state["release"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("TRACEBI_UPDATE_CHECK", "1")
    monkeypatch.setenv("TRACEBI_UPDATE_URL", f"http://127.0.0.1:{server.server_port}/latest")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(_updates, "get_version", lambda: "0.6.0")
    monkeypatch.setattr(_updates, "_last_attempt", 0.0)   # no throttle carried between tests
    yield state
    server.shutdown()


@pytest.mark.parametrize("latest, current, newer", [
    ("0.6.0", "0.6.0.dev0", True), ("0.7.0", "0.6.0", True), ("v0.6.1", "0.6.0", True),
    ("0.6.0", "0.6.0", False), ("0.6.0rc1", "0.6.0", False), ("0.6.0", "unknown", False),
])
def test_version_order(latest, current, newer):
    assert _updates.is_newer(latest, current) is newer
    # the fallback parser (no `packaging`) agrees
    a, b = _updates._fallback_key(latest), _updates._fallback_key(current)
    assert (a is not None and b is not None and a > b) is newer


def test_a_newer_release_says_how_each_install_updates(feed, monkeypatch):
    for kind, needle in [("pip", 'pip install --upgrade "https://example.test/tracebi-9.9.0-py3-none-any.whl"'),
                         ("docker", "TRACEBI_VERSION=9.9.0 docker compose -f deploy/compose.yml pull"),
                         ("checkout", "git checkout v9.9.0")]:
        monkeypatch.setattr(_updates, "install_kind", lambda k=kind: k)
        st = _updates.status(force=True)
        assert st["available"] is True and st["latest"] == "9.9.0"
        assert needle in st["command"], (kind, st["command"])


def test_up_to_date_and_the_cache(feed):
    feed["release"]["tag_name"] = "v0.6.0"
    st = _updates.status(force=True)
    assert st["available"] is False and st["latest"] == "0.6.0"
    hits = feed["hits"]
    _updates.status()                      # within a day: answered from cache
    assert feed["hits"] == hits


def test_off_means_no_request(feed, monkeypatch):
    monkeypatch.setenv("TRACEBI_UPDATE_CHECK", "0")
    st = _updates.status(force=True)
    assert st["checked"] is False and st["latest"] is None and feed["hits"] == 0


def test_offline_is_quiet(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACEBI_UPDATE_CHECK", "1")
    monkeypatch.setenv("TRACEBI_UPDATE_URL", "http://127.0.0.1:9/nothing-listens-here")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    st = _updates.status(force=True)
    assert st["latest"] is None and st["available"] is False


def test_the_cli_prints_whats_new_and_the_command(feed, monkeypatch, capsys):
    from tracebi import cli

    monkeypatch.setattr(_updates, "install_kind", lambda: "docker")
    assert cli.main(["update", "--check"]) == 0
    out = capsys.readouterr().out
    assert "latest:    9.9.0" in out and "- Faster" in out
    assert "docker compose -f deploy/compose.yml pull" in out
    assert "tracebi verify" in out


def test_status_endpoint_never_waits_on_the_network(feed, monkeypatch):
    """/api/status answers from the cache; a missing cache is filled in the
    background, and the next call reports it."""
    import time

    st = _updates.status(wait=False)
    assert st["latest"] is None                    # nothing cached yet
    for _ in range(50):
        if _updates.status(wait=False)["latest"]:
            break
        time.sleep(0.05)
    assert _updates.status(wait=False)["available"] is True
