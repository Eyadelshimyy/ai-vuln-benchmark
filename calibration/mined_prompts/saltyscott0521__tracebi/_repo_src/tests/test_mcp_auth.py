"""
Gateway auth tests — the HTTP transport must not ship unauthenticated.

Three layers, mirroring the implementation:

* the ``serve()`` gate — http refuses to start without either
  ``TRACEBI_MCP_TOKEN`` or an explicit ``insecure=True``, with a message
  that says exactly what to do; stdio is untouched by any of it
* the verifier — ``StaticTokenVerifier`` accepts exactly the configured
  token, in constant time
* the wire — the streamable-http ASGI app answers 401 to a missing or
  wrong bearer and serves an MCP ``initialize`` with the right one

The serve-gate tests stub ``build_server`` so no port is ever bound; the
wire tests drive the real ASGI app through Starlette's TestClient and are
skipped when the optional ``mcp`` package is absent.
"""

import pytest

from tracebi.mcp_server import (
    GatewayAuthError,
    StaticTokenVerifier,
    serve,
)


class _StubServer:
    """Records the run() call serve() would hand to a real MCPServer."""

    def __init__(self):
        self.run_calls = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)


@pytest.fixture()
def stub_build_server(monkeypatch):
    """Replace build_server with a recorder so serve() never binds a port."""
    import tracebi.mcp_server as gw

    stub = _StubServer()
    calls = {"tokens": []}

    def fake_build_server(token=None):
        calls["tokens"].append(token)
        return stub

    monkeypatch.setattr(gw, "build_server", fake_build_server)
    return stub, calls


# ── The serve() gate ───────────────────────────────────────────────────────

def test_http_without_token_or_insecure_refuses_to_start(monkeypatch):
    monkeypatch.delenv("TRACEBI_MCP_TOKEN", raising=False)
    with pytest.raises(GatewayAuthError) as excinfo:
        serve(transport="http")
    # The refusal must say exactly what to do, not just say no.
    message = str(excinfo.value)
    assert "TRACEBI_MCP_TOKEN" in message
    assert "--insecure" in message
    assert "Authorization: Bearer" in message


def test_http_empty_token_counts_as_no_token(monkeypatch):
    monkeypatch.setenv("TRACEBI_MCP_TOKEN", "")
    with pytest.raises(GatewayAuthError):
        serve(transport="http")


def test_http_with_insecure_is_allowed(monkeypatch, stub_build_server, capsys):
    monkeypatch.delenv("TRACEBI_MCP_TOKEN", raising=False)
    stub, calls = stub_build_server
    serve(transport="http", port=9999, insecure=True)
    assert calls["tokens"] == [None], "insecure mode must not invent a token"
    assert stub.run_calls == [{
        "transport": "streamable-http", "host": "127.0.0.1", "port": 9999,
    }]
    # The operator must be able to see the posture they just chose.
    log = capsys.readouterr().err
    assert "transport=http" in log
    assert "host=127.0.0.1" in log
    assert "auth=none (--insecure)" in log
    assert "actor=mcp:agent" in log


def test_http_with_token_serves_with_bearer_auth(monkeypatch,
                                                 stub_build_server, capsys):
    monkeypatch.setenv("TRACEBI_MCP_TOKEN", "s3cret")
    stub, calls = stub_build_server
    serve(transport="http", insecure=False)
    assert calls["tokens"] == ["s3cret"]
    assert stub.run_calls == [{
        "transport": "streamable-http", "host": "127.0.0.1", "port": 8765,
    }]
    log = capsys.readouterr().err
    assert "host=127.0.0.1" in log
    assert "auth=bearer (TRACEBI_MCP_TOKEN)" in log
    assert "actor=mcp:agent" in log


def test_stdio_needs_no_token_and_stays_silent(monkeypatch,
                                               stub_build_server, capsys):
    monkeypatch.delenv("TRACEBI_MCP_TOKEN", raising=False)
    stub, calls = stub_build_server
    serve()  # defaults: stdio, no insecure flag needed
    assert calls["tokens"] == [None]
    assert stub.run_calls == [{"transport": "stdio"}]
    assert capsys.readouterr().err == "", "stdio transport is unchanged"


def test_http_passes_a_non_loopback_host_when_authenticated(
        monkeypatch, stub_build_server, capsys):
    monkeypatch.setenv("TRACEBI_MCP_TOKEN", "s3cret")
    stub, _calls = stub_build_server
    serve(transport="http", host="0.0.0.0", port=8765)
    assert stub.run_calls == [{
        "transport": "streamable-http", "host": "0.0.0.0", "port": 8765,
    }]
    assert "host=0.0.0.0" in capsys.readouterr().err


def test_insecure_non_loopback_refuses(monkeypatch, stub_build_server):
    monkeypatch.delenv("TRACEBI_MCP_TOKEN", raising=False)
    stub, _calls = stub_build_server
    with pytest.raises(GatewayAuthError) as excinfo:
        serve(transport="http", host="0.0.0.0", insecure=True)
    assert stub.run_calls == []
    message = str(excinfo.value)
    assert "non-loopback" in message
    assert "--allow-insecure-bind" in message


def test_insecure_non_loopback_allowed_with_explicit_flag(
        monkeypatch, stub_build_server):
    monkeypatch.delenv("TRACEBI_MCP_TOKEN", raising=False)
    stub, _calls = stub_build_server
    serve(transport="http", host="0.0.0.0", insecure=True,
          allow_insecure_bind=True)
    assert stub.run_calls == [{
        "transport": "streamable-http", "host": "0.0.0.0", "port": 8765,
    }]


def test_cli_passes_host(monkeypatch):
    from tracebi.cli import main

    seen = {}

    def fake_serve(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr("tracebi.mcp_server.serve", fake_serve)
    rc = main(["mcp", "--transport", "http", "--host", "0.0.0.0",
               "--port", "9000"])
    assert rc == 0
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 9000
    assert seen["transport"] == "http"


def test_cli_refusal_is_printed_and_exits_nonzero(monkeypatch, capsys):
    from tracebi.cli import main

    monkeypatch.delenv("TRACEBI_MCP_TOKEN", raising=False)
    rc = main(["mcp", "--transport", "http"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "TRACEBI_MCP_TOKEN" in err
    assert "--insecure" in err


# ── The verifier ───────────────────────────────────────────────────────────

def _verify(verifier, presented):
    anyio = pytest.importorskip("anyio")
    return anyio.run(verifier.verify_token, presented)


def test_verifier_accepts_the_configured_token():
    pytest.importorskip("mcp")
    access = _verify(StaticTokenVerifier("s3cret"), "s3cret")
    assert access is not None
    assert access.client_id == "mcp:agent"


def test_verifier_rejects_wrong_and_prefix_tokens():
    pytest.importorskip("mcp")
    verifier = StaticTokenVerifier("s3cret")
    for wrong in ("nope", "s3cre", "s3cret2", ""):
        assert _verify(verifier, wrong) is None


def test_verifier_carries_the_configured_actor(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("TRACEBI_MCP_ACTOR", "quarterly-bot")
    access = _verify(StaticTokenVerifier("s3cret"), "s3cret")
    assert access.client_id == "mcp:quarterly-bot"


# ── The wire: 401s from the real ASGI app ──────────────────────────────────

@pytest.fixture()
def http_client():
    pytest.importorskip("mcp")
    from starlette.testclient import TestClient

    from tracebi.mcp_server import build_server

    app = build_server(token="s3cret").streamable_http_app()
    # base_url sets a Host header the transport's DNS-rebinding protection
    # accepts, so a correct token reaches the MCP layer instead of a 421.
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        yield client


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}

_ACCEPT = {"Accept": "application/json, text/event-stream"}


def test_missing_bearer_is_401(http_client):
    resp = http_client.post("/mcp", json=_INITIALIZE, headers=_ACCEPT)
    assert resp.status_code == 401
    assert "Bearer" in resp.headers["www-authenticate"]


def test_wrong_bearer_is_401(http_client):
    resp = http_client.post(
        "/mcp", json=_INITIALIZE,
        headers={**_ACCEPT, "Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_correct_bearer_initializes(http_client):
    resp = http_client.post(
        "/mcp", json=_INITIALIZE,
        headers={**_ACCEPT, "Authorization": "Bearer s3cret"},
    )
    assert resp.status_code == 200
    assert '"serverInfo"' in resp.text
