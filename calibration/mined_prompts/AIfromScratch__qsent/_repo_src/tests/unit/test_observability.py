"""
Unit tests for structured logging, trace context, and TraceMiddleware.
"""
import re

import structlog.contextvars
from fastapi.testclient import TestClient

from qsf.api.main import app
from qsf.common.logging import configure_logging, get_current_trace, get_langfuse, set_current_trace

client = TestClient(app)

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# Trace context ContextVar
# ---------------------------------------------------------------------------


def test_get_current_trace_returns_none_by_default():
    set_current_trace(None)
    assert get_current_trace() is None


def test_set_and_get_current_trace_round_trip():
    sentinel = object()
    set_current_trace(sentinel)
    assert get_current_trace() is sentinel
    set_current_trace(None)


# ---------------------------------------------------------------------------
# Langfuse
# ---------------------------------------------------------------------------


def test_init_langfuse_returns_none_without_env_var(monkeypatch):
    # _init_langfuse() checks LANGFUSE_PUBLIC_KEY at call time, unlike the module-level
    # singleton which is already initialized from whatever env vars the process started
    # with (e.g. via `doppler run`), not from a .env file — no such file is loaded for tests.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    from qsf.common.logging import _init_langfuse
    assert _init_langfuse() is None


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def test_configure_logging_does_not_raise():
    configure_logging()


def test_configure_logging_json_mode_does_not_raise(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging()


# ---------------------------------------------------------------------------
# TraceMiddleware — X-Trace-ID header
# ---------------------------------------------------------------------------


def test_trace_id_header_present_on_health():
    response = client.get("/health")
    assert "x-trace-id" in response.headers


def test_trace_id_is_32_char_hex():
    response = client.get("/health")
    trace_id = response.headers.get("x-trace-id", "")
    assert TRACE_ID_RE.match(trace_id), f"Expected 32-char hex, got: {trace_id!r}"


def test_trace_id_is_unique_per_request():
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert r1.headers["x-trace-id"] != r2.headers["x-trace-id"]


def test_trace_id_present_on_unauthenticated_response():
    response = client.get("/auth/me")
    assert response.status_code == 401
    assert "x-trace-id" in response.headers
