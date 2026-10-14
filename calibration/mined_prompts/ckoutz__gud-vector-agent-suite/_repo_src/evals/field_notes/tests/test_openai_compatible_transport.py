"""Offline tests for the live transport's request shape and response parsing."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from field_notes.adapters.base import AdapterUnavailableError
from field_notes.transports.openai_compatible import (
    DEFAULT_BASE_URL,
    OpenAICompatibleTransport,
    _request_body,
    _to_response,
)


def _call(transport: OpenAICompatibleTransport, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "request_id": "case:0",
        "base_url": None,
        "api_key": "test-key",
        "model": "vendor/model",
        "system": "system",
        "user_content": "user",
        "json_schema": {"type": "object"},
        "parameters": {"temperature": 0.0, "max_output_tokens": 128},
    }
    kwargs.update(overrides)
    return transport.complete(**kwargs)


def test_request_body_holds_decoding_settings_and_requests_usage() -> None:
    body = _request_body(
        model="vendor/model",
        system="system",
        user_content="user",
        json_schema={"type": "object"},
        parameters={"temperature": 0.0, "max_output_tokens": 128},
        strict_schema=False,
    )

    assert body["model"] == "vendor/model"
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.0
    assert "max_output_tokens" not in body
    assert body["usage"] == {"include": True}
    assert body["response_format"]["json_schema"]["strict"] is False
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_response_keeps_reported_cost_and_usage() -> None:
    response = _to_response(
        {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "total_tokens": 16,
                "cost": 0.0004,
            },
        }
    )

    assert response.raw == '{"ok": true}'
    assert response.cost_usd == 0.0004
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.total_tokens == 16


def test_response_without_choices_is_reported_as_unavailable() -> None:
    with pytest.raises(AdapterUnavailableError):
        _to_response({"error": {"message": "no capacity"}})


def test_missing_api_key_never_reaches_the_network() -> None:
    transport = OpenAICompatibleTransport()
    with pytest.raises(AdapterUnavailableError):
        _call(transport, api_key=None)


def test_transport_posts_to_the_endpoint_and_returns_the_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        seen["url"] = url
        seen["headers"] = kwargs["headers"]
        seen["json"] = kwargs["json"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}], "usage": {"cost": 0.001}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    response = _call(OpenAICompatibleTransport())

    assert seen["url"] == f"{DEFAULT_BASE_URL}/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert response.cost_usd == 0.001


def test_retryable_status_is_retried_then_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(503, text="unavailable", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("field_notes.transports.openai_compatible._sleep", lambda _: None)
    transport = OpenAICompatibleTransport(max_attempts=2)

    with pytest.raises(AdapterUnavailableError):
        _call(transport)
    assert attempts["count"] == 2
