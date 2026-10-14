"""MLX provider tests against a mock HTTP server. CI does not need the model."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.ai.mlx_provider import MlxProvider
from app.ai.provider import CompletionRequest
from app.ai.structured import StructuredParseError, extract_json_object
from app.ai.thinking import split_thinking
from app.core.config import Settings
from app.plugins import get_plugin_catalog, reset_plugin_catalog


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "ai_provider": "mlx",
        "ai_model": "gpt-4o-mini",
        "ai_api_key": "",
        "ai_base_url": "",
        "mlx_base_url": "http://127.0.0.1:8080/v1",
        "mlx_model": "Qwen3.6-35B-A3B-8bit",
        "discovery_mode": "disabled",
        "discovery_languages": "",
        "discovery_excluded_topics": "",
        "discovery_excluded_owners": "",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _chat_response(
    content: str,
    *,
    status: int = 200,
    reasoning: str | None = None,
    usage: dict[str, int] | None = None,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": content,
                    **({"reasoning_content": reasoning} if reasoning else {}),
                }
            }
        ],
        "usage": usage or {"prompt_tokens": 11, "completion_tokens": 7},
    }
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    return httpx.Response(status, json=payload, request=request)


def _patch_client(post_response: object, get_response: object | None = None):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if isinstance(post_response, Exception):
        mock_client.post = AsyncMock(side_effect=post_response)
    else:
        mock_client.post = AsyncMock(return_value=post_response)
    if get_response is None:
        models = MagicMock(spec=httpx.Response)
        models.status_code = 200
        models.json.return_value = {"data": [{"id": "Qwen3.6-35B-A3B-8bit"}]}
        mock_client.get = AsyncMock(return_value=models)
    elif isinstance(get_response, Exception):
        mock_client.get = AsyncMock(side_effect=get_response)
    else:
        mock_client.get = AsyncMock(return_value=get_response)
    return patch("app.ai.mlx_provider.httpx.AsyncClient", return_value=mock_client), mock_client


@pytest.fixture(autouse=True)
def _catalog() -> None:
    reset_plugin_catalog()
    yield
    reset_plugin_catalog()


def test_mlx_registered_and_configurable() -> None:
    catalog = get_plugin_catalog()
    assert catalog.ai_providers.has("mlx")
    provider = catalog.ai_providers.create("mlx", make_settings())
    assert provider.provider_name == "mlx"
    assert provider.model_name == "Qwen3.6-35B-A3B-8bit"
    custom = catalog.ai_providers.create(
        "mlx", make_settings(ai_model="local-other-model", ai_provider="mlx")
    )
    assert custom.model_name == "local-other-model"


def test_split_thinking_strips_trace() -> None:
    thinking, final = split_thinking('<think>secret chain</think>\n{"ok": true}')
    assert thinking == "secret chain"
    assert final == '{"ok": true}'
    thinking2, final2 = split_thinking("hello", reasoning_field="internal")
    assert thinking2 == "internal"
    assert final2 == "hello"


def test_extract_json_from_fence_and_rejects_garbage() -> None:
    data = extract_json_object(
        'prefix\n```json\n{"title": "x", "hypothesis": "y", "not_verified": true}\n```'
    )
    assert data["title"] == "x"
    with pytest.raises(StructuredParseError):
        extract_json_object("not json at all")
    with pytest.raises(StructuredParseError):
        extract_json_object('{"title": "x"}', required_keys=("hypothesis", "not_verified"))


@pytest.mark.asyncio
async def test_successful_clean_response() -> None:
    body = json.dumps({"title": "t", "hypothesis": "h", "not_verified": True})
    response = _chat_response(body)
    patcher, client = _patch_client(response)
    with patcher:
        provider = MlxProvider(model="local-model", thinking_enabled=False)
        result = await provider.complete(
            CompletionRequest(system_prompt="sys", user_message="u", json_mode=True)
        )
    assert result.error is None
    assert result.content == body
    assert result.thinking is None
    assert result.usage.prompt_tokens == 11
    sent = client.post.await_args.kwargs["json"]
    assert sent["enable_thinking"] is False
    assert "response_format" not in sent


@pytest.mark.asyncio
async def test_thinking_response_separated() -> None:
    response = _chat_response(
        '<think>step by step</think>\n{"title": "t", "hypothesis": "h", "not_verified": true}',
        reasoning=None,
    )
    patcher, client = _patch_client(response)
    with patcher:
        provider = MlxProvider(model="local-model", thinking_enabled=True)
        result = await provider.complete(
            CompletionRequest(system_prompt="sys", user_message="u", json_mode=True, thinking=True)
        )
    assert result.thinking == "step by step"
    assert "step by step" not in result.content
    assert result.error is None
    sent = client.post.await_args.kwargs["json"]
    assert sent["enable_thinking"] is True


@pytest.mark.asyncio
async def test_malformed_json_is_error_not_evidence() -> None:
    response = _chat_response("I think this is vulnerable because reasons")
    patcher, _client = _patch_client(response)
    with patcher:
        provider = MlxProvider(model="local-model")
        result = await provider.complete(
            CompletionRequest(system_prompt="sys", user_message="u", json_mode=True)
        )
    assert result.error is not None
    assert "malformed_structured_output" in result.error


@pytest.mark.asyncio
async def test_timeout() -> None:
    patcher, _client = _patch_client(httpx.TimeoutException("slow"))
    with patcher:
        provider = MlxProvider(model="local-model", max_retries=0)
        result = await provider.complete(CompletionRequest(user_message="u"))
    assert result.error is not None
    assert "timeout" in result.error


@pytest.mark.asyncio
async def test_http_errors() -> None:
    for code, needle in ((401, "401"), (429, "429"), (500, "500")):
        request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
        response = httpx.Response(code, request=request, text="nope")
        patcher, _client = _patch_client(response)
        with patcher:
            provider = MlxProvider(model="local-model", max_retries=0)
            result = await provider.complete(CompletionRequest(user_message="u"))
        assert result.error is not None
        assert needle in result.error


@pytest.mark.asyncio
async def test_model_missing() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    response = httpx.Response(404, request=request, text="not found")
    patcher, _client = _patch_client(response)
    with patcher:
        provider = MlxProvider(model="missing-model", max_retries=0)
        result = await provider.complete(CompletionRequest(user_message="u"))
    assert result.error is not None
    assert "404" in result.error


@pytest.mark.asyncio
async def test_health_model_discovery() -> None:
    models = MagicMock(spec=httpx.Response)
    models.status_code = 200
    models.json.return_value = {"data": [{"id": "Qwen3.6-35B-A3B-8bit"}]}
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    dummy = httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}, request=request)
    patcher, _client = _patch_client(dummy, get_response=models)
    with patch("app.ai.health.httpx.AsyncClient", return_value=_client_from_get(models)):
        provider = MlxProvider(model="Qwen3.6-35B-A3B-8bit")
        health = await provider.health()
    assert health.is_local is True
    assert health.reachable is True
    assert health.model_available is True
    assert "thinking" in (health.capabilities or [])


def _client_from_get(response: object) -> AsyncMock:
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=response)
    return mock_client


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("BUGFORGE_MLX_INTEGRATION"),
    reason="Set BUGFORGE_MLX_INTEGRATION=1 to hit a real local MLX server",
)
async def test_optional_live_mlx_server() -> None:
    provider = MlxProvider.from_settings(make_settings())
    available = await provider.is_available()
    assert available is True
    result = await provider.complete(
        CompletionRequest(
            system_prompt='Reply with JSON {"ok": true} only.',
            user_message="ping",
            json_mode=False,
            thinking=False,
        )
    )
    assert result.error is None
    assert result.content
