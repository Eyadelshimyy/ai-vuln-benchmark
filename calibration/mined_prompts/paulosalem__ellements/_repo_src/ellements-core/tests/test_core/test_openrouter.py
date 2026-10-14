"""Exercise the real LiteLLM OpenRouter adapter with an offline HTTP boundary."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock

import httpx
import litellm
import pytest
from ellements.core import LLMClient, LLMError
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from pydantic import BaseModel

MODEL = "openrouter/openai/gpt-4.1-mini"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


@pytest.fixture
async def http_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[[list[httpx.Response]], list[httpx.Request]]]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-router-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-direct-env-key")
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.delenv("OR_SITE_URL", raising=False)
    monkeypatch.delenv("OR_APP_NAME", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "headers", None)

    def stub(responses: list[httpx.Response]) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        async def send(
            self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
        ) -> httpx.Response:
            requests.append(request)
            assert responses, f"Unexpected HTTP request: {request.method} {request.url}"
            response = responses.pop(0)
            response.request = request
            return response

        def unexpected_send(
            self: httpx.Client, request: httpx.Request, **kwargs: Any
        ) -> httpx.Response:
            raise AssertionError(f"Unexpected synchronous HTTP request: {request.url}")

        monkeypatch.setattr(httpx.AsyncClient, "send", send)
        monkeypatch.setattr(httpx.Client, "send", unexpected_send)
        return requests

    yield stub
    # Drain LiteLLM callbacks before pytest closes this test's event loop.
    await GLOBAL_LOGGING_WORKER.flush()
    await GLOBAL_LOGGING_WORKER.stop()


def completion(
    content: str | None = "done", *, tool_calls: list[dict[str, Any]] | None = None
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return httpx.Response(
        200,
        json={
            "id": "gen-test",
            "object": "chat.completion",
            "created": 1,
            "model": "openai/gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    )


@pytest.mark.parametrize("explicit_key", [False, True])
@pytest.mark.parametrize(
    "model", [MODEL, "openrouter/openai/gpt-5-mini", "openrouter/anthropic/claude-sonnet-4"]
)
async def test_openrouter_endpoint_authentication_and_options(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
    explicit_key: bool,
    model: str,
) -> None:
    requests = http_boundary([completion()])
    client = LLMClient(
        model=model,
        use_responses_api=True,
        extra_headers={
            "HTTP-Referer": "https://example.com",
            "X-OpenRouter-Title": "Ellements example",
        },
        extra_body={"provider": {"sort": "price", "require_parameters": True}},
        **({"api_key": "test-explicit-key"} if explicit_key else {}),
    )
    assert await client.complete("A short explanation", max_tokens=64) == "done"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == ENDPOINT
    assert request.headers["authorization"] == (
        "Bearer test-explicit-key" if explicit_key else "Bearer test-router-env-key"
    )
    assert request.headers["http-referer"] == "https://example.com"
    assert request.headers["x-openrouter-title"] == "Ellements example"
    payload = json.loads(request.content)
    assert payload["model"] == model.removeprefix("openrouter/")
    assert payload["messages"] == [{"role": "user", "content": "A short explanation"}]
    assert payload["max_tokens"] == 64
    assert payload["provider"] == {"sort": "price", "require_parameters": True}
    assert "max_output_tokens" not in payload
    assert client.config["extra_body"]["provider"]["sort"] == "price"


async def test_openrouter_custom_endpoint_and_request_overrides(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
) -> None:
    requests = http_boundary([completion(), completion(), completion()])
    client = LLMClient(
        model=MODEL, api_key="client-key", api_base="https://router.example/api/v1"
    )
    await client.complete("First")
    await client.complete(
        "Second",
        api_key="request-key",
        base_url="https://override.example/api/v1",
    )
    await client.complete("Third")
    assert [str(request.url) for request in requests] == [
        "https://router.example/api/v1/chat/completions",
        "https://override.example/api/v1/chat/completions",
        "https://router.example/api/v1/chat/completions",
    ]
    assert [request.headers["authorization"] for request in requests] == [
        "Bearer client-key",
        "Bearer request-key",
        "Bearer client-key",
    ]


async def test_openrouter_environment_configuration_survives_direct_client(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://router.example/api/v1")
    requests = http_boundary([completion(), completion(), completion()])
    router = LLMClient(model=MODEL)
    direct = LLMClient(
        model="openai/gpt-4.1-mini",
        api_key="direct-key",
        base_url="https://direct.example/v1",
    )
    await router.complete("First")
    await direct.complete("Second")
    await router.complete("Third")
    assert [str(request.url) for request in requests] == [
        "https://router.example/api/v1/chat/completions",
        "https://direct.example/v1/chat/completions",
        "https://router.example/api/v1/chat/completions",
    ]
    assert [request.headers["authorization"] for request in requests] == [
        "Bearer test-router-env-key",
        "Bearer direct-key",
        "Bearer test-router-env-key",
    ]


class Answer(BaseModel):
    answer: int


async def test_openrouter_native_structured_output(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
) -> None:
    requests = http_boundary([completion('{"answer":42}')])
    client = LLMClient(model=MODEL)
    assert await client.complete_structured("Return the answer", Answer) == Answer(
        answer=42
    )
    payload = json.loads(requests[0].content)
    assert str(requests[0].url) == ENDPOINT
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["answer"]["type"] == "integer"
    assert schema["required"] == ["answer"]


async def test_openrouter_streaming(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
) -> None:
    events = [
        {
            "id": "gen-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "openai/gpt-4.1-mini",
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": finish}
            ],
        }
        for text, finish in [("Hello", None), (" world", None), ("", "stop")]
    ]
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    requests = http_boundary(
        [
            httpx.Response(
                200,
                content=stream + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        ]
    )
    client = LLMClient(model=MODEL, extra_headers={"X-OpenRouter-Title": "Stream"})
    assert "".join([text async for text in client.stream("Greeting")]) == "Hello world"
    assert str(requests[0].url) == ENDPOINT
    assert requests[0].headers["x-openrouter-title"] == "Stream"
    assert json.loads(requests[0].content)["stream"] is True


async def test_openrouter_executes_tool_and_returns_result(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
) -> None:
    requests = http_boundary(
        [
            completion(
                None,
                tool_calls=[
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {"name": "double", "arguments": '{"value":21}'},
                    }
                ],
            ),
            completion("42"),
        ]
    )

    def double(value: int) -> int:
        """Double an integer."""
        return value * 2

    client = LLMClient(
        model=MODEL,
        use_responses_api=True,
        extra_body={"provider": {"require_parameters": True}},
    )
    result = await client.complete_with_tools("Double 21", tools=[double])
    assert result.content == "42"
    assert result.tool_calls[0].result == "42"
    assert len(requests) == 2
    for request in requests:
        assert str(request.url) == ENDPOINT
        payload = json.loads(request.content)
        assert payload["tools"][0]["type"] == "function"
        assert payload["provider"] == {"require_parameters": True}
    assert json.loads(requests[1].content)["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-test",
        "content": "42",
    }


async def test_openrouter_authentication_error_is_observed_without_retry(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
) -> None:
    requests = http_boundary(
        [httpx.Response(401, json={"error": {"message": "Invalid API key", "code": 401}})]
    )
    observer = AsyncMock()
    client = LLMClient(model=MODEL, observers=[observer])
    with pytest.raises(LLMError) as error:
        await client.complete("Question")
    assert isinstance(error.value.__cause__, litellm.AuthenticationError)
    assert len(requests) == 1
    observer.on_request.assert_awaited_once()
    observer.on_error.assert_awaited_once()
    observer.on_response.assert_not_awaited()


async def test_openrouter_rate_limit_uses_client_retry(
    http_boundary: Callable[[list[httpx.Response]], list[httpx.Request]],
) -> None:
    requests = http_boundary(
        [
            httpx.Response(
                429, json={"error": {"message": "Rate limited", "code": 429}}
            ),
            completion(),
        ]
    )
    observer = AsyncMock()
    client = LLMClient(
        model=MODEL,
        observers=[observer],
        max_retries=1,
        retry_base_delay=0,
        retry_max_delay=0,
    )
    assert await client.complete("Question") == "done"
    assert len(requests) == 2
    assert all(str(request.url) == ENDPOINT for request in requests)
    observer.on_request.assert_awaited_once()
    observer.on_response.assert_awaited_once()
    observer.on_error.assert_not_awaited()
