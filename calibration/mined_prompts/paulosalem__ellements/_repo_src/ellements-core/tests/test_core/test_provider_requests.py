"""Provider routing and client-local configuration contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import litellm
import pytest
from ellements.core import LLMClient
from ellements.core.llm.requests import (
    merge_request_params,
    prepare_completion_request,
    resolve_model_name,
)
from ellements.core.tools import default_dialect_for_model
from pydantic import BaseModel


@pytest.mark.parametrize("use_responses_api", [False, True])
@pytest.mark.parametrize(
    "model",
    [
        "openrouter/openai/gpt-5",
        "openrouter/openai/gpt-5-mini",
        "openrouter/anthropic/claude-sonnet-4",
        "openrouter/google/gemini-2.5-flash",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "azure/gpt-5",
    ],
)
def test_explicit_provider_routes_are_preserved(
    model: str, use_responses_api: bool
) -> None:
    request = prepare_completion_request(
        model=model,
        temperature=0.5,
        max_tokens=100,
        use_responses_api=use_responses_api,
    )
    assert request.model == model
    assert request.params == {"temperature": 0.5, "max_tokens": 100}
    assert request.logged_max_tokens == 100
    assert default_dialect_for_model(
        model, use_responses_api=use_responses_api
    ).name == "openai_chat"


@pytest.mark.parametrize(
    ("model", "use_responses_api", "expected_model", "expected_params"),
    [
        ("gpt-5-mini", False, "openai/gpt-5-mini", {"max_completion_tokens": 100}),
        (
            "gpt-5-mini",
            True,
            "openai/responses/gpt-5-mini",
            {"max_output_tokens": 100, "reasoning_effort": "medium"},
        ),
        (
            "openai/gpt-5-mini",
            False,
            "openai/gpt-5-mini",
            {"max_completion_tokens": 100},
        ),
        ("openai/gpt-4.1", False, "openai/gpt-4.1", {"temperature": 0.5, "max_tokens": 100}),
    ],
)
def test_direct_openai_routing_is_unchanged(
    model: str,
    use_responses_api: bool,
    expected_model: str,
    expected_params: dict[str, Any],
) -> None:
    request = prepare_completion_request(
        model=model,
        temperature=0.5,
        max_tokens=100,
        use_responses_api=use_responses_api,
    )
    assert request.model == expected_model
    assert request.params == expected_params


def test_model_name_substrings_do_not_select_openai() -> None:
    assert resolve_model_name("custom/not-gpt-5") == ("custom/not-gpt-5", False)


@pytest.mark.parametrize("config_url_key", ["api_base", "base_url"])
@pytest.mark.parametrize("override_url_key", ["api_base", "base_url"])
def test_configuration_merge_normalizes_urls_before_overrides(
    config_url_key: str, override_url_key: str
) -> None:
    config = {config_url_key: "https://default.example/v1", "api_key": "default-key"}
    overrides = {override_url_key: "https://override.example/v1", "api_key": "call-key"}
    assert merge_request_params(config, overrides) == {
        "api_base": "https://override.example/v1",
        "api_key": "call-key",
    }
    assert config[config_url_key] == "https://default.example/v1"
    assert overrides[override_url_key] == "https://override.example/v1"


def test_api_base_wins_within_one_configuration_layer() -> None:
    assert merge_request_params(
        {}, {"api_base": "https://canonical.example", "base_url": "https://other.example"}
    ) == {"api_base": "https://canonical.example"}


def test_mapping_options_are_replaced_not_deep_merged() -> None:
    assert merge_request_params(
        {"extra_body": {"provider": {"sort": "price", "allow_fallbacks": False}}},
        {"extra_body": {"provider": {"sort": "latency"}}},
    ) == {"extra_body": {"provider": {"sort": "latency"}}}


class Answer(BaseModel):
    answer: int


@pytest.mark.parametrize(
    "method",
    [
        "complete",
        "complete_structured",
        "stream",
        "complete_with_tools",
        "loglikelihood",
        "generate_image",
        "edit_image",
    ],
)
@pytest.mark.parametrize("override", [False, True])
async def test_every_call_uses_local_provider_configuration(
    monkeypatch: pytest.MonkeyPatch, method: str, override: bool
) -> None:
    monkeypatch.setattr(litellm, "api_key", "untouched-global-key")
    monkeypatch.setattr(litellm, "api_base", "https://untouched.example/v1")
    monkeypatch.setattr(litellm, "supports_response_schema", lambda **kwargs: True)
    model = "openrouter/openai/gpt-5-mini"
    headers = {"HTTP-Referer": "https://example.com"}
    body = {"provider": {"sort": "price", "require_parameters": True}}
    client = LLMClient(
        model=model,
        api_key="router-key",
        base_url="https://router.example/v1",
        extra_headers=headers,
        extra_body=body,
        use_responses_api=True,
    )
    other = LLMClient(
        model="openai/gpt-4.1-mini",
        api_key="direct-key",
        api_base="https://direct.example/v1",
    )
    assert litellm.api_key == "untouched-global-key"
    assert litellm.api_base == "https://untouched.example/v1"

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"answer":42}', tool_calls=None),
                logprobs={"token_logprobs": [-0.5]},
            )
        ],
        usage=None,
    )

    async def chunks() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="done"))]
        )

    provider = AsyncMock(
        return_value=(
            chunks()
            if method == "stream"
            else SimpleNamespace(created=1, data=[{"b64_json": "image"}], usage=None)
            if method in {"generate_image", "edit_image"}
            else response
        )
    )
    provider_name = {
        "generate_image": "aimage_generation",
        "edit_image": "aimage_edit",
    }.get(method, "acompletion")
    monkeypatch.setattr(litellm, provider_name, provider)
    options: dict[str, Any] = (
        {"api_key": "call-key", "api_base": "https://override.example/v1"}
        if override
        else {}
    )

    def ping() -> str:
        """Return a ping response."""
        return "pong"

    if method == "complete":
        assert await client.complete("Question", **options) == '{"answer":42}'
    elif method == "complete_structured":
        assert await client.complete_structured("Question", Answer, **options) == Answer(
            answer=42
        )
    elif method == "stream":
        assert [chunk async for chunk in client.stream("Question", **options)] == ["done"]
    elif method == "complete_with_tools":
        await client.complete_with_tools("Question", tools=[ping], **options)
        assert provider.await_args.kwargs["tools"][0]["type"] == "function"
    elif method == "loglikelihood":
        assert await client.loglikelihood("Question", "Answer", **options) == (-0.5, False)
    elif method == "generate_image":
        await client.generate_image("A blue circle", **options)
    else:
        await client.edit_image("A blue circle", [b"reference"], **options)

    params = provider.await_args.kwargs
    assert params["model"] == model
    assert params["api_key"] == ("call-key" if override else "router-key")
    assert params["api_base"] == (
        "https://override.example/v1" if override else "https://router.example/v1"
    )
    assert "base_url" not in params
    assert params["extra_headers"] == headers
    assert params["extra_body"] == body
    assert client.config["api_key"] == "router-key"

    direct_provider = AsyncMock(return_value=response)
    monkeypatch.setattr(litellm, "acompletion", direct_provider)
    await other.complete("Question")
    assert direct_provider.await_args.kwargs["api_key"] == "direct-key"
    assert direct_provider.await_args.kwargs["api_base"] == "https://direct.example/v1"
    assert "extra_body" not in direct_provider.await_args.kwargs


async def test_call_model_override_preserves_openrouter_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
            usage=None,
        )
    )
    monkeypatch.setattr(litellm, "acompletion", provider)
    client = LLMClient(model="openai/gpt-4.1-mini", use_responses_api=True)
    await client.complete(
        "Question", model="openrouter/openai/gpt-5-mini", api_key="router-key"
    )
    assert provider.await_args.kwargs["model"] == "openrouter/openai/gpt-5-mini"
    assert provider.await_args.kwargs["api_key"] == "router-key"


async def test_client_credentials_are_not_added_to_request_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = SimpleNamespace(
        on_request=AsyncMock(), on_response=AsyncMock(), on_error=AsyncMock()
    )
    monkeypatch.setattr(
        litellm,
        "acompletion",
        AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
                usage=None,
            )
        ),
    )
    client = LLMClient(
        model="openrouter/openai/gpt-4.1-mini",
        api_key="private-test-key",
        extra_headers={"Authorization": "private-test-header"},
        observers=[observer],
    )
    await client.complete("Question")
    assert observer.on_request.await_args.args[0].extra_params == {}
