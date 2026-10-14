from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ellements.core.llm.client import LLMClient
from ellements.core.llm.image_transport import OpenAIImageGenerationTransport


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["req-synthetic", None, "invalid\nheader"])
async def test_receipt_preserving_shared_client(monkeypatch, identifier):
    import openai

    raw = SimpleNamespace(
        headers={"x-request-id": identifier},
        status_code=200,
        parse=Mock(return_value=SimpleNamespace(
            created=1, data=[{"b64_json": "cG5n"}],
            usage={"input_tokens": 4, "output_tokens": 2},
        )),
    )
    generate = AsyncMock(return_value=raw)
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.images = SimpleNamespace(with_raw_response=SimpleNamespace(generate=generate))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    client = LLMClient(
        model="openai/gpt-image-2", max_retries=0,
        image_generation_transport=OpenAIImageGenerationTransport(timeout_seconds=900),
    )
    response = await client.generate_image(
        "Synthetic text only", size="1536x1024", quality="medium",
        output_format="png", background="opaque",
    )
    assert response.provider_receipt_ids == (("req-synthetic",) if identifier == "req-synthetic" else ())
    assert response.usage["input_tokens"] == 4
    assert response.request_parameters == generate.call_args.kwargs
    assert generate.await_count == 1
    assert generate.call_args.kwargs["model"] == "gpt-image-2"
    assert "image" not in generate.call_args.kwargs
    assert observed["max_retries"] == 0
    assert observed["base_url"] == "https://api.openai.com/v1"
    assert observed["timeout"] == 900


@pytest.mark.asyncio
async def test_real_async_sdk_raw_response_is_parsed_synchronously(monkeypatch):
    import httpx
    import openai

    actual_client = openai.AsyncOpenAI
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, headers={"x-request-id": "req-isolated-sdk"},
            json={"created": 1, "data": [{"b64_json": "cG5n"}],
                  "usage": {"input_tokens": 4, "output_tokens": 2}},
        )

    def client(**kwargs):
        return actual_client(
            **kwargs, api_key="synthetic-not-a-secret",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    response = await OpenAIImageGenerationTransport(timeout_seconds=900).generate({
        "model": "openai/gpt-image-2", "prompt": "Synthetic",
        "size": "1536x1024", "quality": "medium", "n": 1,
        "output_format": "png", "background": "opaque",
    })
    assert len(requests) == 1
    assert response.provider_receipt_ids == ("req-isolated-sdk",)
    assert response.data[0].b64_json == "cG5n"
    assert response.usage["input_tokens"] == 4


@pytest.mark.asyncio
async def test_no_retry_or_direct_provider_fallback():
    transport = SimpleNamespace(generate=AsyncMock(side_effect=RuntimeError("synthetic failure")))
    client = LLMClient(model="openai/gpt-image-2", max_retries=0, image_generation_transport=transport)
    with pytest.raises(Exception, match="synthetic failure"):
        await client.generate_image("Synthetic")
    assert transport.generate.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "http_error", "parse_error"])
async def test_sdk_failures_preserve_only_non_content_diagnostics(monkeypatch, failure):
    import httpx
    import openai
    from ellements.core.llm.image_transport import (
        ImageGenerationNotStartedError,
        ImageGenerationTransportError,
    )

    actual_client = openai.AsyncOpenAI
    calls = []

    def respond(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private provider content", request=request)
        if failure == "http_error":
            return httpx.Response(
                400, headers={"x-request-id": "req-isolated-error"},
                json={"error": {"message": "private provider content", "type": "invalid_request_error"}},
            )
        return httpx.Response(
            200, headers={"x-request-id": "req-isolated-error", "content-type": "application/json"},
            content=b"private provider content",
        )

    def client(**kwargs):
        return actual_client(
            **kwargs, api_key="synthetic-not-a-secret",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    transport = OpenAIImageGenerationTransport(timeout_seconds=900)
    with pytest.raises(ImageGenerationTransportError) as caught:
        await transport.generate({
            "model": "openai/gpt-image-2", "prompt": "Synthetic",
            "size": "1536x1024", "quality": "medium", "n": 1,
            "output_format": "png", "background": "opaque",
        })
    diagnostic = caught.value.diagnostic
    assert not isinstance(caught.value, ImageGenerationNotStartedError)
    assert len(calls) == 1
    assert "private provider" not in str(caught.value) + repr(diagnostic)
    assert diagnostic["timeout_seconds"] == 900
    assert diagnostic["phase"] == ("response_parse" if failure == "parse_error" else "request")
    if failure == "timeout":
        assert diagnostic["exception_type"] == "APITimeoutError"
        assert diagnostic["cause_type"] == "ReadTimeout"
        assert diagnostic["provider_receipt_ids"] == []
    else:
        assert diagnostic["provider_receipt_ids"] == ["req-isolated-error"]
        assert diagnostic["http_status"] == (400 if failure == "http_error" else 200)


@pytest.mark.asyncio
async def test_real_sdk_constructor_without_credentials_never_hands_off(monkeypatch, caplog):
    import httpx
    from ellements.core.llm.image_transport import ImageGenerationNotStartedError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    send = AsyncMock(side_effect=AssertionError("No provider request is permitted"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    with pytest.raises(ImageGenerationNotStartedError) as caught:
        await OpenAIImageGenerationTransport(timeout_seconds=900).generate({
            "model": "openai/gpt-image-2", "prompt": "private-synthetic-prompt",
            "size": "1536x1024", "quality": "medium", "n": 1,
            "output_format": "png", "background": "opaque",
        })
    assert caught.value.diagnostic["phase"] == "client_initialization"
    assert caught.value.diagnostic["exception_type"] == "OpenAIError"
    assert caught.value.diagnostic["provider_receipt_ids"] == []
    assert caught.value.diagnostic["http_status"] is None
    send.assert_not_called()
    assert "private-synthetic-prompt" not in str(caught.value) + repr(caught.value.diagnostic) + caplog.text


@pytest.mark.asyncio
async def test_constructor_exception_message_is_never_retained(monkeypatch, caplog):
    import openai
    from ellements.core.llm.image_transport import ImageGenerationNotStartedError

    secret = "synthetic-secret-value-do-not-retain"
    constructor = Mock(side_effect=openai.OpenAIError(secret))
    monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
    with pytest.raises(ImageGenerationNotStartedError) as caught:
        await OpenAIImageGenerationTransport(timeout_seconds=900).generate({
            "model": "openai/gpt-image-2", "prompt": "Synthetic",
            "size": "1536x1024", "quality": "medium", "n": 1,
            "output_format": "png", "background": "opaque",
        })
    assert secret not in str(caught.value) + repr(caught.value.diagnostic) + caplog.text
    assert constructor.call_count == 1


@pytest.mark.parametrize("credential", [None, "", " \t"])
def test_preflight_only_requires_explicit_nonempty_credentials(monkeypatch, credential):
    if credential is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", credential)
    with pytest.raises(ValueError, match="credentials are unavailable"):
        OpenAIImageGenerationTransport.preflight()
