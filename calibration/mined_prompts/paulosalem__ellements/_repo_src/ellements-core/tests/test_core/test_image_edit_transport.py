"""Official SDK multipart edits against a synthetic in-process HTTP transport."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from ellements.core.llm.client import LLMClient
from ellements.core.llm.image_transport import (
    ImageGenerationNotStartedError,
    ImageGenerationTransportError,
    OpenAIImageEditTransport,
)
from ellements.core.llm.images import ImageUpload

PARAMETERS = {
    "model": "openai/gpt-image-2", "prompt": "Synthetic reference composition",
    "size": "1536x1024", "quality": "high", "n": 1,
    "output_format": "png", "background": "opaque",
}
UPLOADS = (ImageUpload("first.png", b"first-exact-bytes"), ImageUpload("second.jpg", b"second-exact-bytes"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "timeout", "http", "parse"])
async def test_official_multipart_edit_preserves_order_receipt_and_uncertainty(monkeypatch, failure):
    actual = openai.AsyncOpenAI
    calls = []
    options = []

    def respond(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private", request=request)
        if failure == "http":
            return httpx.Response(400, headers={"x-request-id": "req-edit"}, json={"error": {"message": "private"}})
        if failure == "parse":
            return httpx.Response(200, headers={"x-request-id": "req-edit"}, content=b"not-json")
        return httpx.Response(200, headers={"x-request-id": "req-edit"}, json={
            "created": 1, "data": [{"b64_json": "cG5n"}],
            "usage": {"input_tokens": 12, "input_tokens_details": {"text_tokens": 2, "image_tokens": 10}},
        })

    def factory(**kwargs):
        options.append(kwargs)
        return actual(**kwargs, api_key="synthetic", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr(openai, "AsyncOpenAI", factory)
    transport = OpenAIImageEditTransport(timeout_seconds=30)
    if failure:
        with pytest.raises(ImageGenerationTransportError) as caught:
            await transport.edit(PARAMETERS, UPLOADS)
        assert not isinstance(caught.value, ImageGenerationNotStartedError)
        assert "private" not in str(caught.value) + repr(caught.value.diagnostic)
        assert caught.value.diagnostic["phase"] == ("response_parse" if failure == "parse" else "request")
    else:
        result = await transport.edit(PARAMETERS, UPLOADS)
        assert result.provider_receipt_ids == ("req-edit",)
        assert result.request_parameters["images"] == [item.identity() for item in UPLOADS]
        assert result.request_parameters["images"][0]["sha256"] == hashlib.sha256(UPLOADS[0].content).hexdigest()
        assert result.usage["input_tokens_details"]["image_tokens"] == 10
    assert len(calls) == 1
    assert calls[0].url.path == "/v1/images/edits"
    assert calls[0].content.index(UPLOADS[0].content) < calls[0].content.index(UPLOADS[1].content)
    assert b'name="mask"' not in calls[0].content
    assert options[0]["max_retries"] == 0
    assert options[0]["base_url"] == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_client_edit_transport_is_mandatory_and_not_retried(monkeypatch):
    failure = ImageGenerationTransportError({"phase": "request"})
    transport = SimpleNamespace(edit=AsyncMock(side_effect=failure))
    monkeypatch.setattr("ellements.core.llm.client.litellm.aimage_edit", AsyncMock(side_effect=AssertionError("fallback")))
    monkeypatch.setattr("ellements.core.llm.client.litellm.aimage_generation", AsyncMock(side_effect=AssertionError("fallback")))
    client = LLMClient(model="openai/gpt-image-2", max_retries=0, image_edit_transport=transport)
    with pytest.raises(ImageGenerationTransportError):
        await client.edit_image("Synthetic", [(item.filename, item.content) for item in UPLOADS])
    assert transport.edit.await_count == 1
    with pytest.raises(ValueError, match="zero retries"):
        LLMClient(model="openai/gpt-image-2", image_edit_transport=transport)


@pytest.mark.asyncio
async def test_edit_constructor_failure_is_proven_not_started(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    send = AsyncMock(side_effect=AssertionError("network forbidden"))
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    with pytest.raises(ImageGenerationNotStartedError):
        await OpenAIImageEditTransport(timeout_seconds=30).edit(PARAMETERS, UPLOADS)
    send.assert_not_called()


@pytest.mark.asyncio
# `n` is no longer here: several variants from one prompt is a supported
# capability, priced per variant by the caller. A mask is still refused *as a
# parameter* because it is an upload, passed as the third argument to edit().
@pytest.mark.parametrize("extra", [
    {"mask": b"x"}, {"images": []}, {"extra_body": {}}, {"input_fidelity": "high"},
    {"n": 0}, {"n": 11}, {"n": "2"},
    {"output_format": "gif"}, {"background": "rainbow"},
])
async def test_unsupported_edit_fields_refuse_before_sdk(monkeypatch, extra):
    constructor = AsyncMock(side_effect=AssertionError("SDK forbidden"))
    monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
    with pytest.raises(ValueError):
        await OpenAIImageEditTransport(timeout_seconds=30).edit({**PARAMETERS, **extra}, UPLOADS)
    constructor.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [
    {"n": 2}, {"n": 10},
    {"output_format": "webp"}, {"output_format": "jpeg"},
    {"background": "transparent"}, {"background": "auto"},
    {"background": "transparent", "output_format": "webp"},
])
async def test_the_full_endpoint_surface_reaches_the_sdk(monkeypatch, supported):
    """Variants, encodings and transparency are capabilities, not hazards.

    They were pinned to one PNG opaque image, so a caller could not ask for
    several candidates in one receipt, a cut-out with no background, or a
    smaller encoding — all of which the endpoint supports.
    """
    constructor = AsyncMock(side_effect=AssertionError("SDK reached"))
    monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
    # Reaching the SDK is the proof: the transport wraps the constructor's
    # failure, so a refusal before it would never call the constructor at all.
    with pytest.raises(ImageGenerationNotStartedError):
        await OpenAIImageEditTransport(timeout_seconds=30).edit(
            {**PARAMETERS, **supported}, UPLOADS,
        )
    constructor.assert_called_once()


@pytest.mark.asyncio
async def test_a_transparent_background_needs_an_alpha_encoding(monkeypatch):
    constructor = AsyncMock(side_effect=AssertionError("SDK forbidden"))
    monkeypatch.setattr(openai, "AsyncOpenAI", constructor)
    with pytest.raises(ValueError):
        await OpenAIImageEditTransport(timeout_seconds=30).edit(
            {**PARAMETERS, "background": "transparent", "output_format": "jpeg"}, UPLOADS,
        )
    constructor.assert_not_called()
