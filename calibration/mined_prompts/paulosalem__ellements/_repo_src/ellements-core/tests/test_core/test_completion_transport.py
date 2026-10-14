"""Governed transports cover actual attempts, not just high-level LLM calls."""

from __future__ import annotations

import litellm
import pytest
from ellements.core.exceptions import LLMError, MaxToolIterationsError
from ellements.core.llm import CompletionRequest, LLMClient
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str


class Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def no_direct_provider(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("A governed completion must not fall through to the direct provider.")
    monkeypatch.setattr(litellm, "acompletion", forbidden)


def response(content):
    return litellm.ModelResponse(
        id="synthetic-receipt",
        choices=[{"message": {"role": "assistant", "content": content}}],
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    )


@pytest.mark.asyncio
async def test_transport_retains_parsing_and_provider_usage():
    transport = Transport([response('{"answer":"synthetic"}')])
    client = LLMClient(model="openai/gpt-6-astra", completion_transport=transport)
    result = await client.complete_structured("Question", Answer)
    assert result.answer == "synthetic"
    payload = transport.requests[0].as_json()
    assert payload["model"] == "openai/gpt-6-astra"
    schema = payload["response_format"]["json_schema"]
    assert schema["schema"]["additionalProperties"] is False
    assert schema["strict"] is True


@pytest.mark.asyncio
async def test_transport_refusal_and_stream_never_bypass_admission():
    transport = Transport([RuntimeError("Budget exhausted")])
    client = LLMClient(model="openai/gpt-6-astra", completion_transport=transport)
    with pytest.raises(LLMError, match="Budget exhausted"):
        await client.complete("Question")
    with pytest.raises(LLMError, match="no direct-provider fallback"):
        async for _ in client.stream("Question"):
            pytest.fail("Streaming must not start.")
    assert len(transport.requests) == 1


@pytest.mark.asyncio
async def test_each_retry_goes_through_transport():
    transport = Transport([
        litellm.RateLimitError("Synthetic retry", "openai", "gpt-6-astra"),
        response("Done"),
    ])
    client = LLMClient(
        model="openai/gpt-6-astra", completion_transport=transport, retry_base_delay=0
    )
    assert await client.complete("Question") == "Done"
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_every_tool_continuation_is_a_separate_admitted_attempt():
    tool = litellm.ModelResponse(choices=[{"message": {
        "role": "assistant", "content": None, "tool_calls": [{
            "id": "call-synthetic", "type": "function",
            "function": {"name": "read_record", "arguments": "{}"},
        }],
    }}])
    transport = Transport([tool, tool])
    client = LLMClient(model="openai/gpt-6-astra", completion_transport=transport)

    def read_record() -> str:
        """Read a frozen record."""
        return "Frozen evidence"

    with pytest.raises(MaxToolIterationsError):
        await client.complete_with_tools("Question", [read_record], max_iterations=2)
    assert len(transport.requests) == 2
    assert transport.requests[1].messages[-1]["role"] == "tool"


def test_transport_refuses_nonserializable_request_values():
    request = CompletionRequest(model="openai/gpt-6-astra", messages=[], parameters={"x": object()})
    with pytest.raises(ValueError):
        request.as_json()
