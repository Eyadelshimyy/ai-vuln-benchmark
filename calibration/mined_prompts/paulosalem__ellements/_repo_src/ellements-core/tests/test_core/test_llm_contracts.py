"""Contract tests for LLMClient: required-model, retry classification, errors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from ellements.core import (
    LLMClient,
    LLMError,
    LocalCacheConfig,
    LogprobsUnsupportedError,
    StructuredOutputUnsupportedError,
)
from ellements.core.llm.client import (
    _classify_retryable,
    _full_jitter_backoff,
    _usage_dict,
    _with_cache_control,
)
from pydantic import BaseModel


def _mock_completion(content: str = "ok") -> Any:
    message = MagicMock()
    message.content = content
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = None
    return response


def test_usage_includes_provider_reported_response_cost():
    response = _mock_completion()
    response.usage = {"prompt_tokens": 10, "completion_tokens": 3}
    response._hidden_params = {"response_cost": 0.0012}

    assert _usage_dict(response) == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "response_cost": 0.0012,
    }


# ── Required-model construction ────────────────────────────────────


def test_model_required_at_construction():
    with pytest.raises(ValueError, match="model"):
        LLMClient(model="")


def test_default_retry_settings():
    client = LLMClient(model="openai/gpt-4o-mini")
    assert client.max_retries == 3
    assert client.retry_base_delay > 0
    assert client.retry_max_delay > 0


def test_disabled_cache_control_blocks_reads_and_writes() -> None:
    assert _with_cache_control({}, False) == {
        "caching": False,
        "cache": {
            "no-cache": True,
            "no-store": True,
        },
    }


def test_explicit_cache_opt_in_is_preserved() -> None:
    assert _with_cache_control({"caching": True}, False) == {"caching": True}


# ── Retry classification ────────────────────────────────────────────


def test_retryable_classifies_rate_limit():
    exc = litellm.RateLimitError(
        message="429",
        llm_provider="openai",
        model="gpt-4o-mini",
    )
    assert _classify_retryable(exc) is True


def test_retryable_classifies_timeout():
    exc = litellm.Timeout(
        message="timeout",
        model="gpt-4o-mini",
        llm_provider="openai",
    )
    assert _classify_retryable(exc) is True


def test_non_retryable_for_authentication_error():
    exc = litellm.AuthenticationError(
        message="bad key",
        llm_provider="openai",
        model="gpt-4o-mini",
    )
    assert _classify_retryable(exc) is False


def test_non_retryable_for_bad_request():
    exc = litellm.BadRequestError(
        message="bad request",
        llm_provider="openai",
        model="gpt-4o-mini",
    )
    assert _classify_retryable(exc) is False


def test_full_jitter_backoff_in_bounds():
    for attempt in range(5):
        delay = _full_jitter_backoff(attempt, base=0.5, cap=30.0)
        assert 0.0 <= delay <= min(30.0, 0.5 * (2**attempt))


def test_full_jitter_backoff_varies():
    samples = {round(_full_jitter_backoff(3, base=0.5, cap=30.0), 6) for _ in range(50)}
    assert len(samples) > 5, "Jitter should produce a spread of delays"


@pytest.mark.asyncio
async def test_image_edit_recreates_upload_streams_for_retry():
    client = LLMClient(
        model="openai/gpt-image-1",
        retry_base_delay=0.0,
        retry_max_delay=0.0,
    )
    payload = b"reference image bytes"
    attempts: list[bytes] = []

    async def fake_image_edit(*, image: list[Any], **kwargs: Any) -> Any:
        del kwargs
        attempts.append(image[0].read())
        if len(attempts) == 1:
            raise litellm.RateLimitError(
                message="retry",
                llm_provider="openai",
                model="gpt-image-1",
            )
        return SimpleNamespace(
            created=1,
            data=[{"b64_json": "result"}],
            model="gpt-image-1",
            usage=None,
        )

    with patch("ellements.core.llm.client.litellm.aimage_edit", fake_image_edit):
        result = await client.edit_image(
            "Preserve this reference.",
            [("reference.png", payload)],
        )

    assert attempts == [payload, payload]
    assert result.data[0].b64_json == "result"


# ── Auth/BadRequest are not retried by complete() ───────────────────


@pytest.mark.asyncio
async def test_complete_does_not_retry_authentication_error():
    client = LLMClient(model="openai/gpt-4o-mini")
    auth_err = litellm.AuthenticationError(
        message="bad key", llm_provider="openai", model="gpt-4o-mini"
    )
    mock = AsyncMock(side_effect=auth_err)
    with (
        patch("ellements.core.llm.client.litellm.acompletion", mock),
        pytest.raises(LLMError),
    ):
        await client.complete("hi")
    assert mock.await_count == 1


@pytest.mark.asyncio
async def test_complete_retries_rate_limit_then_succeeds():
    client = LLMClient(
        model="openai/gpt-4o-mini",
        max_retries=2,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
    )
    rate_limit = litellm.RateLimitError(
        message="429", llm_provider="openai", model="gpt-4o-mini"
    )
    mock = AsyncMock(side_effect=[rate_limit, _mock_completion("done")])
    with patch("ellements.core.llm.client.litellm.acompletion", mock):
        result = await client.complete("hello")
    assert result == "done"
    assert mock.await_count == 2


# ── Structured-output unsupported error ─────────────────────────────


class _SmallModel(BaseModel):
    x: int


@pytest.mark.asyncio
async def test_complete_structured_raises_when_unsupported():
    client = LLMClient(model="openai/gpt-4o-mini")
    with (
        patch(
            "ellements.core.llm.structured.litellm.supports_response_schema",
            return_value=False,
        ),
        pytest.raises(StructuredOutputUnsupportedError),
    ):
        await client.complete_structured("hi", _SmallModel)


# ── Loglikelihood unsupported error ─────────────────────────────────


@pytest.mark.asyncio
async def test_loglikelihood_raises_when_logprobs_missing():
    client = LLMClient(model="openai/gpt-4o-mini")
    bad_response = MagicMock()
    bad_response.choices = [MagicMock(logprobs=None, message=MagicMock(content=""))]
    mock = AsyncMock(return_value=bad_response)
    with (
        patch("ellements.core.llm.client.litellm.acompletion", mock),
        pytest.raises(LogprobsUnsupportedError),
    ):
        await client.loglikelihood("ctx", "cont")


# ── Observers receive events on every method ────────────────────────


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def on_request(self, event):  # noqa: ANN001
        self.events.append(("request", event.method))

    async def on_response(self, event):  # noqa: ANN001
        self.events.append(("response", event.method))

    async def on_error(self, event):  # noqa: ANN001
        self.events.append(("error", event.method))


class MetadataObserver:
    def __init__(self) -> None:
        self.responses: list[Any] = []

    async def on_request(self, event: Any) -> None:
        del event

    async def on_response(self, event: Any) -> None:
        self.responses.append(event)

    async def on_error(self, event: Any) -> None:
        del event


@pytest.mark.asyncio
async def test_observer_fires_request_and_response_on_complete():
    observer = RecordingObserver()
    client = LLMClient(model="openai/gpt-4o-mini", observers=[observer])
    mock = AsyncMock(return_value=_mock_completion("ok"))
    with patch("ellements.core.llm.client.litellm.acompletion", mock):
        await client.complete("hello")
    assert observer.events == [("request", "complete"), ("response", "complete")]


@pytest.mark.asyncio
async def test_observer_fires_error_on_failure():
    observer = RecordingObserver()
    client = LLMClient(
        model="openai/gpt-4o-mini",
        observers=[observer],
        max_retries=0,
    )
    auth_err = litellm.AuthenticationError(
        message="bad key", llm_provider="openai", model="gpt-4o-mini"
    )
    mock = AsyncMock(side_effect=auth_err)
    with (
        patch("ellements.core.llm.client.litellm.acompletion", mock),
        pytest.raises(LLMError),
    ):
        await client.complete("hi")
    assert ("request", "complete") in observer.events
    assert ("error", "complete") in observer.events


@pytest.mark.asyncio
async def test_completion_forwards_cache_control_and_reports_local_hit() -> None:
    observer = MetadataObserver()
    client = LLMClient(model="openai/gpt-4o-mini", observers=[observer])
    response = _mock_completion("cached")
    response._hidden_params = {"cache_hit": True, "response_cost": 1.5}
    mock = AsyncMock(return_value=response)

    with patch("ellements.core.llm.client.litellm.acompletion", mock):
        result = await client.complete("hello", caching=True)

    assert result == "cached"
    assert mock.await_args.kwargs["caching"] is True
    assert observer.responses[0].usage is None
    assert observer.responses[0].metadata == {
        "local_cache": {
            "hit": True,
            "hits": 1,
            "backend": "litellm-disk",
        }
    }


@pytest.mark.asyncio
async def test_image_generation_cache_persists_and_reports_hit(tmp_path: Path) -> None:
    first_observer = MetadataObserver()
    second_observer = MetadataObserver()
    cache = LocalCacheConfig(
        tmp_path,
        cache_responses=False,
        cache_images=True,
    )
    first_client = LLMClient(
        model="openai/gpt-image-1",
        observers=[first_observer],
        local_cache=cache,
    )
    second_client = LLMClient(
        model="openai/gpt-image-1",
        observers=[second_observer],
        local_cache=cache,
    )
    provider = AsyncMock(
        return_value=SimpleNamespace(
            created=1,
            data=[{"b64_json": "cached-image"}],
            model="gpt-image-1",
            usage=None,
        )
    )

    with patch("ellements.core.llm.client.litellm.aimage_generation", provider):
        first = await first_client.generate_image("A precise blue circle.")
        second = await second_client.generate_image("A precise blue circle.")

    assert first == second
    assert provider.await_count == 1
    assert first_observer.responses[0].metadata == {}
    assert second_observer.responses[0].metadata["local_cache"]["hit"] is True
    assert (
        second_observer.responses[0].metadata["local_cache"]["backend"]
        == "ellements-image-disk"
    )


@pytest.mark.asyncio
async def test_image_edit_cache_keys_include_reference_content(tmp_path: Path) -> None:
    cache = LocalCacheConfig(
        tmp_path,
        cache_responses=False,
        cache_images=True,
    )
    client = LLMClient(model="openai/gpt-image-1", local_cache=cache)
    provider = AsyncMock(
        side_effect=[
            SimpleNamespace(
                created=1,
                data=[{"b64_json": "first"}],
                model="gpt-image-1",
                usage=None,
            ),
            SimpleNamespace(
                created=2,
                data=[{"b64_json": "second"}],
                model="gpt-image-1",
                usage=None,
            ),
        ]
    )

    with patch("ellements.core.llm.client.litellm.aimage_edit", provider):
        first = await client.edit_image("Preserve it.", [("ref.png", b"one")])
        repeated = await client.edit_image("Preserve it.", [("ref.png", b"one")])
        changed = await client.edit_image("Preserve it.", [("ref.png", b"two")])

    assert first == repeated
    assert changed.data[0].b64_json == "second"
    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_image_cache_does_not_persist_expiring_urls(tmp_path: Path) -> None:
    cache = LocalCacheConfig(
        tmp_path,
        cache_responses=False,
        cache_images=True,
    )
    client = LLMClient(model="openai/dall-e-3", local_cache=cache)
    provider = AsyncMock(
        side_effect=[
            SimpleNamespace(
                created=1,
                data=[{"url": "https://provider.example/first"}],
                model="dall-e-3",
                usage=None,
            ),
            SimpleNamespace(
                created=2,
                data=[{"url": "https://provider.example/second"}],
                model="dall-e-3",
                usage=None,
            ),
        ]
    )

    with patch("ellements.core.llm.client.litellm.aimage_generation", provider):
        first = await client.generate_image("A transient URL.")
        second = await client.generate_image("A transient URL.")

    assert first.data[0].url != second.data[0].url
    assert provider.await_count == 2
