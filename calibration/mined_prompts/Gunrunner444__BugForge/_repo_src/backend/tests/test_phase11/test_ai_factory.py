"""Tests for the AI provider factory and health check (Phase 11)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ai.factory import check_ai_health, create_provider
from app.core.config import Settings


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "ai_provider": "mock",
        "ai_model": "test-model",
        "ai_api_key": "",
        "ai_base_url": "",
        "discovery_mode": "disabled",
        "discovery_languages": "",
        "discovery_excluded_topics": "",
        "discovery_excluded_owners": "",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# create_provider
# ---------------------------------------------------------------------------


def test_create_mock_provider() -> None:
    from app.ai.mock_provider import MockLLMProvider

    provider = create_provider(make_settings(ai_provider="mock"))
    assert isinstance(provider, MockLLMProvider)


def test_create_openai_provider() -> None:
    from app.ai.openai_provider import OpenAIProvider

    provider = create_provider(
        make_settings(ai_provider="openai", ai_api_key="sk-test", ai_model="gpt-4")
    )
    assert isinstance(provider, OpenAIProvider)
    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-4"


def test_create_ollama_provider_uses_local_adapter() -> None:
    from app.ai.local_provider import LocalAIProvider

    provider = create_provider(make_settings(ai_provider="ollama", ai_model="llama3"))
    assert isinstance(provider, LocalAIProvider)
    assert provider.model_name == "llama3"


def test_create_openai_compatible_provider() -> None:
    from app.ai.local_provider import LocalAIProvider

    provider = create_provider(
        make_settings(
            ai_provider="openai_compatible",
            ai_base_url="http://localhost:1234/v1",
            ai_model="my-local-model",
        )
    )
    assert isinstance(provider, LocalAIProvider)


def test_unknown_provider_raises() -> None:
    """Settings itself rejects unknown providers via the field validator."""
    from pydantic import ValidationError

    with pytest.raises((ValueError, ValidationError)):
        make_settings(ai_provider="nonexistent")


def test_is_local_ai_mock() -> None:
    s = make_settings(ai_provider="mock")
    assert s.is_local_ai() is True


def test_is_local_ai_local() -> None:
    s = make_settings(ai_provider="local")
    assert s.is_local_ai() is True


def test_is_local_ai_ollama() -> None:
    s = make_settings(ai_provider="ollama")
    assert s.is_local_ai() is True


def test_is_local_ai_mlx() -> None:
    s = make_settings(ai_provider="mlx")
    assert s.is_local_ai() is True


def test_is_local_ai_openai() -> None:
    s = make_settings(ai_provider="openai", ai_api_key="sk-x")
    assert s.is_local_ai() is False


def test_is_local_ai_anthropic() -> None:
    s = make_settings(ai_provider="anthropic", ai_api_key="ant-x")
    assert s.is_local_ai() is False


# ---------------------------------------------------------------------------
# check_ai_health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_mock() -> None:
    health = await check_ai_health(make_settings(ai_provider="mock"))
    assert health.provider == "mock"
    assert health.reachable is True
    assert health.configured is True
    assert health.model_available is True
    assert health.is_local is True
    assert health.error is None


@pytest.mark.asyncio
async def test_health_check_ollama_not_reachable() -> None:
    """When Ollama is not running, health check returns reachable=False."""
    import httpx

    with patch("app.ai.health.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        mock_client_cls.return_value = mock_client

        health = await check_ai_health(
            make_settings(
                ai_provider="ollama",
                ai_model="llama3",
                ai_base_url="http://localhost:11434/v1",
            )
        )

    assert health.provider == "ollama"
    assert health.is_local is True
    assert health.reachable is False
    assert health.error is not None


@pytest.mark.asyncio
async def test_health_check_ollama_reachable_model_found() -> None:
    """Simulate Ollama responding with a matching model."""
    import httpx

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"id": "llama3:latest"}, {"id": "qwen2.5-coder:7b"}]
    }
    mock_response.headers = {}

    with patch("app.ai.health.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        health = await check_ai_health(
            make_settings(
                ai_provider="ollama",
                ai_model="llama3",
                ai_base_url="http://localhost:11434/v1",
            )
        )

    assert health.reachable is True
    assert health.model_available is True


@pytest.mark.asyncio
async def test_health_check_ollama_model_not_found() -> None:
    """Simulate Ollama running but without the configured model."""
    import httpx

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "other-model"}]}
    mock_response.headers = {}

    with patch("app.ai.health.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        health = await check_ai_health(
            make_settings(
                ai_provider="ollama",
                ai_model="llama3",
                ai_base_url="http://localhost:11434/v1",
            )
        )

    assert health.reachable is True
    assert health.model_available is False


@pytest.mark.asyncio
async def test_health_check_anthropic_no_key() -> None:
    health = await check_ai_health(make_settings(ai_provider="anthropic", ai_api_key=""))
    assert health.provider == "anthropic"
    assert health.configured is False
    assert health.error is not None


@pytest.mark.asyncio
async def test_health_check_anthropic_with_key() -> None:
    health = await check_ai_health(make_settings(ai_provider="anthropic", ai_api_key="ant-key"))
    assert health.configured is True
    assert health.is_local is False
