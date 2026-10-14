"""Restore an LLM provider from persisted session configuration.

API credentials are never persisted and never reconstructed from snapshots.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from app.ai.mock_provider import MockLLMProvider
from app.ai.provider import LLMProvider
from app.core.config import get_settings


def snapshot_provider(provider: LLMProvider, *, thinking: bool) -> dict[str, Any]:
    caps = provider.capabilities()
    base_url = getattr(provider, "_base_url", None) or getattr(provider, "base_url", None)
    return {
        "provider_id": provider.provider_name,
        "provider_backend": provider.provider_name,
        "model": provider.model_name,
        "thinking": thinking,
        "max_output": getattr(provider, "_max_tokens", None) or caps.max_output_tokens,
        "temperature": getattr(provider, "_temperature", None),
        "context_limit": caps.max_context_tokens,
        "base_url": _safe_base_url(str(base_url) if base_url else ""),
        "native_json_mode": bool(getattr(provider, "_native_json_mode", False)),
    }


def restore_provider(
    config: dict[str, Any] | None, *, fallback: LLMProvider | None = None
) -> LLMProvider:
    if not config:
        return fallback or MockLLMProvider()
    provider_id = str(config.get("provider_id") or config.get("provider_backend") or "mock")
    model = str(config.get("model") or "")
    thinking = bool(config.get("thinking", True))
    max_output = int(config.get("max_output") or 2000)
    temperature = float(config.get("temperature") or 0.1)
    context_limit = config.get("context_limit")
    base_url = _safe_base_url(str(config.get("base_url") or ""))
    if provider_id == "cursor_external":
        from app.ai.external_controller import ExternalControllerProvider

        return ExternalControllerProvider(model_name=model or "cursor-selected-model")
    if provider_id == "mock":
        return MockLLMProvider(model_name=model or "mock-v1")
    settings = get_settings()
    overlay = settings.model_copy(
        update={
            "ai_provider": provider_id
            if provider_id in {"mlx", "local", "ollama", "openai", "anthropic", "openai_compatible"}
            else settings.ai_provider,
            "ai_model": model or settings.ai_model,
            "ai_max_output_tokens": max_output or settings.ai_max_output_tokens,
            "ai_temperature": temperature,
            "ai_thinking_enabled": thinking,
            "mlx_base_url": base_url or settings.mlx_base_url,
            "ai_max_context_tokens": int(context_limit or settings.ai_max_context_tokens),
        }
    )
    try:
        from app.ai.factory import create_provider

        return create_provider(overlay)
    except Exception:
        return fallback or MockLLMProvider(model_name=model or "mock-v1")


def _safe_base_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return text
    return ""
