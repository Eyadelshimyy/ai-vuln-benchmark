"""AI provider status and test endpoints (v1.1.0).

GET  /api/v1/ai/status   — probe current AI provider
POST /api/v1/ai/test     — send a test prompt
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter

from app.ai.factory import check_ai_health
from app.core.config import settings
from app.schemas.discovery import AIStatusResponse, AITestRequest, AITestResponse

router = APIRouter(prefix="/ai", tags=["AI"])
logger = logging.getLogger(__name__)


@router.get("/status", response_model=AIStatusResponse)
async def get_ai_status() -> AIStatusResponse:
    """Probe the configured AI provider and return its health status.

    API keys are never included in the response.
    """
    health = await check_ai_health(settings)
    catalog = None
    try:
        from app.domain.language import LanguageCapability
        from app.plugins import get_plugin_catalog

        catalog = get_plugin_catalog()
        analyzers = [
            adapter.language_id
            for adapter in catalog.languages.all_adapters()
            if adapter.supports(LanguageCapability.SECURITY_ANALYSIS)
            or adapter.supports(LanguageCapability.STATIC_ANALYSIS)
        ]
    except Exception:
        analyzers = []
    return AIStatusResponse(
        provider=health.provider,
        model=health.model,
        is_local=health.is_local,
        reachable=health.reachable,
        configured=health.configured,
        model_available=health.model_available,
        error=health.error,
        capabilities=health.capabilities,
        thinking_enabled=settings.ai_thinking_enabled,
        language_analyzers=analyzers,
        security_analysis_status="available",
    )


@router.post("/test", response_model=AITestResponse)
async def test_ai_provider(request: AITestRequest) -> AITestResponse:
    """Send a simple test prompt to the configured AI provider.

    The prompt must be 500 characters or fewer.
    API keys are never included in the response.
    """
    from app.ai.factory import create_provider

    provider = create_provider(settings)
    system_prompt = "You are a helpful assistant. Reply concisely."
    # User content is not repository data so the prompt-injection fence is not required here,
    # but we still validate length to prevent abuse.
    user_message = request.prompt[:500]

    start = time.monotonic()
    try:
        resp = await provider.generate_structured(system_prompt, user_message)
        duration = time.monotonic() - start
        return AITestResponse(
            provider=resp.provider,
            model=resp.model,
            response=resp.content[:1000] if resp.content else "",
            duration_seconds=round(duration, 3),
            error=resp.error,
        )
    except Exception as exc:
        duration = time.monotonic() - start
        logger.warning("AI test call failed: %s", exc)
        return AITestResponse(
            provider=settings.ai_provider,
            model=settings.ai_model,
            response="",
            duration_seconds=round(duration, 3),
            error=str(exc),
        )
