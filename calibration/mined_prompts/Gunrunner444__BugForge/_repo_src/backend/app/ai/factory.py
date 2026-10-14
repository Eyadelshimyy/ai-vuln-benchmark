"""AI provider factory and health-check helpers.

Core code must not construct vendor providers directly. Lookups go through
the plugin catalog so new backends can be registered without editing this
module's conditionals.
"""

from __future__ import annotations

import logging

from app.ai.health import AIHealthStatus
from app.ai.provider import LLMProvider
from app.core.config import Settings
from app.plugins.errors import AdapterError, AdapterNotFoundError

logger = logging.getLogger(__name__)

# Re-export so existing `from app.ai.factory import AIHealthStatus` imports keep working.
__all__ = ["AIHealthStatus", "check_ai_health", "create_provider"]


def create_provider(settings: Settings) -> LLMProvider:
    """Return the configured LLM provider instance via the adapter registry."""
    from app.plugins import get_plugin_catalog

    catalog = get_plugin_catalog()
    try:
        return catalog.ai_providers.create(settings.ai_provider, settings)
    except AdapterNotFoundError as exc:
        raise ValueError(str(exc)) from exc


async def check_ai_health(settings: Settings) -> AIHealthStatus:
    """Probe the configured AI provider and return a health report.

    Never includes API keys in the returned object.
    """
    try:
        provider = create_provider(settings)
    except (ValueError, AdapterError) as exc:
        return AIHealthStatus(
            provider=settings.ai_provider,
            model=settings.ai_model,
            is_local=settings.is_local_ai(),
            reachable=False,
            configured=False,
            model_available=None,
            error=str(exc),
        )

    status = await provider.health()
    # Report the configured names so callers see `ollama` rather than the
    # underlying transport's identifier.
    status.provider = settings.ai_provider
    status.model = settings.ai_model
    status.is_local = settings.is_local_ai()
    return status
