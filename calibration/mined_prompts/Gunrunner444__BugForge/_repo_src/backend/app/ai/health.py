"""AI provider health types and OpenAI-compatible HTTP probes.

API keys are never included in returned objects or log messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class AIHealthStatus:
    """Result of a provider health check."""

    provider: str
    model: str
    is_local: bool
    reachable: bool
    configured: bool
    model_available: bool | None  # None = unknown (couldn't verify)
    error: str | None = None
    capabilities: list[str] | None = None


_DEFAULT_CAPABILITIES = [
    "analysis",
    "test_generation",
    "patch_generation",
    "completion",
]


def default_capabilities() -> list[str]:
    return list(_DEFAULT_CAPABILITIES)


async def probe_openai_compatible(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    is_local: bool,
    configured: bool,
) -> AIHealthStatus:
    """Cheap GET /models probe used by OpenAI and local OpenAI-compatible backends."""
    reachable = False
    model_available: bool | None = None
    error: str | None = None

    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            models_url = base_url.rstrip("/").removesuffix("/chat/completions") + "/models"
            resp = await client.get(models_url, headers=headers)
            if resp.status_code in {200, 401, 403}:
                reachable = True
                if resp.status_code == 200:
                    data = resp.json()
                    model_ids: list[str] = []
                    for entry in data.get("data", []):
                        if isinstance(entry, dict):
                            mid = entry.get("id") or entry.get("name") or ""
                            model_ids.append(str(mid))
                    model_available = any(
                        mid == model or mid.startswith(model) or model.startswith(mid.split(":")[0])
                        for mid in model_ids
                    )
                    if model_available is False and not model_ids:
                        model_available = None
            else:
                reachable = True
                error = f"unexpected_status:{resp.status_code}"
    except httpx.ConnectError:
        error = f"connection_refused:{base_url}"
    except httpx.TimeoutException:
        error = f"timeout:{base_url}"
    except Exception as exc:
        error = f"error:{exc}"

    return AIHealthStatus(
        provider=provider,
        model=model,
        is_local=is_local,
        reachable=reachable,
        configured=configured,
        model_available=model_available,
        error=error,
        capabilities=default_capabilities() if reachable else None,
    )
