"""
Anthropic Claude provider using the Messages API.

Uses httpx directly — no SDK dependency.
API reference: https://docs.anthropic.com/en/api/messages

Security: API key is never logged. Only forward explicitly listed env overrides to analyzed repositories.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.ai.health import AIHealthStatus, default_capabilities
from app.ai.openai_provider import _parse_hypotheses
from app.ai.provider import (
    AICapabilities,
    AIUsage,
    DebuggingRequest,
    LLMProvider,
    ProviderResponse,
    StructuredTextResponse,
    TestGenerationResponse,
)

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """Calls Anthropic's Messages API (Claude models)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 2_000,
        temperature: float = 0.1,
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    async def is_available(self) -> bool:
        return bool(self._api_key)

    def capabilities(self) -> AICapabilities:
        return AICapabilities(
            chat=True,
            structured_output=True,
            structured_generation=True,
            text_generation=True,
            tool_calls=False,
            thinking=False,
            thinking_can_disable=True,
            max_output_tokens=self._max_tokens,
        )

    async def health(self) -> AIHealthStatus:
        configured = bool(self._api_key)
        return AIHealthStatus(
            provider=self.provider_name,
            model=self._model,
            is_local=False,
            reachable=configured,
            configured=configured,
            model_available=None,
            error=None if configured else "AI_API_KEY not configured",
            capabilities=default_capabilities(),
        )

    async def generate_tests(self, system_prompt: str, user_message: str) -> TestGenerationResponse:
        start = time.monotonic()
        try:
            raw, usage = await self._call_api(system_prompt, user_message)
            raw_json = _extract_json_from_text(raw)
        except Exception as exc:
            return TestGenerationResponse(
                candidates_json="",
                provider=self.provider_name,
                model=self._model,
                usage=AIUsage(),
                duration_seconds=time.monotonic() - start,
                error=str(exc),
            )
        return TestGenerationResponse(
            candidates_json=raw_json,
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            duration_seconds=time.monotonic() - start,
        )

    async def generate_structured(
        self, system_prompt: str, user_message: str
    ) -> StructuredTextResponse:
        start = time.monotonic()
        try:
            raw, usage = await self._call_api(system_prompt, user_message)
            content = _extract_json_from_text(raw)
        except Exception as exc:
            return StructuredTextResponse(
                content="",
                provider=self.provider_name,
                model=self._model,
                usage=AIUsage(),
                duration_seconds=time.monotonic() - start,
                error=str(exc),
            )
        return StructuredTextResponse(
            content=content,
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            duration_seconds=time.monotonic() - start,
        )

    async def generate_patch(self, system_prompt: str, user_message: str) -> StructuredTextResponse:
        return await self.generate_structured(system_prompt, user_message)

    async def analyze(self, request: DebuggingRequest) -> ProviderResponse:
        from app.ai.prompt_builder import PromptBuilder

        prompt = PromptBuilder().build(request)
        start = time.monotonic()
        raw_json = ""
        usage = AIUsage()

        for attempt in range(self._max_retries + 1):
            try:
                raw_json, usage = await self._call_api(prompt.system_message, prompt.user_message)
                break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self._max_retries:
                    return ProviderResponse(
                        hypotheses=[],
                        provider=self.provider_name,
                        model=self._model,
                        usage=AIUsage(),
                        duration_seconds=time.monotonic() - start,
                        error=f"Network error after {self._max_retries + 1} attempts: {exc}",
                    )
                logger.warning("Anthropic attempt %d failed: %s", attempt + 1, exc)

        hypotheses, parse_error = _parse_hypotheses(raw_json, request.max_hypotheses)
        return ProviderResponse(
            hypotheses=hypotheses,
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            duration_seconds=time.monotonic() - start,
            raw_json=raw_json,
            error=parse_error,
        )

    async def _call_api(self, system_msg: str, user_msg: str) -> tuple[str, AIUsage]:
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system_msg,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": self._temperature,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(_ANTHROPIC_API_URL, json=payload, headers=headers)

        if response.status_code == 429:
            raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)
        if response.status_code in (401, 403):
            raise httpx.HTTPStatusError(
                "Unauthorized — check ANTHROPIC_API_KEY",
                request=response.request,
                response=response,
            )
        response.raise_for_status()

        data = response.json()
        # Claude returns content as a list of content blocks
        content_blocks = data.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        raw_usage = data.get("usage", {})
        usage = AIUsage(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
        )

        # Claude doesn't natively return JSON objects — extract JSON from text
        extracted = _extract_json_from_text(text)
        return extracted, usage


def _extract_json_from_text(text: str) -> str:
    """Extract a JSON object from Claude's text response."""
    text = text.strip()
    # Try direct parse first
    if text.startswith("{"):
        return text
    # Look for ```json ... ``` blocks
    import re

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Find first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text
