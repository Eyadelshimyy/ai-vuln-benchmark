"""
OpenAI-compatible LLM provider using httpx directly.

Supports the OpenAI API and any compatible endpoint (Azure OpenAI,
local LLM servers, etc.) by overriding ai_base_url in config.

Security:
  - API key comes only from settings (environment variable); never logged.
  - Prompts are sent over TLS.
  - Repository content is labeled as untrusted data (see PromptBuilder).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app.ai.health import AIHealthStatus, probe_openai_compatible
from app.ai.provider import (
    AICapabilities,
    AIUsage,
    DebuggingRequest,
    HypothesisResult,
    LLMProvider,
    ProviderResponse,
    StructuredTextResponse,
    TestGenerationResponse,
)

logger = logging.getLogger(__name__)

_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

_VALID_CONFIDENCE_LABELS = frozenset(
    {"confirmed", "highly_likely", "likely", "possible", "insufficient_evidence"}
)


class OpenAIProvider(LLMProvider):
    """Calls the OpenAI chat-completions API with structured JSON output."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "",
        max_tokens: int = 2_000,
        temperature: float = 0.1,
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or _OPENAI_API_URL
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "openai"

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
            native_json_response_format=True,
        )

    async def health(self) -> AIHealthStatus:
        return await probe_openai_compatible(
            provider=self.provider_name,
            model=self._model,
            base_url=self._base_url,
            api_key=self._api_key,
            is_local=False,
            configured=bool(self._api_key),
        )

    async def generate_tests(self, system_prompt: str, user_message: str) -> TestGenerationResponse:
        start = time.monotonic()
        try:
            raw_json, usage = await self._call_api(system_prompt, user_message)
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
            content=raw,
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            duration_seconds=time.monotonic() - start,
        )

    async def generate_patch(self, system_prompt: str, user_message: str) -> StructuredTextResponse:
        return await self.generate_structured(system_prompt, user_message)

    async def analyze(self, request: DebuggingRequest) -> ProviderResponse:
        from app.ai.prompt_builder import PromptBuilder

        builder = PromptBuilder()
        prompt = builder.build(request)

        start = time.monotonic()
        raw_json = ""

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
                logger.warning("OpenAI attempt %d failed: %s", attempt + 1, exc)
                usage = AIUsage()
        else:
            usage = AIUsage()

        duration = time.monotonic() - start
        hypotheses, parse_error = _parse_hypotheses(raw_json, request.max_hypotheses)

        return ProviderResponse(
            hypotheses=hypotheses,
            provider=self.provider_name,
            model=self._model,
            usage=usage,
            duration_seconds=duration,
            raw_json=raw_json,
            error=parse_error,
        )

    async def _call_api(self, system_msg: str, user_msg: str) -> tuple[str, AIUsage]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(self._base_url, json=payload, headers=headers)

        if response.status_code == 429:
            raise httpx.HTTPStatusError("Rate limited", request=response.request, response=response)
        if response.status_code == 401:
            raise httpx.HTTPStatusError(
                "Unauthorized — check AI_API_KEY", request=response.request, response=response
            )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        raw_usage = data.get("usage", {})
        usage = AIUsage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
        )
        return content, usage


def _parse_hypotheses(raw_json: str, max_count: int) -> tuple[list[HypothesisResult], str | None]:
    """Parse the AI's JSON response into typed hypotheses. Returns (list, error_or_None)."""
    if not raw_json:
        return [], "Empty response from AI provider"
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return [], f"Invalid JSON from AI provider: {exc}"

    raw_list = data.get("hypotheses", [])
    if not isinstance(raw_list, list):
        return [], "AI response missing 'hypotheses' array"

    hypotheses: list[HypothesisResult] = []
    for item in raw_list[:max_count]:
        if not isinstance(item, dict):
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            label = item.get("confidence_label", "possible")
            if label not in _VALID_CONFIDENCE_LABELS:
                label = "possible"

            hypotheses.append(
                HypothesisResult(
                    root_cause=str(item.get("root_cause", "")),
                    confidence=confidence,
                    confidence_label=label,
                    affected_files=_str_list(item.get("affected_files", [])),
                    affected_symbols=_str_list(item.get("affected_symbols", [])),
                    evidence_summary=_str_list(item.get("evidence_summary", [])),
                    contradictory_evidence=_str_list(item.get("contradictory_evidence", [])),
                    reproduction_strategy=str(item.get("reproduction_strategy", "")),
                    recommended_tests=_str_list(item.get("recommended_tests", [])),
                    explanation=str(item.get("explanation", "")),
                )
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed hypothesis: %s", exc)

    return hypotheses, None


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v is not None]
